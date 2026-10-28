"""Generate MUSA BF16 references on a fixed Qwen3.5 token trajectory."""

import argparse
import gc
import json
import os
from pathlib import Path

import torch

from examples.musa.aoti_model_acceptance import _write_tensor
from examples.musa.aoti_qwen35_export import load_model, STATE_SUFFIXES
from examples.musa.aoti_qwen35_validate import file_hash


def load_hf_model(checkpoint_dir):
    from safetensors.torch import load_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = Qwen3_5TextConfig(
        **json.loads((checkpoint_dir / "config.json").read_text())["text_config"]
    )
    original = load_file(
        str(checkpoint_dir / "model.safetensors-00001-of-00001.safetensors")
    )
    weights = {
        name.replace("model.language_model.", "model.", 1): value
        for name, value in original.items()
        if name.startswith("model.language_model.")
    }
    if not config.tie_word_embeddings:
        raise ValueError("Expected tied Qwen3.5 text embeddings")
    model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
        None,
        config=config,
        state_dict=weights,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        output_loading_info=True,
    )
    if any(loading_info.values()):
        raise ValueError(f"Incomplete HF checkpoint load: {loading_info}")
    model = model.eval().to(device="musa")
    if any(value.dtype != torch.bfloat16 for value in model.parameters()):
        raise ValueError("Expected BF16 HF parameters")
    if any(
        value.device.type != "musa" for value in (*model.parameters(), *model.buffers())
    ):
        raise ValueError("HF parameters and buffers must reside on MUSA")
    rotary = model.model.rotary_emb
    if (
        rotary.inv_freq.dtype != torch.float32
        or rotary.original_inv_freq.dtype != torch.float32
    ):
        raise ValueError("HF RoPE inverse frequencies must remain FP32")
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError("HF text embeddings must remain tied")
    return model


@torch.no_grad()
def generate(args, report):
    import transformers

    if os.environ.get("MUSA_LAUNCH_BLOCKING") != "1":
        raise ValueError("Use MUSA_LAUNCH_BLOCKING=1 for reference generation")
    source_hash = file_hash(args.trajectories)
    source = json.loads(args.trajectories.read_text())
    if source["status"] != "complete" or source["dtype"] != "bf16":
        raise ValueError("Expected complete BF16 input trajectories")
    tracked = {
        str(path): file_hash(path)
        for path in sorted(args.checkpoint_dir.iterdir())
        if path.suffix in (".json", ".safetensors", ".jinja")
    }
    if {Path(path).name: digest for path, digest in tracked.items()} != {
        Path(path).name: digest for path, digest in source["checkpoint_hashes"].items()
    }:
        raise ValueError("Trajectory checkpoint hashes do not match")
    fixtures = []
    for original in source["fixtures"]:
        tokens = original["token_ids"]
        name = original["name"]
        if not name or name in (".", "..") or Path(name).name != name:
            raise ValueError("Fixture name must be one directory component")
        if not 1 <= len(tokens) <= 64 or not 1 <= original["prompt_token_count"] <= len(
            tokens
        ):
            raise ValueError("Fixture must fit the exported context")
        if any(type(token) is not int or not 0 <= token < 248320 for token in tokens):
            raise ValueError("Invalid token ID")
        fixture = {
            "name": name,
            "token_ids": tokens,
            "prompt_token_count": original["prompt_token_count"],
            "hashes": {},
            "hf_predictions": [],
            "et_predictions": [],
        }
        if "prompt" in original:
            fixture["prompt"] = original["prompt"]
        (args.output_dir / name).mkdir()
        fixtures.append(fixture)
    if not fixtures:
        raise ValueError("Expected at least one fixture")
    report.update(
        dtype="bf16",
        context_length=64,
        teacher_forced=True,
        checkpoint_hashes=tracked,
        trajectory_source=str(args.trajectories),
        trajectory_sha256=source_hash,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        device=torch.musa.get_device_name(0),
        hf_loader="from_pretrained(dtype=bfloat16).to(device=musa)",
        attention_implementation="eager",
        launch_blocking=True,
        allow_tf32=False,
        prefill="token_by_token",
        fixtures=fixtures,
    )
    model = load_hf_model(args.checkpoint_dir)
    report["hf_buffer_dtypes"] = {
        name: str(value.dtype) for name, value in model.named_buffers()
    }
    report["hf_parameters"] = sum(value.numel() for value in model.parameters())
    for fixture in fixtures:
        cache = None
        for position, token in enumerate(fixture["token_ids"]):
            output = model(
                input_ids=torch.tensor([[token]], device="musa"),
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            path = args.output_dir / fixture["name"] / f"hf-{position}.bin"
            logits = _write_tensor(output.logits, path).flatten()
            if not torch.isfinite(logits).all().item():
                raise ValueError("Nonfinite HF logits")
            fixture["hashes"][path.name] = file_hash(path)
            fixture["hf_predictions"].append(logits.argmax().item())
            print(f"hf {fixture['name']} {position}", flush=True)
    del model, output, cache, logits
    gc.collect()
    torch.musa.empty_cache()
    model = load_model(args.checkpoint_dir, 64, torch.bfloat16)
    for fixture in fixtures:
        for name, buffer in model.named_buffers():
            if name.endswith(STATE_SUFFIXES):
                buffer.zero_()
        for position, token in enumerate(fixture["token_ids"]):
            output = model(
                torch.tensor([[token]], device="musa"),
                {"input_pos": torch.tensor([position], device="musa")},
            )
            path = args.output_dir / fixture["name"] / f"et-{position}.bin"
            logits = _write_tensor(output, path).flatten()
            if not torch.isfinite(logits).all().item():
                raise ValueError("Nonfinite ET eager logits")
            fixture["hashes"][path.name] = file_hash(path)
            fixture["et_predictions"].append(logits.argmax().item())
            print(f"et {fixture['name']} {position}", flush=True)
    if tracked != {
        path: file_hash(Path(path)) for path in tracked
    } or source_hash != file_hash(args.trajectories):
        raise ValueError("Reference inputs changed during generation")
    report["status"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.trajectories = args.trajectories.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    report = {"status": "running"}
    try:
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            generate(args, report)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (args.output_dir / "manifest.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )


if __name__ == "__main__":
    main()
