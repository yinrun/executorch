"""Generate independent HF BF16 trajectories and replay them with ET eager."""

import argparse
import gc
import json
import os
from pathlib import Path

import torch

from examples.musa.aoti_model_acceptance import _write_tensor
from examples.musa.aoti_qwen35_export import load_model, STATE_SUFFIXES
from examples.musa.aoti_qwen35_reference import load_hf_model
from examples.musa.aoti_qwen35_validate import file_hash, VOCAB_SIZE


def validate_prompts(entries):
    if not isinstance(entries, list) or not entries:
        raise ValueError("Expected a nonempty list of prompts")
    names = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each prompt must be an object")
        name, prompt = entry.get("name"), entry.get("prompt")
        if (
            not isinstance(name, str)
            or not name
            or name in (".", "..")
            or Path(name).name != name
            or "\\" in name
            or "\0" in name
            or name in names
        ):
            raise ValueError("Prompt names must be unique directory components")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt text must be nonempty")
        names.add(name)
    return entries


def validate_prompt_tokens(tokens, context_length):
    if not isinstance(tokens, list) or not 1 <= len(tokens) < context_length:
        raise ValueError("Prompt must leave space for generation within the context")
    if any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in tokens):
        raise ValueError("Invalid prompt token ID")


def generation_stop_reason(predicted, eos_token_id, position, context_length):
    if predicted == eos_token_id:
        return "eos"
    if position == context_length - 1:
        return "context_limit"
    return None


def save_logits(output, path):
    if output.dtype != torch.bfloat16 or output.numel() != VOCAB_SIZE:
        raise ValueError("Expected a complete BF16 vocabulary row")
    logits = _write_tensor(output, path).flatten()
    if not torch.isfinite(logits).all().item():
        raise ValueError("Nonfinite reference logits")
    return logits.argmax().item(), file_hash(path)


@torch.no_grad()
def generate(args, report):
    import torch_musa
    import transformers
    from transformers import AutoTokenizer

    if os.environ.get("MUSA_LAUNCH_BLOCKING") != "1":
        raise ValueError("Use MUSA_LAUNCH_BLOCKING=1 for reference generation")
    if not torch_musa.is_available():
        raise ValueError("MUSA device is required")
    tracked = {
        str(path): file_hash(path)
        for path in sorted(args.checkpoint_dir.iterdir())
        if path.suffix in (".json", ".safetensors", ".jinja")
    }
    prompt_hash = file_hash(args.prompts)
    entries = validate_prompts(json.loads(args.prompts.read_text()))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, local_files_only=True)
    eos_token_id = tokenizer.eos_token_id
    if type(eos_token_id) is not int or not 0 <= eos_token_id < VOCAB_SIZE:
        raise ValueError("Expected one valid EOS token ID")
    fixtures = []
    for entry in entries:
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": entry["prompt"]}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        validate_prompt_tokens(tokens, args.context_length)
        fixtures.append(
            dict(
                name=entry["name"],
                prompt=entry["prompt"],
                token_ids=list(tokens),
                prompt_token_count=len(tokens),
                hashes={},
                hf_predictions=[],
                et_predictions=[],
                generated_token_ids=[],
            )
        )
    report.update(
        dtype="bf16",
        context_length=args.context_length,
        teacher_forced=False,
        reference_generation="independent_hf_greedy",
        checkpoint_hashes=tracked,
        prompts_source=str(args.prompts),
        prompts_sha256=prompt_hash,
        eos_token_id=eos_token_id,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        device=torch.musa.get_device_name(0),
        hf_loader="from_pretrained(dtype=bfloat16).to(device=musa)",
        attention_implementation="eager",
        launch_blocking=True,
        allow_tf32=False,
        prefill="token_by_token",
        fixtures=fixtures,
        eager_projection_fusion="disabled",
        eager_byte_equal=0,
        eager_steps=0,
        first_eager_difference=None,
    )
    torch.set_num_threads(4)
    with torch.backends.mudnn.flags(allow_tf32=False):
        model = load_hf_model(args.checkpoint_dir)
        report["hf_buffer_dtypes"] = {
            name: str(value.dtype) for name, value in model.named_buffers()
        }
        report["hf_parameters"] = sum(value.numel() for value in model.parameters())
        for fixture in fixtures:
            destination = args.output_dir / fixture["name"]
            destination.mkdir()
            cache = None
            position = 0
            while position < len(fixture["token_ids"]):
                output = model(
                    input_ids=torch.tensor(
                        [[fixture["token_ids"][position]]], device="musa"
                    ),
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values
                path = destination / f"hf-{position}.bin"
                predicted, digest = save_logits(output.logits, path)
                fixture["hashes"][path.name] = digest
                fixture["hf_predictions"].append(predicted)
                if position >= fixture["prompt_token_count"] - 1:
                    fixture["generated_token_ids"].append(predicted)
                    stop = generation_stop_reason(
                        predicted, eos_token_id, position, args.context_length
                    )
                    if stop is None:
                        fixture["token_ids"].append(predicted)
                    else:
                        fixture["stop_reason"] = stop
                print(f"hf {fixture['name']} {position} {predicted}", flush=True)
                position += 1
            fixture["generated_text"] = tokenizer.decode(
                fixture["generated_token_ids"], skip_special_tokens=True
            )
        del model, output, cache
        gc.collect()
        torch.musa.empty_cache()
        model = load_model(args.checkpoint_dir, args.context_length, torch.bfloat16)
        if args.fuse_projections:
            from examples.musa.aoti_qwen35_fusion import (
                FUSION_NAME,
                fuse_qwen35_projections,
            )

            model = fuse_qwen35_projections(model)
            report["eager_projection_fusion"] = FUSION_NAME
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
                predicted, digest = save_logits(output, path)
                fixture["hashes"][path.name] = digest
                fixture["et_predictions"].append(predicted)
                equal = digest == fixture["hashes"][f"hf-{position}.bin"]
                report["eager_steps"] += 1
                report["eager_byte_equal"] += equal
                if not equal and report["first_eager_difference"] is None:
                    report["first_eager_difference"] = dict(
                        fixture=fixture["name"], position=position
                    )
                print(f"et {fixture['name']} {position} byte_equal={equal}", flush=True)
    if tracked != {path: file_hash(Path(path)) for path in tracked}:
        raise ValueError("Checkpoint changed during reference generation")
    if prompt_hash != file_hash(args.prompts):
        raise ValueError("Prompts changed during reference generation")
    report["all_eager_byte_equal"] = (
        report["eager_steps"] == report["eager_byte_equal"]
    )
    report["status"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--context-length", type=int, choices=(64, 128, 256), required=True)
    parser.add_argument("--fuse-projections", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.prompts = args.prompts.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "generator_sha256": file_hash(Path(__file__))}
    try:
        generate(args, report)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (args.output_dir / "manifest.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
    if not report["all_eager_byte_equal"]:
        raise SystemExit("ET eager differs from HF; diagnostics retained in manifest")


if __name__ == "__main__":
    main()
