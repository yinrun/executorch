"""Export local Qwen3.5-0.8B text weights with persistent MUSA decode state."""

import argparse
import gc
import hashlib
import json
import operator
import os
from pathlib import Path

import torch

from executorch.backends.musa import (
    MusaBackend,
    MusaPartitioner,
    export_with_musa_sdpa_decomposition,
)
from executorch.examples.models.llama.attention import (
    AttentionGatedDeltaNet,
    AttentionMHA,
    SDPA,
)
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.qwen3_5.convert_weights import (
    load_checkpoint,
    qwen_3_5_to_meta,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program
from examples.musa.aoti_model_acceptance import _serialization_schema, _write_tensor


STATE_SUFFIXES = ("conv_state", "recurrent_state", "k_cache", "v_cache")


class Qwen35MusaSDPA(SDPA):
    def forward(self, input_pos, query, key, value, batch_size, sequence_length, mask):
        key = key.repeat_interleave(self.n_rep, dim=1)
        value = value.repeat_interleave(self.n_rep, dim=1)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores * (self.head_dim**-0.5)
        if mask is not None:
            scores = (
                scores.masked_fill(~mask, float("-inf"))
                if mask.dtype == torch.bool
                else scores + mask
            )
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        output = torch.matmul(probabilities, value)
        return output.transpose(1, 2).reshape(batch_size, sequence_length, self.dim)


def load_model(checkpoint_dir: Path, context_length: int, dtype: torch.dtype):
    if context_length < 2:
        raise ValueError("Context length must be at least two")
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Qwen3.5 bring-up supports fp32 or bf16")
    upstream = json.loads((checkpoint_dir / "config.json").read_text())["text_config"]
    params = json.loads(
        (
            Path(__file__).parents[1] / "models/qwen3_5/config/0_8b_config.json"
        ).read_text()
    )
    mapping = {
        "hidden_size": "dim",
        "intermediate_size": "hidden_dim",
        "num_hidden_layers": "n_layers",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "n_kv_heads",
        "head_dim": "head_dim",
        "vocab_size": "vocab_size",
        "layer_types": "layer_types",
        "rms_norm_eps": "norm_eps",
        "linear_conv_kernel_dim": "linear_conv_kernel_dim",
        "linear_key_head_dim": "linear_key_head_dim",
        "linear_value_head_dim": "linear_value_head_dim",
        "linear_num_key_heads": "linear_num_key_heads",
        "linear_num_value_heads": "linear_num_value_heads",
    }
    if any(upstream[name] != params[target] for name, target in mapping.items()):
        raise ValueError("Checkpoint configuration does not match Qwen3.5-0.8B")
    if (
        not upstream["tie_word_embeddings"]
        or not upstream["attn_output_gate"]
        or upstream["attention_bias"]
        or upstream["hidden_act"] != "silu"
        or upstream["rope_parameters"]["rope_type"] != "default"
        or upstream["rope_parameters"]["rope_theta"] != params["rope_theta"]
        or upstream["rope_parameters"]["partial_rotary_factor"]
        != params["partial_rotary_factor"]
        or context_length > upstream["max_position_embeddings"]
    ):
        raise ValueError("Unsupported Qwen3.5 text configuration")
    weights = qwen_3_5_to_meta(load_checkpoint(str(checkpoint_dir)))
    args = ModelArgs(
        **params,
        max_seq_len=context_length,
        max_context_len=context_length,
        use_kv_cache=True,
    )
    with torch.device("meta"):
        model = construct_transformer(args)
    missing, unexpected = model.load_state_dict(weights, strict=False, assign=True)
    parameter_names = set(dict(model.named_parameters()))
    if unexpected or set(missing) - set(dict(model.named_buffers())):
        raise ValueError(f"Checkpoint mismatch: {missing=}, {unexpected=}")
    if parameter_names - set(weights):
        raise ValueError("Checkpoint is missing text model parameters")
    for name, value in list(model.named_buffers()):
        if value.is_meta:
            if not name.endswith(STATE_SUFFIXES):
                raise ValueError(f"Unexpected uninitialized model buffer: {name}")
            owner, field = name.rsplit(".", 1)
            model.get_submodule(owner).register_buffer(
                field, torch.zeros(value.shape, dtype=value.dtype, device="cpu")
            )
    model.output.weight = model.tok_embeddings.weight
    model = model.eval().to(device="musa", dtype=dtype)
    for layer in model.layers:
        if isinstance(layer.attention, AttentionGatedDeltaNet):
            layer.attention.recurrent_state = layer.attention.recurrent_state.float()
        elif dtype == torch.bfloat16 and isinstance(layer.attention, AttentionMHA):
            previous = layer.attention.SDPA
            layer.attention.SDPA = Qwen35MusaSDPA(
                previous.dim,
                previous.head_dim,
                previous.n_rep,
                previous.max_context_len,
            ).eval()
    del weights
    gc.collect()
    if any(value.device.type != "musa" for value in model.parameters()):
        raise RuntimeError("Model parameters must reside on MUSA")
    if any(value.device.type != "musa" for value in model.buffers()):
        raise RuntimeError("Model buffers must reside on MUSA")
    return model


@torch.no_grad()
def export_model(args, report):
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    model = load_model(args.checkpoint_dir, args.context_length, dtype)
    report["parameters"] = sum(value.numel() for value in model.parameters())
    report["checkpoint_hashes"] = {}
    for path in sorted(args.checkpoint_dir.iterdir()):
        if path.suffix in (".json", ".safetensors", ".jinja"):
            with path.open("rb") as checkpoint_file:
                report["checkpoint_hashes"][path.name] = hashlib.file_digest(
                    checkpoint_file, "sha256"
                ).hexdigest()
    tokenizer = None
    if args.prompt is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint_dir, local_files_only=True
        )
        args.token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not args.token_ids or any(
            not isinstance(token, int) or not 0 <= token < 248320
            for token in args.token_ids
        ):
            raise ValueError("Tokenizer must return a nonempty list of valid token IDs")
        if len(args.token_ids) + args.generate_tokens - 1 > args.context_length:
            raise ValueError("Prompt and requested generation exceed context length")
        report.update(
            prompt=args.prompt,
            prompt_token_count=len(args.token_ids),
            generated_token_ids=[],
        )
    prompt_length = len(args.token_ids)
    report["token_ids"] = args.token_ids
    position = 0
    while position < len(args.token_ids):
        token = args.token_ids[position]
        token_ids = torch.tensor([[token]], dtype=torch.int64, device="musa")
        input_pos = torch.tensor([position], dtype=torch.int64, device="musa")
        expected = model(token_ids, {"input_pos": input_pos})
        if not torch.isfinite(expected).all().item():
            raise RuntimeError("MUSA eager reference produced non-finite logits")
        _write_tensor(expected, args.output_dir / f"eager-{position}.bin")
        predicted = expected.argmax().item()
        report["reference_tokens"].append(predicted)
        if tokenizer is not None and position >= prompt_length - 1:
            report["generated_token_ids"].append(predicted)
            if (
                predicted != tokenizer.eos_token_id
                and len(report["generated_token_ids"]) < args.generate_tokens
            ):
                args.token_ids.append(predicted)
        position += 1
    if tokenizer is not None:
        report["generated_text"] = tokenizer.decode(
            report["generated_token_ids"], skip_special_tokens=True
        )
        print(report["generated_text"], flush=True)
    for name, buffer in model.named_buffers():
        if name.endswith(STATE_SUFFIXES):
            buffer.zero_()
    token_ids = torch.tensor([[args.token_ids[0]]], dtype=torch.int64, device="musa")
    input_pos = torch.zeros(1, dtype=torch.int64, device="musa")
    exported = export_with_musa_sdpa_decomposition(
        model, (token_ids, {"input_pos": input_pos})
    ).run_decompositions({})
    mutated = set(exported.graph_signature.buffers_to_mutate.values())
    expected_states = {
        name for name, buffer in model.named_buffers() if name.endswith(STATE_SUFFIXES)
    }
    if mutated != expected_states or len(mutated) != 48:
        raise RuntimeError(
            f"Missing persistent Qwen3.5 states: {mutated ^ expected_states}"
        )
    report["mutated_buffers"] = sorted(mutated)
    report["graph_nodes"] = len(list(exported.graph.nodes))
    (args.output_dir / "exported-graph.txt").write_text(str(exported.graph))
    print(
        f"Lowering {report['graph_nodes']} nodes with {len(mutated)} mutable states",
        flush=True,
    )
    specs = [MusaBackend.generate_method_name_compile_spec("forward")]
    with _serialization_schema():
        manager = to_edge_transform_and_lower(
            exported, partitioner=[MusaPartitioner(specs)]
        )
        calls = [
            node
            for node in manager.exported_program().graph.nodes
            if node.op == "call_function"
        ]
        delegates = [
            node for node in calls if "executorch_call_delegate" in str(node.target)
        ]
        if len(delegates) != 1 or any(
            node not in delegates and node.target is not operator.getitem
            for node in calls
        ):
            raise RuntimeError(
                "Qwen3.5 requires one MUSA delegate and no host tensor ops"
            )
        pte = Path(
            save_pte_program(manager.to_executorch(), "qwen35", str(args.output_dir))
        )
    ptd = args.output_dir / "aoti_musa_blob.ptd"
    report.update(pte=str(pte), ptd=str(ptd), delegates=len(delegates))
    report["artifacts"] = {}
    reference_paths = [
        args.output_dir / f"eager-{position}.bin"
        for position in range(len(args.token_ids))
    ]
    for path in (pte, ptd, *reference_paths):
        with path.open("rb") as artifact:
            report["artifacts"][path.name] = hashlib.file_digest(
                artifact, "sha256"
            ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--context-length", type=int, default=64)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--token-ids", type=int, nargs="+", default=[9707, 11, 2082])
    inputs.add_argument("--prompt")
    parser.add_argument("--generate-tokens", type=int, default=16)
    args = parser.parse_args()
    if not args.token_ids or len(args.token_ids) > args.context_length:
        parser.error("Reference tokens must fit the context length")
    if any(token < 0 or token >= 248320 for token in args.token_ids):
        parser.error("Reference token is outside the Qwen3.5 vocabulary")
    if args.prompt is not None and (
        args.generate_tokens < 1 or args.generate_tokens > args.context_length
    ):
        parser.error("Generation length must be positive and fit the context")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "model": "Qwen3.5-0.8B-text",
        "checkpoint": str(args.checkpoint_dir.resolve()),
        "dtype": args.dtype,
        "context_length": args.context_length,
        "token_ids": args.token_ids,
        "reference_tokens": [],
        "mtp": False,
        "runtime_validated": False,
        "torch_version": str(torch.__version__),
        "musa_launch_blocking": os.environ.get("MUSA_LAUNCH_BLOCKING", "0"),
    }
    try:
        torch.set_num_threads(4)
        with torch.backends.mudnn.flags(allow_tf32=False):
            export_model(args, report)
        report["status"] = "exported"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (args.output_dir / "result.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
