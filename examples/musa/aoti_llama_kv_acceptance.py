"""Run the upstream Llama demo checkpoint with a persistent MUSA KV cache."""

from __future__ import annotations

import argparse
import json
import operator
import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.export import Dim, export

from executorch.backends.musa import MusaBackend, MusaPartitioner
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program
from examples.musa.aoti_model_acceptance import (
    DTYPES,
    TOLERANCES,
    _runner_environment,
    _serialization_schema,
    _write_tensor,
)


class ExportableMathSDPA(nn.Module):
    def __init__(self, sdpa: nn.Module) -> None:
        super().__init__()
        self.n_rep = sdpa.n_rep
        self.dim = sdpa.dim

    def forward(
        self,
        input_pos: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_size: int,
        sequence: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        key = key.repeat_interleave(self.n_rep, dim=1)
        value = value.repeat_interleave(self.n_rep, dim=1)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (
            query.size(-1) ** -0.5
        )
        context = torch.matmul(
            scores.masked_fill(~mask, float("-inf")).softmax(-1), value
        )
        return context.transpose(1, 2).reshape(batch_size, sequence, self.dim)


def run_acceptance(
    output_dir: Path, runner: Path, dtype: torch.dtype, dynamic_prefill: bool = False
) -> dict:
    if not torch.musa.is_available():
        raise RuntimeError("MUSA is not available")
    params_dir = Path(__file__).parents[1] / "models/llama/params"
    config = json.loads((params_dir / "demo_config.json").read_text())
    args = ModelArgs(
        **config, max_seq_len=8, max_context_len=8, use_kv_cache=True
    )
    model = construct_transformer(args)
    checkpoint_path = params_dir / "demo_rand_params.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    expected_masks = {
        f"layers.{index}.attention.mask" for index in range(args.n_layers)
    }
    expected_caches = {
        f"layers.{index}.attention.kv_cache.{name}_cache"
        for index in range(args.n_layers)
        for name in ("k", "v")
    }
    if set(missing) != expected_caches or set(unexpected) != expected_masks:
        raise RuntimeError(f"Demo checkpoint mismatch: {missing=}, {unexpected=}")
    model = model.eval().to(device="musa", dtype=dtype)
    original_model = None
    if dynamic_prefill:
        original_model = construct_transformer(args).eval()
        original_model.load_state_dict(checkpoint, strict=False)
        original_model = original_model.to(device="musa", dtype=dtype)
        for layer in model.layers:
            layer.attention.SDPA = ExportableMathSDPA(layer.attention.SDPA)
    first_tokens = (11, 7) if dynamic_prefill else (11,)
    token_ids = torch.tensor([first_tokens], device="musa", dtype=torch.int64)
    input_pos = torch.arange(len(first_tokens), device="musa", dtype=torch.int64)
    max_prefill_length = args.max_context_len - 1
    dynamic_shapes = (
        (
            {1: Dim("sequence", min=1, max=max_prefill_length)},
            {"input_pos": {0: Dim("sequence", min=1, max=max_prefill_length)}},
        )
        if dynamic_prefill
        else None
    )
    exported = export(
        model, (token_ids, {"input_pos": input_pos}), dynamic_shapes=dynamic_shapes
    )
    compile_spec = [MusaBackend.generate_method_name_compile_spec("forward")]
    with _serialization_schema():
        manager = to_edge_transform_and_lower(
            exported, partitioner=[MusaPartitioner(compile_spec)]
        )
        delegates = [
            node
            for node in manager.exported_program().graph.nodes
            if node.op == "call_function"
            and "executorch_call_delegate" in str(node.target)
        ]
        host_ops = [
            str(node.target)
            for node in manager.exported_program().graph.nodes
            if node.op == "call_function"
            and node not in delegates
            and node.target is not operator.getitem
        ]
        if len(delegates) != 1 or host_ops:
            raise RuntimeError(
                f"Expected one MUSA delegate and no host ops, got "
                f"{len(delegates)} delegates and {host_ops}"
            )
        manager = manager.to_executorch()
        output_dir.mkdir(parents=True, exist_ok=True)
        pte_path = Path(save_pte_program(manager, "llama_kv", str(output_dir)))
    ptd_path = output_dir / "aoti_musa_blob.ptd"
    if not ptd_path.is_file():
        raise RuntimeError(f"AOTI external data was not created: {ptd_path}")

    token_path = output_dir / "token.bin"
    position_path = output_dir / "position.bin"
    output_prefix = output_dir / "runner-output"
    _write_tensor(token_ids, token_path)
    _write_tensor(input_pos, position_path)
    command = [
        str(runner),
        f"--model_path={pte_path}",
        f"--data_path={ptd_path}",
        f"--inputs={token_path},{position_path}",
        f"--output_file={output_prefix}",
        "--print_output=summary",
        "--server_mode",
    ]
    dtype_name = next(name for name, candidate in DTYPES.items() if candidate == dtype)
    tolerance = TOLERANCES[dtype_name]
    sessions = (
        (
            ((11, 7, 22), (5,), (17,)),
            ((11, 7, 22, 5, 17, 13, 19), (3,)),
        )
        if dynamic_prefill
        else (tuple((token,) for token in (11, 7, 22, 5, 17)),)
    )
    session_steps = []
    for input_steps in sessions:
        for eager_model in (model, original_model):
            if eager_model is not None:
                with torch.no_grad():
                    for layer in eager_model.layers:
                        layer.attention.kv_cache.k_cache.zero_()
                        layer.attention.kv_cache.v_cache.zero_()
        steps = []
        position = 0
        with subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_runner_environment(),
        ) as process:
            for tokens in input_steps:
                token_ids = torch.tensor([tokens], device="musa", dtype=torch.int64)
                input_pos = torch.arange(
                    position, position + len(tokens), device="musa", dtype=torch.int64
                )
                _write_tensor(token_ids, token_path)
                _write_tensor(input_pos, position_path)
                if dynamic_prefill:
                    padding = (
                        max_prefill_length - len(tokens)
                    ) * token_ids.element_size()
                    with token_path.open("ab") as token_file:
                        token_file.write(bytes(padding))
                    with position_path.open("ab") as position_file:
                        position_file.write(bytes(padding))
                with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
                    expected = model(token_ids, {"input_pos": input_pos})
                    if original_model is not None:
                        original = original_model(token_ids, {"input_pos": input_pos})
                request = (
                    f"input_shapes=1x{len(tokens)},{len(tokens)}"
                    if dynamic_prefill
                    else ""
                )
                process.stdin.write(f"{request}\n")
                process.stdin.flush()
                while True:
                    line = process.stderr.readline()
                    if "SERVER MODE DONE" in line:
                        break
                    if not line:
                        raise RuntimeError(
                            f"executor_runner stopped at position {position}: "
                            f"exit code {process.poll()}"
                        )
                output_path = Path(f"{output_prefix}-0.bin")
                actual = torch.frombuffer(
                    bytearray(output_path.read_bytes()), dtype=dtype
                )
                reference = expected.detach().cpu().flatten()
                torch.testing.assert_close(
                    actual, reference, atol=tolerance, rtol=tolerance
                )
                if original_model is not None:
                    torch.testing.assert_close(
                        expected, original, atol=tolerance, rtol=tolerance
                    )
                predicted = actual.argmax().item()
                if predicted != reference.argmax().item():
                    raise AssertionError(
                        f"Llama demo predicted a different token at {position}"
                    )
                step = {
                    "position": position,
                    "input_tokens": list(tokens),
                    "predicted_token": predicted,
                    "max_error": float(
                        (actual.float() - reference.float()).abs().max()
                    ),
                }
                if len(tokens) == 1:
                    step["input_token"] = tokens[0]
                steps.append(step)
                position += len(tokens)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("executor_runner failed after the final decode step")
        session_steps.append(steps)
    result = {
        "checkpoint": str(checkpoint_path),
        "dtype": dtype_name,
        "pte": str(pte_path),
        "ptd": str(ptd_path),
        "delegates": len(delegates),
        "dynamic_prefill": dynamic_prefill,
        "steps": session_steps[0],
    }
    if dynamic_prefill:
        result["long_prefill_steps"] = session_steps[1]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--dynamic-prefill", action="store_true")
    args = parser.parse_args()
    result = run_acceptance(
        args.output_dir, args.runner, DTYPES[args.dtype], args.dynamic_prefill
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
