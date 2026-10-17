"""Run the upstream Llama demo checkpoint with a persistent MUSA KV cache."""

from __future__ import annotations

import argparse
import json
import operator
import subprocess
from pathlib import Path

import torch
from torch.export import export

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


def run_acceptance(output_dir: Path, runner: Path, dtype: torch.dtype) -> dict:
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
    token_ids = torch.tensor([[11]], device="musa", dtype=torch.int64)
    input_pos = torch.tensor([0], device="musa", dtype=torch.int64)
    exported = export(model, (token_ids, {"input_pos": input_pos}))
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
    steps = []
    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_runner_environment(),
    ) as process:
        for position, token in enumerate((11, 7, 22, 5, 17)):
            token_ids = torch.tensor([[token]], device="musa", dtype=torch.int64)
            input_pos = torch.tensor([position], device="musa", dtype=torch.int64)
            _write_tensor(token_ids, token_path)
            _write_tensor(input_pos, position_path)
            with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
                expected = model(token_ids, {"input_pos": input_pos})
            process.stdin.write("\n")
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
            actual = torch.frombuffer(bytearray(output_path.read_bytes()), dtype=dtype)
            reference = expected.detach().cpu().flatten()
            torch.testing.assert_close(
                actual, reference, atol=tolerance, rtol=tolerance
            )
            predicted = actual.argmax().item()
            if predicted != reference.argmax().item():
                raise AssertionError(
                    f"Llama demo predicted a different token at {position}"
                )
            steps.append(
                {
                    "position": position,
                    "input_token": token,
                    "predicted_token": predicted,
                    "max_error": float(
                        (actual.float() - reference.float()).abs().max()
                    ),
                }
            )
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("executor_runner failed after the final decode step")
    return {
        "checkpoint": str(checkpoint_path),
        "dtype": dtype_name,
        "pte": str(pte_path),
        "ptd": str(ptd_path),
        "delegates": len(delegates),
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    args = parser.parse_args()
    result = run_acceptance(args.output_dir, args.runner, DTYPES[args.dtype])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
