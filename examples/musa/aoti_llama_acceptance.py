"""Export ExecuTorch's five-layer Llama demo checkpoint and check MUSA output."""

from __future__ import annotations

import argparse
import json
import operator
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
    _run_runner,
    _serialization_schema,
    _write_tensor,
)


def run_acceptance(output_dir: Path, runner: Path, dtype: torch.dtype) -> dict:
    if not torch.musa.is_available():
        raise RuntimeError("MUSA is not available")
    params_dir = Path(__file__).parents[1] / "models/llama/params"
    config = json.loads((params_dir / "demo_config.json").read_text())
    args = ModelArgs(**config, max_seq_len=128, max_context_len=128)
    model = construct_transformer(args)
    checkpoint_path = params_dir / "demo_rand_params.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    legacy_masks = {f"layers.{index}.attention.mask" for index in range(args.n_layers)}
    if missing or set(unexpected) != legacy_masks:
        raise RuntimeError(f"Demo checkpoint mismatch: {missing=}, {unexpected=}")
    model = model.eval().to(device="musa", dtype=dtype)
    token_ids = torch.tensor([[11, 7, 22]], dtype=torch.int64, device="musa")
    with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
        expected = model(token_ids)
    exported = export(model, (token_ids,))
    compile_spec = [MusaBackend.generate_method_name_compile_spec("forward")]
    with _serialization_schema():
        manager = to_edge_transform_and_lower(
            exported, partitioner=[MusaPartitioner(compile_spec)]
        )
        delegate_calls = [
            node
            for node in manager.exported_program().graph.nodes
            if node.op == "call_function"
            and "executorch_call_delegate" in str(node.target)
        ]
        if len(delegate_calls) != 1:
            raise RuntimeError(
                f"Expected one MUSA delegate for Llama demo, got {len(delegate_calls)}"
            )
        host_ops = [
            str(node.target)
            for node in manager.exported_program().graph.nodes
            if node.op == "call_function"
            and node not in delegate_calls
            and node.target is not operator.getitem
        ]
        if host_ops:
            raise RuntimeError(f"Llama demo has non-delegated operators: {host_ops}")
        manager = manager.to_executorch()
        output_dir.mkdir(parents=True, exist_ok=True)
        pte_path = Path(save_pte_program(manager, "llama_demo", str(output_dir)))
    ptd_path = output_dir / "aoti_musa_blob.ptd"
    if not ptd_path.is_file():
        raise RuntimeError(f"AOTI external data was not created: {ptd_path}")
    input_path = output_dir / "prompt.bin"
    _write_tensor(token_ids, input_path)
    output_prefix = output_dir / "runner-output"
    _run_runner(runner, pte_path, ptd_path, input_path, output_prefix)
    output_path = Path(f"{output_prefix}-0.bin")
    actual = torch.frombuffer(bytearray(output_path.read_bytes()), dtype=dtype)
    reference = expected.detach().cpu().flatten()
    dtype_name = next(name for name, candidate in DTYPES.items() if candidate == dtype)
    tolerance = TOLERANCES[dtype_name]
    torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
    predicted = actual.argmax().item()
    if predicted != reference.argmax().item():
        raise AssertionError("Llama demo predicted a different token")
    return {
        "checkpoint": str(checkpoint_path),
        "config": str(params_dir / "demo_config.json"),
        "dtype": dtype_name,
        "pte": str(pte_path),
        "ptd": str(ptd_path),
        "output": str(output_path),
        "predicted_token": predicted,
        "delegates": len(delegate_calls),
        "max_error": float((actual.float() - reference.float()).abs().max()),
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
