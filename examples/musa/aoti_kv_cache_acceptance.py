"""Validate two incremental decode steps with rolling or growing KV caches."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.export import Dim, export

from executorch.backends.musa import MusaBackend, MusaPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program
from examples.musa.aoti_model_acceptance import (
    DTYPES,
    TOLERANCES,
    _runner_environment,
    _serialization_schema,
    _write_tensor,
)


class RollingCacheDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, 16)
        self.query = nn.Linear(16, 16, bias=False)
        self.key = nn.Linear(16, 16, bias=False)
        self.value = nn.Linear(16, 16, bias=False)
        self.projection = nn.Linear(16, 16)
        self.norm = nn.LayerNorm(16)
        self.head = nn.Linear(16, 64, bias=False)

    def _update_cache(
        self, past: torch.Tensor, current: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((past[:, :, 1:, :], current), dim=2)

    def forward(
        self, token_ids: torch.Tensor, past_key: torch.Tensor, past_value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.embedding(token_ids)
        query = self.query(features).view(1, 1, 2, 8).transpose(1, 2)
        key = self.key(features).view(1, 1, 2, 8).transpose(1, 2)
        value = self.value(features).view(1, 1, 2, 8).transpose(1, 2)
        next_key = self._update_cache(past_key, key)
        next_value = self._update_cache(past_value, value)
        scores = torch.matmul(query, next_key.transpose(-2, -1)) * (8**-0.5)
        context = torch.matmul(scores.softmax(dim=-1), next_value)
        context = context.transpose(1, 2).reshape(1, 1, 16)
        logits = self.head(self.norm(features + self.projection(context)))
        return logits, next_key, next_value


class GrowingCacheDecoder(RollingCacheDecoder):
    def _update_cache(
        self, past: torch.Tensor, current: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((past, current), dim=2)


def run_acceptance(
    output_dir: Path,
    runner: Path,
    dtype: torch.dtype,
    cache_mode: str = "rolling",
    initial_cache_length: int | None = None,
) -> dict:
    if not torch.musa.is_available():
        raise RuntimeError("MUSA is not available")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    if cache_mode not in ("rolling", "growing"):
        raise ValueError(f"Unknown cache mode: {cache_mode}")
    if cache_mode == "growing":
        initial_length = 3 if initial_cache_length is None else initial_cache_length
        if initial_length not in (2, 3):
            raise ValueError("Growing cache requires initial length 2 or 3")
    else:
        if initial_cache_length not in (None, 4):
            raise ValueError("Rolling cache requires initial length 4")
        initial_length = 4
    model_type = GrowingCacheDecoder if cache_mode == "growing" else RollingCacheDecoder
    model = model_type().eval().to(device="musa", dtype=dtype)
    token_ids = torch.tensor([[5]], device="musa", dtype=torch.int64)
    past_key = torch.randn((1, 2, initial_length, 8), device="musa", dtype=dtype)
    past_value = torch.randn((1, 2, initial_length, 8), device="musa", dtype=dtype)
    example = (token_ids, past_key, past_value)
    dynamic_shapes = (
        ({}, {2: Dim("past", min=2, max=4)}, {2: Dim("past", min=2, max=4)})
        if cache_mode == "growing"
        else None
    )
    exported = export(model, example, dynamic_shapes=dynamic_shapes)
    compile_spec = [MusaBackend.generate_method_name_compile_spec("forward")]
    with _serialization_schema():
        manager = to_edge_transform_and_lower(
            exported, partitioner=[MusaPartitioner(compile_spec)]
        ).to_executorch()
        pte_path = Path(
            save_pte_program(manager, f"{cache_mode}_cache_decoder", str(output_dir))
        )
    ptd_path = output_dir / "aoti_musa_blob.ptd"
    if not ptd_path.exists():
        raise RuntimeError(f"AOTI external data was not created: {ptd_path}")

    dtype_name = next(name for name, candidate in DTYPES.items() if candidate == dtype)
    tolerance = TOLERANCES[dtype_name]
    steps = []
    for step, token in enumerate((5, 17)):
        token_ids = torch.tensor([[token]], device="musa", dtype=torch.int64)
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            expected = model(token_ids, past_key, past_value)
        input_paths = [output_dir / f"step-{step}-token.bin"]
        _write_tensor(token_ids, input_paths[0])
        if step == 0:
            for name, cache in (("key", past_key), ("value", past_value)):
                path = output_dir / f"step-0-{name}.bin"
                _write_tensor(cache, path)
                if cache_mode == "growing":
                    with path.open("ab") as cache_file:
                        padding = (4 - initial_length) * 2 * 8 * cache.element_size()
                        cache_file.write(bytes(padding))
                input_paths.append(path)
        else:
            if cache_mode == "growing" and past_key.shape[2] < 4:
                for name, path in zip(("key", "value"), previous_cache_paths):
                    padded_path = output_dir / f"step-{step}-{name}-input.bin"
                    padding = (4 - past_key.shape[2]) * 2 * 8 * past_key.element_size()
                    padded_path.write_bytes(path.read_bytes() + bytes(padding))
                    input_paths.append(padded_path)
            else:
                input_paths.extend(previous_cache_paths)
        output_prefix = output_dir / f"step-{step}-output"
        command = [
            str(runner),
            f"--model_path={pte_path}",
            f"--data_path={ptd_path}",
            f"--inputs={','.join(map(str, input_paths))}",
            f"--output_file={output_prefix}",
            "--print_output=summary",
        ]
        if cache_mode == "growing":
            length = past_key.shape[2]
            command.append(
                f"--input_shapes=1x1,1x2x{length}x8,1x2x{length}x8"
            )
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=_runner_environment(),
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"executor_runner step {step} failed\n"
                f"stdout:\n{error.stdout}\nstderr:\n{error.stderr}"
            ) from error
        output_paths = [Path(f"{output_prefix}-{index}.bin") for index in range(3)]
        actual = [
            torch.frombuffer(bytearray(path.read_bytes()), dtype=dtype).reshape(
                reference.shape
            )
            for path, reference in zip(output_paths, expected)
        ]
        for reference, result in zip(expected, actual):
            torch.testing.assert_close(
                result, reference.cpu(), atol=tolerance, rtol=tolerance
            )
        predicted = actual[0].argmax(dim=-1)
        expected_token = expected[0].argmax(dim=-1).cpu()
        if not torch.equal(predicted, expected_token):
            raise AssertionError(
                f"predicted token differs at step {step}: "
                f"expected {expected_token.tolist()}, got {predicted.tolist()}"
            )
        steps.append(
            {
                "token": token,
                "cache_length": past_key.shape[2],
                "predicted_token": predicted.tolist(),
                "max_errors": [
                    float((result.float() - reference.cpu().float()).abs().max())
                    for reference, result in zip(expected, actual)
                ],
                "outputs": [str(path) for path in output_paths],
            }
        )
        past_key, past_value = expected[1:]
        previous_cache_paths = output_paths[1:]
    return {
        "pte": str(pte_path),
        "ptd": str(ptd_path),
        "dtype": dtype_name,
        "cache_mode": cache_mode,
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument(
        "--cache-mode", choices=("rolling", "growing"), default="rolling"
    )
    parser.add_argument("--initial-cache-length", type=int)
    args = parser.parse_args()
    result = run_acceptance(
        args.output_dir,
        args.runner,
        DTYPES[args.dtype],
        args.cache_mode,
        args.initial_cache_length,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
