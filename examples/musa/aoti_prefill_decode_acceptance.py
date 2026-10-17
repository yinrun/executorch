"""Export a causal prompt prefill and decode twice using its generated KV cache."""

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
from examples.musa.aoti_kv_cache_acceptance import GrowingCacheDecoder
from examples.musa.aoti_model_acceptance import (
    DTYPES,
    TOLERANCES,
    _runner_environment,
    _serialization_schema,
    _write_tensor,
)


class PromptPrefill(nn.Module):
    def __init__(self, decoder: GrowingCacheDecoder) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.decoder.embedding(token_ids)
        sequence = token_ids.shape[1]
        query = self.decoder.query(features).view(1, sequence, 2, 8).transpose(1, 2)
        key = self.decoder.key(features).view(1, sequence, 2, 8).transpose(1, 2)
        value = self.decoder.value(features).view(1, sequence, 2, 8).transpose(1, 2)
        causal_mask = torch.ones(
            sequence, sequence, dtype=torch.bool, device=token_ids.device
        ).tril()
        scores = torch.matmul(query, key.transpose(-2, -1)) * (8**-0.5)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        context = torch.matmul(scores.softmax(dim=-1), value)
        context = context.transpose(1, 2).reshape(1, sequence, 16)
        logits = self.decoder.head(
            self.decoder.norm(features + self.decoder.projection(context))
        )[:, -1:, :]
        return logits, key, value


def _export_model(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output_dir: Path,
    name: str,
    dynamic_shapes: tuple[dict, ...] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = export(model, inputs, dynamic_shapes=dynamic_shapes)
    compile_spec = [MusaBackend.generate_method_name_compile_spec("forward")]
    with _serialization_schema():
        manager = to_edge_transform_and_lower(
            exported, partitioner=[MusaPartitioner(compile_spec)]
        ).to_executorch()
        pte_path = Path(save_pte_program(manager, name, str(output_dir)))
    ptd_path = output_dir / "aoti_musa_blob.ptd"
    if not ptd_path.is_file():
        raise RuntimeError(f"AOTI external data was not created: {ptd_path}")
    return pte_path, ptd_path


def _run(
    runner: Path,
    pte_path: Path,
    ptd_path: Path,
    input_paths: list[Path],
    output_prefix: Path,
    input_shapes: str | None = None,
) -> list[Path]:
    command = [
        str(runner),
        f"--model_path={pte_path}",
        f"--data_path={ptd_path}",
        f"--inputs={','.join(map(str, input_paths))}",
        f"--output_file={output_prefix}",
        "--print_output=summary",
    ]
    if input_shapes is not None:
        command.append(f"--input_shapes={input_shapes}")
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
            f"executor_runner failed\nstdout:\n{error.stdout}\n"
            f"stderr:\n{error.stderr}"
        ) from error
    return [Path(f"{output_prefix}-{index}.bin") for index in range(3)]


def _check_outputs(
    paths: list[Path], expected: tuple[torch.Tensor, ...], tolerance: float
) -> dict:
    actual = [
        torch.frombuffer(bytearray(path.read_bytes()), dtype=reference.dtype).reshape(
            reference.shape
        )
        for path, reference in zip(paths, expected)
    ]
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(
            result, reference.cpu(), atol=tolerance, rtol=tolerance
        )
    token = actual[0].argmax(dim=-1)
    if not torch.equal(token, expected[0].argmax(dim=-1).cpu()):
        raise AssertionError("executor_runner predicted a different token")
    return {
        "predicted_token": token.tolist(),
        "max_errors": [
            float((result.float() - reference.cpu().float()).abs().max())
            for result, reference in zip(actual, expected)
        ],
        "outputs": [str(path) for path in paths],
    }


def run_acceptance(
    output_dir: Path, runner: Path, dtype: torch.dtype, dynamic_prefill: bool = False
) -> dict:
    if not torch.musa.is_available():
        raise RuntimeError("MUSA is not available")
    torch.manual_seed(0)
    decoder = GrowingCacheDecoder().eval().to(device="musa", dtype=dtype)
    prefill = PromptPrefill(decoder).eval()
    prompt = torch.tensor([[11, 7, 22]], device="musa", dtype=torch.int64)
    with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
        expected_prefill = prefill(prompt)
        last_token = decoder(
            prompt[:, -1:],
            expected_prefill[1][:, :, :-1, :],
            expected_prefill[2][:, :, :-1, :],
        )[0]
    dtype_name = next(name for name, candidate in DTYPES.items() if candidate == dtype)
    tolerance = TOLERANCES[dtype_name]
    torch.testing.assert_close(
        expected_prefill[0], last_token, atol=tolerance, rtol=tolerance
    )
    prefill_pte, prefill_ptd = _export_model(
        prefill,
        (prompt,),
        output_dir / "prefill",
        "prompt_prefill",
        ({1: Dim("prompt", min=2, max=4)},) if dynamic_prefill else None,
    )
    initial_key, initial_value = expected_prefill[1:]
    dynamic_shapes = (
        {}, {2: Dim("past", min=2, max=4)}, {2: Dim("past", min=2, max=4)}
    )
    decode_pte, decode_ptd = _export_model(
        decoder,
        (
            torch.tensor([[5]], device="musa"),
            initial_key.contiguous(),
            initial_value.contiguous(),
        ),
        output_dir / "decode",
        "growing_cache_decoder",
        dynamic_shapes,
    )

    prompt_path = output_dir / "prompt.bin"
    _write_tensor(prompt, prompt_path)
    if dynamic_prefill:
        with prompt_path.open("ab") as prompt_file:
            prompt_file.write(bytes((4 - prompt.shape[1]) * prompt.element_size()))
    prefill_paths = _run(
        runner,
        prefill_pte,
        prefill_ptd,
        [prompt_path],
        output_dir / "prefill-output",
        "1x3" if dynamic_prefill else None,
    )
    prefill_result = _check_outputs(prefill_paths, expected_prefill, tolerance)
    dynamic_results = []
    if dynamic_prefill:
        for current_prompt in (prompt[:, :2], torch.cat((prompt, prompt[:, :1]), 1)):
            length = current_prompt.shape[1]
            with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
                expected = prefill(current_prompt)
            path = output_dir / f"prompt-{length}.bin"
            _write_tensor(current_prompt, path)
            if length < 4:
                with path.open("ab") as prompt_file:
                    prompt_file.write(bytes((4 - length) * prompt.element_size()))
            paths = _run(
                runner,
                prefill_pte,
                prefill_ptd,
                [path],
                output_dir / f"prefill-{length}-output",
                f"1x{length}",
            )
            result = _check_outputs(paths, expected, tolerance)
            result["prompt_length"] = length
            dynamic_results.append(result)

    steps = []
    past_key, past_value = initial_key, initial_value
    previous_paths = prefill_paths[1:]
    for step, token in enumerate((5, 17)):
        token_ids = torch.tensor([[token]], device="musa", dtype=torch.int64)
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            expected = decoder(token_ids, past_key, past_value)
        token_path = output_dir / f"step-{step}-token.bin"
        _write_tensor(token_ids, token_path)
        cache_paths = previous_paths
        if past_key.shape[2] < 4:
            cache_paths = []
            for name, path in zip(("key", "value"), previous_paths):
                padded_path = output_dir / f"step-{step}-{name}-input.bin"
                padding = (4 - past_key.shape[2]) * 2 * 8 * past_key.element_size()
                padded_path.write_bytes(path.read_bytes() + bytes(padding))
                cache_paths.append(padded_path)
        length = past_key.shape[2]
        output_paths = _run(
            runner,
            decode_pte,
            decode_ptd,
            [token_path, *cache_paths],
            output_dir / f"step-{step}-output",
            f"1x1,1x2x{length}x8,1x2x{length}x8",
        )
        result = _check_outputs(output_paths, expected, tolerance)
        result.update({"token": token, "past_length": length})
        steps.append(result)
        past_key, past_value = expected[1:]
        previous_paths = output_paths[1:]
    return {
        "dtype": dtype_name,
        "prompt": prompt.tolist(),
        "prefill": prefill_result,
        "dynamic_prefill": dynamic_results,
        "decode_steps": steps,
        "prefill_pte": str(prefill_pte),
        "prefill_ptd": str(prefill_ptd),
        "decode_pte": str(decode_pte),
        "decode_ptd": str(decode_ptd),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--dynamic-prefill", action="store_true")
    args = parser.parse_args()
    result = run_acceptance(
        args.output_dir, args.runner, DTYPES[args.dtype], args.dynamic_prefill
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
