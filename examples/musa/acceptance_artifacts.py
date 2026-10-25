"""Verify recorded MUSA acceptance tensors without importing PyTorch or using a GPU."""

import argparse
import hashlib
import json
import math
from pathlib import Path


ITEM_SIZES = {"fp32": 4, "fp16": 2, "bf16": 2, "int64": 8}


def record_artifact(path: Path, payload: bytes | None = None) -> dict:
    stored = path.read_bytes()
    if payload is not None and stored != payload:
        raise ValueError(f"Artifact differs from tensor snapshot: {path}")
    payload = stored
    return {
        "file": path.name,
        "nbytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def verify_artifacts(root: Path, report: dict) -> None:
    if report.get("artifact_version") != 1 or not report.get("artifact_cases"):
        raise ValueError("Missing or unsupported tensor artifact manifest")
    root = root.resolve()
    seen = set()
    repeats = report["runner_repeats"] if report["runner"] is not None else 0
    if type(repeats) is not int or repeats < 0 or (report["runner"] and repeats == 0):
        raise ValueError("Invalid runner repeat count")

    def verify(record, shape, dtype, padded=False):
        filename = record["file"]
        path = root / filename
        if (
            Path(filename).name != filename
            or path.resolve().parent != root
            or filename in seen
        ):
            raise ValueError(f"Invalid or duplicate artifact path: {filename}")
        seen.add(filename)
        if any(type(size) is not int or size < 0 for size in shape):
            raise ValueError(f"Invalid tensor shape: {shape}")
        logical_size = math.prod(shape) * ITEM_SIZES[dtype]
        payload = path.read_bytes()
        if len(payload) != record["nbytes"]:
            raise ValueError(f"Artifact size mismatch: {filename}")
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {filename}")
        if padded:
            if (
                len(payload) < logical_size
                or len(payload) % ITEM_SIZES[dtype]
                or any(payload[logical_size:])
            ):
                raise ValueError(f"Invalid input padding: {filename}")
        elif len(payload) != logical_size:
            raise ValueError(f"Artifact tensor shape/size mismatch: {filename}")

    shapes = set()
    for case in report["artifact_cases"]:
        shape = tuple(case["input_shape"])
        if shape in shapes:
            raise ValueError(f"Duplicate input shape: {shape}")
        shapes.add(shape)
        verify(case["input"], shape, report["input_dtype"], padded=True)
        verify(case["expected"], case["expected_shape"], report["dtype"])
        if len(case["outputs"]) != repeats:
            raise ValueError(f"Incomplete runner artifacts for shape {shape}")
        for output in case["outputs"]:
            verify(output, case["expected_shape"], report["dtype"])
    shape_results = report.get("dynamic_shape_results") or report.get(
        "dynamic_spatial_results"
    )
    if shape_results:
        expected_shapes = {tuple(result["shape"]) for result in shape_results}
        if shapes != expected_shapes:
            raise ValueError("Artifact cases do not match executed dynamic shapes")
    elif report.get("dynamic_batch_results"):
        expected_batches = {
            result["batch_size"] for result in report["dynamic_batch_results"]
        }
        if (
            len(shapes) != len(expected_batches)
            or {shape[0] for shape in shapes} != expected_batches
        ):
            raise ValueError("Artifact cases do not match executed dynamic batches")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    report = json.loads(args.result.read_text())
    verify_artifacts(args.result.parent, report)
    print(f"Verified tensor artifacts: {args.result}")


if __name__ == "__main__":
    main()
