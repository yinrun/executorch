"""Validate exported Qwen3.5 logits through a persistent native MUSA runner."""

import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time


TOLERANCES = {"fp32": (0.001, 0.001), "bf16": (0.04, 0.04)}
VOCAB_SIZE = 248320


def tensor_values(payload, dtype):
    width = 4 if dtype == "fp32" else 2
    if len(payload) != VOCAB_SIZE * width:
        raise ValueError("Expected one full-vocabulary Qwen3.5 logit vector")
    if dtype == "bf16":
        words = array.array("H", payload)
        if sys.byteorder != "little":
            words.byteswap()
        payload = b"".join(struct.pack("<I", value << 16) for value in words)
    values = array.array("f", payload)
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite Qwen3.5 logits")
    return values


def compare_logits(actual_payload, reference_payload, dtype):
    actual = tensor_values(actual_payload, dtype)
    reference = tensor_values(reference_payload, dtype)
    atol, rtol = TOLERANCES[dtype]
    differences = [abs(value - expected) for value, expected in zip(actual, reference)]
    return {
        "predicted_token": max(range(VOCAB_SIZE), key=actual.__getitem__),
        "reference_token": max(range(VOCAB_SIZE), key=reference.__getitem__),
        "max_error": max(differences),
        "mean_error": math.fsum(differences) / VOCAB_SIZE,
        "outside_tolerance": sum(
            error > atol + rtol * abs(expected)
            for error, expected in zip(differences, reference)
        ),
    }


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_libraries(log):
    loaded = [
        value.strip() for value in re.findall(r"calling init:[ \t]*([^\r\n]+)", log)
    ]
    if not loaded or any(
        name in Path(library).name
        for library in loaded
        for name in ("libtorch", "libc10", "libpython", "libmusa_python")
    ):
        raise AssertionError("Native runtime library isolation was not established")
    return loaded


def validate(args, report):
    metadata_path = args.artifacts / "result.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["status"] != "exported" or metadata["model"] != "Qwen3.5-0.8B-text":
        raise ValueError("Expected a successful Qwen3.5 text export")
    dtype = metadata["dtype"]
    token_ids = metadata["token_ids"]
    prompt_length = metadata.get("prompt_token_count", len(token_ids))
    if not 1 <= prompt_length <= len(token_ids):
        raise ValueError("Invalid prompt length")
    if not 2 <= len(token_ids) <= metadata["context_length"]:
        raise ValueError("Cache validation requires at least two reference tokens")
    if any(not 0 <= token < VOCAB_SIZE for token in token_ids):
        raise ValueError("Reference token is outside the vocabulary")
    report.update(dtype=dtype, atol=TOLERANCES[dtype][0], rtol=TOLERANCES[dtype][1])
    pte = args.pte.resolve() if args.pte else args.artifacts / "qwen35.pte"
    ptd = args.ptd.resolve() if args.ptd else args.artifacts / "aoti_musa_blob.ptd"
    references = [
        args.artifacts / f"eager-{position}.bin" for position in range(len(token_ids))
    ]
    tracked = [metadata_path, pte, ptd, args.runner, *references]
    before = {str(path): file_hash(path) for path in tracked}
    report["input_hashes"] = before
    artifact_names = [(pte, "qwen35.pte"), (ptd, "aoti_musa_blob.ptd")]
    artifact_names.extend((path, path.name) for path in references)
    for path, name in artifact_names:
        if name not in metadata["artifacts"] and path in references:
            continue
        if before[str(path)] != metadata["artifacts"][name]:
            raise ValueError(f"Export artifact hash changed: {path}")
    token_path = args.output_dir / "token.bin"
    position_path = args.output_dir / "position.bin"
    token_path.write_bytes(struct.pack("<q", token_ids[0]))
    position_path.write_bytes(struct.pack("<q", 0))
    log_path = args.output_dir / "runner.log"
    command = [
        str(args.runner),
        f"--model_path={pte}",
        f"--data_path={ptd}",
        f"--inputs={token_path},{position_path}",
        f"--output_file={args.output_dir / 'output'}",
        "--print_output=none",
        "--server_mode",
    ]
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(args.output_dir),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "LD_BIND_NOW": "1",
        "LD_DEBUG": "libs",
    }
    report.update(command=command, environment=environment)
    with log_path.open("w") as log, subprocess.Popen(
        command,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=log,
        text=True,
    ) as process:
        try:
            for session in range(2):
                previous_prediction = None
                for position, token in enumerate(token_ids):
                    if position >= prompt_length:
                        if previous_prediction != token:
                            raise AssertionError(
                                "Native greedy trajectory differs from the reference"
                            )
                        token = previous_prediction
                    token_path.write_bytes(struct.pack("<q", token))
                    position_path.write_bytes(struct.pack("<q", position))
                    process.stdin.write("\n")
                    process.stdin.flush()
                    deadline = time.monotonic() + args.timeout
                    while log_path.read_text().count("SERVER MODE DONE") <= len(
                        report["steps"]
                    ):
                        if process.poll() is not None:
                            raise RuntimeError(
                                f"Runner exited before completion: {process.returncode}"
                            )
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Runner response timeout")
                        time.sleep(0.01)
                    payload = (args.output_dir / "output-0.bin").read_bytes()
                    output_path = (
                        args.output_dir / f"session-{session}-step-{position}.bin"
                    )
                    output_path.write_bytes(payload)
                    row = compare_logits(
                        payload, references[position].read_bytes(), dtype
                    )
                    row.update(
                        session=session,
                        position=position,
                        token=token,
                        sha256=file_hash(output_path),
                    )
                    report["steps"].append(row)
                    previous_prediction = row["predicted_token"]
                    print(json.dumps(row), flush=True)
                    if (
                        row["predicted_token"] != row["reference_token"]
                        or row["outside_tolerance"]
                    ):
                        raise AssertionError(
                            "Native logits failed numerical or token-ID validation"
                        )
                    if (
                        session == 1
                        and payload
                        != (
                            args.output_dir / f"session-0-step-{position}.bin"
                        ).read_bytes()
                    ):
                        raise AssertionError("Reset sequence is not byte-identical")
            process.stdin.close()
            if process.wait(timeout=30) != 0:
                raise RuntimeError(f"Runner failed at shutdown: {process.returncode}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    report["loaded_libraries"] = runtime_libraries(log_path.read_text())
    if before != {str(path): file_hash(path) for path in tracked}:
        raise AssertionError("Input artifacts changed during validation")
    if "generated_token_ids" in metadata:
        generated = [
            row["predicted_token"]
            for row in report["steps"]
            if row["session"] == 0 and row["position"] >= prompt_length - 1
        ]
        if generated != metadata["generated_token_ids"]:
            raise AssertionError("Generated token IDs differ from the reference")
        report.update(
            generated_token_ids=generated, generated_text=metadata["generated_text"]
        )
    report.update(status="passed", runtime_validated=True, reset_byte_equal=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--pte", type=Path)
    parser.add_argument("--ptd", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Timeout must be finite and positive")
    args.artifacts = args.artifacts.resolve()
    args.runner = args.runner.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "runtime_validated": False,
        "performance_claim": False,
        "steps": [],
    }
    try:
        validate(args, report)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (args.output_dir / "result.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
