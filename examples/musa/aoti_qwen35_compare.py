"""Diagnose teacher-forced Qwen3.5 logits without relaxing acceptance gates."""

import argparse
import heapq
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time

from examples.musa.aoti_qwen35_validate import (
    file_hash,
    runtime_libraries,
    tensor_values,
    TOLERANCES,
)


def compare_distributions(actual, reference, top_k=10):
    if (
        not actual
        or len(actual) != len(reference)
        or not isinstance(top_k, int)
        or top_k < 1
    ):
        raise ValueError(
            "Expected equally sized nonempty logit vectors and positive top_k"
        )
    if not all(
        math.isfinite(value) for values in (actual, reference) for value in values
    ):
        raise ValueError("Logits must be finite")
    count = len(actual)
    top_k = min(top_k, count)
    actual_top = heapq.nlargest(top_k, range(count), key=actual.__getitem__)
    reference_top = heapq.nlargest(top_k, range(count), key=reference.__getitem__)
    actual_peak = actual[actual_top[0]]
    reference_peak = reference[reference_top[0]]
    actual_exp = [math.exp(value - actual_peak) for value in actual]
    reference_exp = [math.exp(value - reference_peak) for value in reference]
    actual_total = math.fsum(actual_exp)
    reference_total = math.fsum(reference_exp)
    actual_probabilities = [value / actual_total for value in actual_exp]
    reference_probabilities = [value / reference_total for value in reference_exp]
    log_normalizer_delta = (
        actual_peak
        - reference_peak
        + math.log(actual_total)
        - math.log(reference_total)
    )
    errors = [abs(value - expected) for value, expected in zip(actual, reference)]
    ordered_errors = sorted(errors)
    return {
        "elements": count,
        "different": sum(error != 0 for error in errors),
        "max_error": max(errors),
        "mean_error": math.fsum(errors) / count,
        "rms_error": math.sqrt(math.fsum(error * error for error in errors) / count),
        "p99_error": ordered_errors[int(0.99 * (count - 1))],
        "predicted_token": actual_top[0],
        "reference_token": reference_top[0],
        "top_k": top_k,
        "top_k_overlap": len(set(actual_top) & set(reference_top)),
        "actual_top_ids": actual_top,
        "reference_top_ids": reference_top,
        "reference_top_probability": reference_probabilities[reference_top[0]],
        "actual_probability_of_reference_top": actual_probabilities[reference_top[0]],
        "total_variation": 0.5
        * math.fsum(
            abs(value - expected)
            for value, expected in zip(actual_probabilities, reference_probabilities)
        ),
        "kl_reference_actual": max(
            0.0,
            math.fsum(
                probability * (expected - value + log_normalizer_delta)
                for probability, expected, value in zip(
                    reference_probabilities, reference, actual
                )
            ),
        ),
    }


def compare_payloads(actual, reference, dtype):
    actual_values = tensor_values(actual, dtype)
    reference_values = tensor_values(reference, dtype)
    row = compare_distributions(actual_values, reference_values)
    atol, rtol = TOLERANCES[dtype]
    row["outside_legacy_tolerance"] = sum(
        abs(value - expected) > atol + rtol * abs(expected)
        for value, expected in zip(actual_values, reference_values)
    )
    row["byte_equal"] = actual == reference
    return row


def summarize_comparison(steps):
    primary = [row for row in steps if row["session"] == 0]
    if not primary:
        raise ValueError("Expected at least one first-session comparison")
    repeated = [row for row in steps if row["session"] > 0]
    fixtures = []
    for name in dict.fromkeys(row["fixture"] for row in primary):
        rows = [row for row in primary if row["fixture"] == name]
        metrics = [row["native_vs_hf"] for row in rows]
        phases = {}
        for phase in ("prompt", "generation"):
            selected = [row for row in rows if row["phase"] == phase]
            phases[phase] = {
                "steps": len(selected),
                "argmax_mismatch_positions": [
                    row["position"]
                    for row in selected
                    if row["native_vs_hf"]["predicted_token"]
                    != row["native_vs_hf"]["reference_token"]
                ],
            }
        fixtures.append(
            {
                "name": name,
                "steps": len(rows),
                "byte_equal_steps": sum(metric["byte_equal"] for metric in metrics),
                "et_hf_byte_equal_steps": sum(
                    row["et_vs_hf"]["byte_equal"] for row in rows
                ),
                "max_error": max(metric["max_error"] for metric in metrics),
                "mean_error": math.fsum(metric["mean_error"] for metric in metrics)
                / len(rows),
                "max_total_variation": max(
                    metric["total_variation"] for metric in metrics
                ),
                "steps_outside_legacy_tolerance": sum(
                    metric["outside_legacy_tolerance"] > 0 for metric in metrics
                ),
                "first_difference_position": next(
                    (
                        row["position"]
                        for row in rows
                        if not row["native_vs_hf"]["byte_equal"]
                    ),
                    None,
                ),
                "phases": phases,
            }
        )
    return {
        "scope": "teacher_forced_fixed_trajectories",
        "native_calls": len(steps),
        "reset_checks": len(repeated),
        "reset_byte_equal": (
            all(row["reset_byte_equal"] for row in repeated) if repeated else None
        ),
        "fixtures": fixtures,
    }


def compare(args, report):
    manifest = json.loads(args.references.read_text())
    if manifest["status"] != "complete" or manifest["dtype"] != "bf16":
        raise ValueError("Expected a complete BF16 reference manifest")
    if not manifest["fixtures"]:
        raise ValueError("Expected at least one reference fixture")
    for fixture in manifest["fixtures"]:
        if not 1 <= len(fixture["token_ids"]) <= 64:
            raise ValueError("Fixture must fit the exported context")
        if not 1 <= fixture["prompt_token_count"] <= len(fixture["token_ids"]):
            raise ValueError("Invalid prompt token count")
        if any(
            type(token) is not int or not 0 <= token < 248320
            for token in fixture["token_ids"]
        ):
            raise ValueError("Invalid token ID")
    tracked = [args.references, args.pte, args.ptd, args.runner]
    before = {str(path): file_hash(path) for path in tracked}
    report.update(input_hashes=before, reference_metadata=manifest)
    token_path = args.output_dir / "token.bin"
    position_path = args.output_dir / "position.bin"
    token_path.write_bytes(struct.pack("<q", 0))
    position_path.write_bytes(struct.pack("<q", 0))
    log_path = args.output_dir / "runner.log"
    command = [
        str(args.runner),
        f"--model_path={args.pte}",
        f"--data_path={args.ptd}",
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
    executions = 0
    with log_path.open("w") as log, subprocess.Popen(
        command,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=log,
        text=True,
    ) as process:
        try:
            for session in range(args.repeats):
                for fixture_index, fixture in enumerate(manifest["fixtures"]):
                    for position, token in enumerate(fixture["token_ids"]):
                        token_path.write_bytes(struct.pack("<q", token))
                        position_path.write_bytes(struct.pack("<q", position))
                        process.stdin.write("\n")
                        process.stdin.flush()
                        deadline = time.monotonic() + args.timeout
                        while (
                            log_path.read_text().count("SERVER MODE DONE") <= executions
                        ):
                            if (
                                process.poll() is not None
                                or time.monotonic() >= deadline
                            ):
                                raise RuntimeError("Native runner exited or timed out")
                            time.sleep(0.01)
                        executions += 1
                        payload = (args.output_dir / "output-0.bin").read_bytes()
                        output_path = (
                            args.output_dir
                            / f"native-{session}-{fixture_index}-{position}.bin"
                        )
                        output_path.write_bytes(payload)
                        reference_payloads = {}
                        for kind in ("hf", "et"):
                            path = (
                                args.references.parent
                                / fixture["name"]
                                / f"{kind}-{position}.bin"
                            )
                            if file_hash(path) != fixture["hashes"][path.name]:
                                raise ValueError("Reference tensor hash changed")
                            reference_payloads[kind] = path.read_bytes()
                        row = {
                            "session": session,
                            "fixture": fixture["name"],
                            "position": position,
                            "input_token": token,
                            "phase": (
                                "prompt"
                                if position < fixture["prompt_token_count"] - 1
                                else "generation"
                            ),
                            "native_vs_hf": compare_payloads(
                                payload, reference_payloads["hf"], "bf16"
                            ),
                        }
                        if session == 0:
                            row["et_vs_hf"] = compare_payloads(
                                reference_payloads["et"],
                                reference_payloads["hf"],
                                "bf16",
                            )
                            row["native_vs_et"] = compare_payloads(
                                payload, reference_payloads["et"], "bf16"
                            )
                        else:
                            row["reset_byte_equal"] = (
                                payload
                                == (
                                    args.output_dir
                                    / f"native-0-{fixture_index}-{position}.bin"
                                ).read_bytes()
                            )
                            if not row["reset_byte_equal"]:
                                raise AssertionError(
                                    "Reset sequence is not byte-identical"
                                )
                        report["steps"].append(row)
                        print(json.dumps(row), flush=True)
            process.stdin.close()
            if process.wait(timeout=30) != 0:
                raise RuntimeError("Runner failed at shutdown")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    report["loaded_libraries"] = runtime_libraries(log_path.read_text())
    report["runtime_hashes"] = {
        path: file_hash(Path(path))
        for path in report["loaded_libraries"]
        if Path(path).name in ("libaoti_cuda_shims.so", "libextension_cuda.so")
    }
    if before != {str(path): file_hash(path) for path in tracked}:
        raise AssertionError("Input artifacts changed during comparison")
    report["summary"] = summarize_comparison(report["steps"])
    report["status"] = "comparison_complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("references", "pte", "ptd", "runner", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    args.references = args.references.resolve()
    args.pte = args.pte.resolve()
    args.ptd = args.ptd.resolve()
    args.runner = args.runner.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.repeats < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Repeats and timeout must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "teacher_forced": True,
        "runtime_validated": False,
        "steps": [],
    }
    try:
        compare(args, report)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (args.output_dir / "result.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
