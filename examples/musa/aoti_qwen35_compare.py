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
    read_greedy_token,
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


def select_input_token(fixture, position, predicted, free_running):
    expected = fixture["token_ids"][position]
    if not free_running or position < fixture["prompt_token_count"]:
        return expected
    if type(predicted) is not int or not 0 <= predicted < 248320:
        raise ValueError("Free generation requires a valid native prediction")
    if predicted != expected:
        raise ValueError(
            f"Native trajectory diverged at {fixture['name']} position {position}: "
            f"{predicted} != {expected}; reference logits no longer share inputs"
        )
    return predicted


def select_fixtures(fixtures, requested):
    if not requested:
        return fixtures
    names = [fixture["name"] for fixture in fixtures]
    if len(names) != len(set(names)):
        raise ValueError("Reference fixture names must be unique")
    if len(requested) != len(set(requested)):
        raise ValueError("Requested fixture names must be unique")
    unknown = set(requested) - set(names)
    if unknown:
        raise ValueError(f"Unknown reference fixtures: {sorted(unknown)}")
    return [fixture for fixture in fixtures if fixture["name"] in requested]


def musa_cache_arguments(cache_bytes):
    if type(cache_bytes) is not int or not 0 <= cache_bytes < 2**64:
        raise ValueError("MUSA cache bytes must be an unsigned 64-bit integer")
    return [f"--musa_cache_bytes={cache_bytes}"] if cache_bytes else []


def native_eos_reached(fixture, position, predicted, eos_token_id):
    if position < fixture["prompt_token_count"] - 1:
        return False
    final_position = position == len(fixture["token_ids"]) - 1
    expects_eos = fixture.get("stop_reason") == "eos"
    if predicted == eos_token_id:
        if not expects_eos or not final_position:
            raise ValueError("Native EOS does not match the reference stopping point")
        return True
    if expects_eos and final_position:
        raise ValueError("Native generation did not reach the reference EOS")
    return False


def validated_context_length(manifest, pte, ptd, metadata_path, gpu_greedy=False):
    context_length = manifest.get("context_length")
    if type(context_length) is not int or context_length < 2:
        raise ValueError("Invalid reference context length")
    if metadata_path is None:
        if context_length != 64:
            raise ValueError("Nonlegacy contexts require --export-metadata")
        return context_length
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("status") != "exported"
        or metadata.get("model") != "Qwen3.5-0.8B-text"
        or metadata.get("dtype") != "bf16"
        or type(metadata.get("context_length")) is not int
        or metadata["context_length"] != context_length
        or metadata.get("delegates") != 1
    ):
        raise ValueError("Export metadata does not match the BF16 reference context")
    for path in (pte, ptd):
        if metadata.get("artifacts", {}).get(path.name) != file_hash(path):
            raise ValueError("Export metadata package hash mismatch")
    if metadata.get("checkpoint_hashes") != {
        Path(path).name: digest
        for path, digest in manifest["checkpoint_hashes"].items()
    }:
        raise ValueError("Export and reference checkpoint hashes differ")
    outputs = metadata.get("output_indices", {})
    if outputs.get("logits") != 0 or (gpu_greedy and outputs.get("greedy_token") != 1):
        raise ValueError("Export metadata does not declare the requested outputs")
    return context_length


def musa_cache_report(log):
    reports = [
        json.loads(line.removeprefix("MUSA ALLOCATION CACHE "))
        for line in log.splitlines()
        if line.startswith("MUSA ALLOCATION CACHE {")
    ]
    if len(reports) != 1:
        raise AssertionError("Expected one native allocation-cache teardown report")
    report = reports[0]
    for name in (
        "cached_bytes",
        "cached_blocks",
        "live_blocks",
        "reuses",
        "allocations",
    ):
        if type(report.get(name)) is not int or report[name] < 0:
            raise AssertionError(f"Invalid allocation-cache counter: {name}")
    if report["cached_bytes"] or report["cached_blocks"] or report["live_blocks"]:
        raise AssertionError("Native allocation cache did not drain at teardown")
    return report


def compare(args, report):
    manifest = json.loads(args.references.read_text())
    if manifest["status"] != "complete" or manifest["dtype"] != "bf16":
        raise ValueError("Expected a complete BF16 reference manifest")
    if not manifest["fixtures"]:
        raise ValueError("Expected at least one reference fixture")
    context_length = validated_context_length(
        manifest, args.pte, args.ptd, args.export_metadata, args.gpu_greedy
    )
    report["context_length"] = context_length
    fixtures = select_fixtures(manifest["fixtures"], args.fixture)
    report["selected_fixtures"] = [fixture["name"] for fixture in fixtures]
    eos_token_id = manifest.get("eos_token_id")
    if eos_token_id is not None and (
        type(eos_token_id) is not int or not 0 <= eos_token_id < 248320
    ):
        raise ValueError("Invalid reference EOS token ID")
    if eos_token_id is None and any(
        fixture.get("stop_reason") == "eos" for fixture in fixtures
    ):
        raise ValueError("EOS references must declare eos_token_id")
    for fixture in manifest["fixtures"]:
        if not 1 <= len(fixture["token_ids"]) <= context_length:
            raise ValueError("Fixture must fit the exported context")
        if not 1 <= fixture["prompt_token_count"] <= len(fixture["token_ids"]):
            raise ValueError("Invalid prompt token count")
        if any(
            type(token) is not int or not 0 <= token < 248320
            for token in fixture["token_ids"]
        ):
            raise ValueError("Invalid token ID")
    tracked = [args.references, args.pte, args.ptd, args.runner]
    if args.export_metadata:
        tracked.append(args.export_metadata)
    if args.runtime_preload:
        tracked.append(args.runtime_preload)
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
        *musa_cache_arguments(args.musa_cache_bytes),
    ]
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(args.output_dir),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "LD_BIND_NOW": "1",
        "LD_DEBUG": "libs",
    }
    if args.runtime_preload:
        environment["LD_PRELOAD"] = str(args.runtime_preload)
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
                for fixture_index, fixture in enumerate(fixtures):
                    predicted = None
                    for position in range(len(fixture["token_ids"])):
                        token = select_input_token(
                            fixture, position, predicted, args.free_running
                        )
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
                        predicted = row["native_vs_hf"]["predicted_token"]
                        if args.gpu_greedy:
                            greedy_payload = (
                                args.output_dir / "output-1.bin"
                            ).read_bytes()
                            gpu_token = read_greedy_token(greedy_payload)
                            row["gpu_greedy_token"] = gpu_token
                            if gpu_token != predicted:
                                raise AssertionError(
                                    "GPU greedy token differs from the full-logit argmax"
                                )
                            predicted = gpu_token
                        if args.free_running and eos_token_id is not None:
                            row["native_eos"] = native_eos_reached(
                                fixture, position, predicted, eos_token_id
                            )
                        print(json.dumps(row), flush=True)
                        if row.get("native_eos"):
                            break
            process.stdin.close()
            if process.wait(timeout=30) != 0:
                raise RuntimeError("Runner failed at shutdown")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    report["loaded_libraries"] = runtime_libraries(log_path.read_text())
    if args.musa_cache_bytes:
        report["allocation_cache"] = musa_cache_report(log_path.read_text())
    if (
        args.runtime_preload
        and str(args.runtime_preload) not in report["loaded_libraries"]
    ):
        raise AssertionError("Requested runtime preload was not observed in the audit")
    report["runtime_hashes"] = {
        path: file_hash(Path(path))
        for path in report["loaded_libraries"]
        if Path(path).name in ("libaoti_cuda_shims.so", "libextension_cuda.so")
    }
    if before != {str(path): file_hash(path) for path in tracked}:
        raise AssertionError("Input artifacts changed during comparison")
    report["summary"] = summarize_comparison(report["steps"])
    if args.free_running and eos_token_id is not None:
        report["native_eos_stops"] = sum(
            row.get("native_eos", False) for row in report["steps"]
        )
    if args.free_running:
        report["summary"]["scope"] = "native_greedy_feedback_against_hf_generation"
    report["numerical_tolerance_passed"] = all(
        row["native_vs_hf"]["outside_legacy_tolerance"] == 0 for row in report["steps"]
    )
    report["generated_token_ids_match"] = all(
        row["native_vs_hf"]["predicted_token"] == row["native_vs_hf"]["reference_token"]
        for row in report["steps"]
        if row["phase"] == "generation"
    )
    report["status"] = "comparison_complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("references", "pte", "ptd", "runner", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--export-metadata",
        type=Path,
        help="Hash-matched exporter result.json; required for contexts other than 64",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--gpu-greedy",
        action="store_true",
        help="Validate and feed back the GPU int64 token in output 1",
    )
    parser.add_argument(
        "--musa-cache-bytes",
        type=int,
        default=0,
        help="Opt into a bounded native MUSA allocation cache (requires a supporting runner)",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        help="Compare only this named fixture; repeat to select more, in manifest order",
    )
    parser.add_argument(
        "--runtime-preload",
        type=Path,
        help="Explicit, hash-tracked isolated runtime experiment",
    )
    parser.add_argument(
        "--free-running",
        action="store_true",
        help="Feed back native argmax after the prompt; stop on trajectory divergence",
    )
    args = parser.parse_args()
    args.references = args.references.resolve()
    args.pte = args.pte.resolve()
    args.ptd = args.ptd.resolve()
    args.runner = args.runner.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.export_metadata:
        args.export_metadata = args.export_metadata.resolve()
    if args.runtime_preload:
        args.runtime_preload = args.runtime_preload.resolve()
    if args.repeats < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Repeats and timeout must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "teacher_forced": not args.free_running,
        "sampling": "native_gpu_argmax" if args.gpu_greedy else "controller_argmax",
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
