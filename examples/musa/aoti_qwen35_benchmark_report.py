"""Summarize native Qwen3.5-4B timings without implying accuracy acceptance."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def _count(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid {name}")
    return value


def _milliseconds(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Expected finite positive {name}")
    return value


def _distribution(values):
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "samples": values,
    }


def summarize_generation(data):
    required = {
        "version": 1,
        "status": "completed",
        "model": "Qwen3.5-4B-GGUF",
        "context_capacity": 4096,
        "prefill_mode": "serial_token",
        "controller": "native_cpp",
        "scope": "synchronized_prepare_execute_output_validation",
        "vision_encoding_included": False,
        "logits_dump_enabled": False,
    }
    for name, expected in required.items():
        if type(data.get(name)) is not type(expected) or data[name] != expected:
            raise ValueError(f"Unsupported generation field: {name}")
    warmup = _count(data.get("warmup_runs"), "warmup_runs", 1)
    measured = _count(data.get("measured_runs"), "measured_runs", 3)
    maximum_generated = _count(
        data.get("maximum_generated_tokens"), "maximum_generated_tokens", 1
    )
    if warmup > 100 or measured > 100 or maximum_generated > 4096:
        raise ValueError("Generation counts exceed runner limits")
    runs = data.get("runs")
    if not isinstance(runs, list) or len(runs) != warmup + measured:
        raise ValueError("Incomplete generation runs")
    first_ids = None
    first_prompt = None
    first_stop = None
    prompt_rates, decode_rates, loop_times = [], [], []
    for index, run in enumerate(runs):
        if type(run.get("run")) is not int or run["run"] != index:
            raise ValueError("Nonsequential run index")
        if type(run.get("warmup")) is not bool or run["warmup"] != (index < warmup):
            raise ValueError("Incorrect warmup classification")
        prompt = _count(run.get("prompt_tokens"), "prompt_tokens", 1)
        decode = _count(run.get("decode_steps"), "decode_steps", 1)
        steps = _count(run.get("steps"), "steps", 1)
        generated = run.get("generated_token_ids")
        if (
            not isinstance(generated, list)
            or len(generated) != decode + 1
            or len(generated) > maximum_generated
            or any(
                type(token) is not int or not 0 <= token < 248320 for token in generated
            )
            or steps != prompt + decode
            or steps > 4096
        ):
            raise ValueError("Inconsistent token counts or IDs")
        stop = run.get("stop_reason")
        if stop not in ("eos", "context_limit", "max_new_tokens"):
            raise ValueError("Invalid stop reason")
        if stop == "context_limit" and steps != 4096:
            raise ValueError("Context stop did not reach capacity")
        if stop == "max_new_tokens" and len(generated) != maximum_generated:
            raise ValueError("Token limit stop has an incorrect length")
        if first_ids is None:
            first_ids, first_prompt, first_stop = generated, prompt, stop
        if generated != first_ids or prompt != first_prompt or stop != first_stop:
            raise ValueError("Generation workload changed between runs")
        prompt_ms = _milliseconds(run.get("prefill_ms"), "prefill_ms")
        decode_ms = _milliseconds(run.get("decode_ms"), "decode_ms")
        loop_ms = _milliseconds(run.get("loop_ms"), "loop_ms")
        if loop_ms + 1e-6 < prompt_ms + decode_ms:
            raise ValueError("Loop time is shorter than its measured phases")
        if index >= warmup:
            prompt_rates.append(prompt * 1000.0 / prompt_ms)
            decode_rates.append(decode * 1000.0 / decode_ms)
            loop_times.append(loop_ms)
    return {
        "scope": required["scope"],
        "warmup_runs_excluded": warmup,
        "measured_runs": measured,
        "prompt_tokens": first_prompt,
        "generated_tokens_including_eos_if_present": len(first_ids),
        "decode_steps": len(first_ids) - 1,
        "stop_reason": first_stop,
        "generated_token_ids": first_ids,
        "serial_prompt_tokens_per_second": _distribution(prompt_rates),
        "decode_tokens_per_second": _distribution(decode_rates),
        "generation_loop_ms": _distribution(loop_times),
        "batched_prefill": False,
        "pure_gpu_time": False,
        "includes_loading_tokenization_vision": False,
        "independent_accuracy_validated": False,
    }


def summarize_vision(data):
    if (
        data.get("scope") != "host_synchronized_method"
        or data.get("synchronization") != "musaDeviceSynchronize"
        or type(data.get("version")) is not int
        or data["version"] != 1
    ):
        raise ValueError("Unsupported vision timing scope")
    device = _count(data.get("device_index"), "device_index")
    warmup = _count(data.get("warmup_executions"), "warmup_executions", 1)
    measured = _count(data.get("num_executions"), "num_executions", 3)
    arrays = {}
    for name in ("execute_ms", "prepare_execute_ms"):
        values = data.get(name)
        if not isinstance(values, list) or len(values) != measured:
            raise ValueError(f"Incomplete {name} samples")
        arrays[name] = [_milliseconds(value, name) for value in values]
    if any(
        prepare + 1e-6 < execute
        for prepare, execute in zip(arrays["prepare_execute_ms"], arrays["execute_ms"])
    ):
        raise ValueError("Prepare/execute time is shorter than execute time")
    return {
        "scope": data["scope"],
        "device_index": device,
        "warmup_executions_excluded": warmup,
        "measured_executions": measured,
        **{name: _distribution(values) for name, values in arrays.items()},
        "pure_gpu_time": False,
        "includes_loading_preprocessing": False,
        "independent_accuracy_validated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, action="append", default=[])
    parser.add_argument("--vision", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.generation and not args.vision:
        parser.error("At least one native report is required")
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {
        "status": "summarized_native_timings_only",
        "independent_accuracy_validated": False,
        "speedup_claim": False,
        "results": [],
    }
    for kind, paths, summarize in (
        ("generation", args.generation, summarize_generation),
        ("vision", args.vision, summarize_vision),
    ):
        for path in paths:
            payload = path.read_bytes()
            report["results"].append(
                {
                    "kind": kind,
                    "source": str(path.resolve()),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "summary": summarize(json.loads(payload)),
                }
            )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
