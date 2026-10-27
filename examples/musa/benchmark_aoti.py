"""Measure a packaged, stateless MUSA model using synchronized in-process samples."""

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def fingerprint(path: Path) -> dict:
    with path.open("rb") as stream:
        return {
            "nbytes": os.fstat(stream.fileno()).st_size,
            "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
        }


def verify_package(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    required = {
        "run.sh",
        "bin/executor_runner",
        "model/model.pte",
        "model/model.ptd",
        "lib/libaoti_cuda_shims.so",
        "lib/libextension_cuda.so",
    }
    if manifest.get("version") != 1 or not required.issubset(manifest.get("files", {})):
        raise ValueError("Incomplete or unsupported runtime package manifest")
    for name, expected in manifest["files"].items():
        path = root / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(
            root.resolve()
        ):
            raise ValueError(f"Invalid package path: {name}")
        if fingerprint(path) != expected:
            raise ValueError(f"Package checksum mismatch: {name}")
    return manifest


def summarize(samples: list[float]) -> dict:
    if not samples or any(
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        for value in samples
    ):
        raise ValueError("Timing samples must be finite and positive")
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p90_ms": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "max_ms": max(samples),
    }


def validate_samples(data: dict, warmup: int, iterations: int) -> None:
    if (
        data.get("version") != 1
        or data.get("synchronization") != "musaDeviceSynchronize"
        or data.get("scope") != "host_synchronized_method"
        or data.get("warmup_executions") != warmup
        or data.get("num_executions") != iterations
        or type(data.get("device_index")) is not int
        or data["device_index"] < 0
    ):
        raise ValueError("Unexpected benchmark configuration or synchronization")
    for key in ("execute_ms", "prepare_execute_ms"):
        if len(data.get(key, [])) != iterations:
            raise ValueError("Missing timing samples")
        summarize(data[key])
    if any(
        total < execute
        for total, execute in zip(data["prepare_execute_ms"], data["execute_ms"])
    ):
        raise ValueError("Inclusive timing is shorter than execution timing")


def positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def host_environment(sysfs_root: Path = Path("/sys")) -> dict:
    context = {"timestamp_utc": datetime.now(timezone.utc).isoformat()}
    try:
        context["launcher_cpu_affinity"] = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as error:
        context["launcher_cpu_affinity_error"] = str(error)
    for key, relative, attributes in (
        (
            "cpu_frequency_policies",
            "devices/system/cpu/cpufreq",
            (
                "affected_cpus",
                "scaling_driver",
                "scaling_governor",
                "scaling_min_freq",
                "scaling_max_freq",
                "scaling_cur_freq",
            ),
        ),
        (
            "device_frequency_policies",
            "class/devfreq",
            ("name", "governor", "min_freq", "max_freq", "cur_freq"),
        ),
    ):
        directory = sysfs_root / relative
        policies = {"path": str(directory), "entries": []}
        context[key] = policies
        try:
            entries = sorted(directory.iterdir())
        except OSError as error:
            policies["error"] = str(error)
            continue
        for entry in entries:
            if key == "cpu_frequency_policies" and not entry.name.startswith("policy"):
                continue
            if not entry.is_dir():
                continue
            record = {"name": entry.name, "values": {}, "errors": {}}
            for attribute in attributes:
                try:
                    record["values"][attribute] = (
                        (entry / attribute).read_text().strip()
                    )
                except OSError as error:
                    record["errors"][attribute] = str(error)
            policies["entries"].append(record)
    return context


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--input-shapes", required=True)
    parser.add_argument(
        "--reference-output",
        type=Path,
        nargs="+",
        required=True,
        help="Previously accepted runner tensor outputs for these exact inputs; compared byte-for-byte",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=positive, default=10)
    parser.add_argument("--iterations", type=positive, default=50)
    parser.add_argument("--repeats", type=positive, default=5)
    parser.add_argument("--timeout", type=positive, default=600)
    parser.add_argument(
        "--musa-home",
        type=Path,
        default=Path(os.environ.get("MUSA_HOME", "/usr/local/musa")),
    )
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    package = args.package.resolve()
    report = {
        "status": "running",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "package": str(package),
        "musa_home": str(args.musa_home.resolve()),
        "input_shapes": args.input_shapes,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "scope": "host_synchronized_method",
        "processes": [],
    }
    try:
        if "," in str(root):
            raise ValueError(
                "Output directory cannot contain commas (runner input delimiter)"
            )
        manifest = verify_package(package)
        manifest_hash = fingerprint(package / "manifest.json")
        report["package_manifest"] = manifest
        report["package_manifest_fingerprint"] = manifest_hash
        inputs = []
        references = []
        records = {}
        for kind, sources, destinations in (
            ("input", args.inputs, inputs),
            ("reference", args.reference_output, references),
        ):
            for index, source in enumerate(sources):
                destination = root / f"{kind}-{index}.bin"
                shutil.copyfile(source, destination)
                destinations.append(destination)
                records[destination.name] = fingerprint(destination)
        report["artifacts"] = records
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(root),
            "MUSA_HOME": str(args.musa_home.resolve()),
            "LD_BIND_NOW": "1",
        }
        for repeat in range(args.repeats):
            sample = root / f"process-{repeat}"
            sample.mkdir()
            timing = sample / "timing.json"
            command = [
                str(package / "run.sh"),
                f"--inputs={','.join(map(str, inputs))}",
                f"--input_shapes={args.input_shapes}",
                f"--output_file={sample / 'output'}",
                "--print_output=none",
                f"--musa_benchmark_output={timing}",
                f"--musa_warmup_executions={args.warmup}",
                f"--num_executions={args.iterations}",
            ]
            host_before = host_environment()
            try:
                with (sample / "stdout.log").open("w") as stdout, (
                    sample / "stderr.log"
                ).open("w") as stderr:
                    subprocess.run(
                        command,
                        env=environment,
                        cwd=sample,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=args.timeout,
                        check=True,
                    )
            finally:
                host_after = host_environment()
                report["last_process_context"] = {
                    "command": command,
                    "host_before": host_before,
                    "host_after": host_after,
                }
            data = json.loads(timing.read_text())
            validate_samples(data, args.warmup, args.iterations)
            outputs = [
                sample / f"output-{index}.bin" for index in range(len(references))
            ]
            if set(sample.glob("output-*.bin")) != set(outputs):
                raise ValueError("Unexpected tensor output count")
            for output, reference in zip(outputs, references):
                if fingerprint(output) != records[reference.name]:
                    raise ValueError(
                        f"Output differs from accepted reference: {output}"
                    )
            report["processes"].append(
                {
                    "command": command,
                    "host_before": host_before,
                    "host_after": host_after,
                    "timing": data,
                    "output_fingerprints": [fingerprint(path) for path in outputs],
                    "execute": summarize(data["execute_ms"]),
                    "prepare_execute": summarize(data["prepare_execute_ms"]),
                }
            )
            (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if (
            verify_package(package) != manifest
            or fingerprint(package / "manifest.json") != manifest_hash
        ):
            raise ValueError("Package changed during benchmarking")
        for name, expected in records.items():
            if fingerprint(root / name) != expected:
                raise ValueError(f"Benchmark artifact changed: {name}")
        for metric in ("execute_ms", "prepare_execute_ms"):
            report[metric] = summarize(
                [
                    value
                    for process in report["processes"]
                    for value in process["timing"][metric]
                ]
            )
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["error"] = str(error)
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "summary": str(root / "summary.json"),
                "execute_ms": report.get("execute_ms"),
                "error": report.get("error"),
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
