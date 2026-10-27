"""Build and run MUSA regressions on a host with an installed TorchMUSA SDK."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NATIVE_TARGETS = tuple(
    f"test_musa_{name}"
    for name in (
        "addmm",
        "convolution",
        "batch_norm",
        "activation",
        "device",
        "storage",
    )
)
MODELS = (
    "mlp",
    "simple_cnn",
    "resnet18",
    "mobilenet_v2",
    "efficientnet_b0",
    "densenet121",
    "convnext_tiny",
)
UNITTEST_DRIVER = """
import sys
import unittest
suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[1])
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.testsRun == 0 or result.skipped:
    print('MUSA CI requires executed tests without skips', file=sys.stderr)
    sys.exit(1)
sys.exit(0 if result.wasSuccessful() else 1)
"""


def positive_integer(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("all", "native", "export", "runner", "models", "deployment"),
        default="all",
    )
    parser.add_argument("--build-dir", type=Path, default=ROOT / "cmake-out-musa")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cmake", default=os.environ.get("CMAKE", "cmake"))
    parser.add_argument("--ctest")
    parser.add_argument("--flatc", default="flatc")
    parser.add_argument("--patchelf", default="patchelf")
    parser.add_argument(
        "--musa-home",
        type=Path,
        default=Path(os.environ.get("MUSA_HOME", "/usr/local/musa")),
    )
    parser.add_argument(
        "--jobs", type=positive_integer, default=min(os.cpu_count() or 1, 8)
    )
    parser.add_argument("--native-repeats", type=positive_integer, default=3)
    parser.add_argument("--runner-repeats", type=positive_integer, default=2)
    parser.add_argument("--timeout", type=positive_integer, default=3600)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS[:4]))
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=("fp32", "fp16", "bf16"),
        default=["fp32", "fp16", "bf16"],
    )
    parser.add_argument(
        "--weights", action="append", nargs=2, default=[], metavar=("MODEL", "PATH")
    )
    args = parser.parse_args(argv)
    weights = {}
    for model, path in args.weights:
        if args.suite not in ("all", "models") or model not in args.models:
            parser.error("--weights requires a selected model in the all/models suite")
        if model not in MODELS[2:]:
            parser.error("--weights requires a torchvision model")
        if model in weights:
            parser.error(f"Duplicate weights for {model}")
        weights[model] = Path(path).expanduser().resolve()
    args.weights = weights
    return args


def require_tool(name):
    path = shutil.which(str(name))
    if path is None:
        raise RuntimeError(f"Required tool is unavailable: {name}")
    return str(Path(path).resolve())


def environment(build_dir, musa_home, flatc, torch_root, torch_musa_root):
    result = {
        key: value for key, value in os.environ.items() if not key.startswith("GTEST_")
    }
    for key, paths in (
        ("PYTHONPATH", [ROOT / "src", ROOT]),
        ("PATH", [Path(flatc).parent, musa_home / "bin"]),
        (
            "LD_LIBRARY_PATH",
            [
                build_dir / "backends/cuda",
                build_dir / "extension/cuda",
                torch_musa_root / "lib",
                torch_root / "lib",
                musa_home / "lib",
            ],
        ),
    ):
        result[key] = os.pathsep.join(
            [
                *(str(path) for path in paths),
                *([result[key]] if result.get(key) else []),
            ]
        )
    result["EXECUTORCH_MUSA_RUNNER"] = str(build_dir / "executor_runner")
    result["MUSA_HOME"] = str(musa_home)
    return result


def commands(args, cmake, ctest, flatc, torch_prefix):
    selected = (
        {"native", "export", "runner", "models", "deployment"}
        if args.suite == "all"
        else {args.suite}
    )
    targets = list(NATIVE_TARGETS) if "native" in selected else []
    if selected & {"runner", "models", "deployment"}:
        targets.append("executor_runner")
    if targets:
        yield "configure", [
            cmake,
            "-S",
            str(ROOT),
            "-B",
            str(args.build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_PREFIX_PATH={torch_prefix}",
            f"-DCMAKE_MODULE_PATH={args.musa_home / 'cmake'}",
            f"-DMUSAToolkit_ROOT={args.musa_home}",
            f"-DFLATC_EXECUTABLE={flatc}",
            "-DEXECUTORCH_BUILD_MUSA=ON",
            "-DEXECUTORCH_BUILD_CUDA=OFF",
            "-DEXECUTORCH_BUILD_ROCM=OFF",
            "-DEXECUTORCH_BUILD_TESTS=ON",
            "-DBUILD_TESTING=ON",
            "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON",
            "-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON",
            "-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON",
            "-DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON",
            "-DEXECUTORCH_BUILD_XNNPACK=OFF",
            "-DEXECUTORCH_BUILD_CPUINFO=OFF",
            "-DEXECUTORCH_BUILD_PTHREADPOOL=OFF",
        ]
        yield "build", [
            cmake,
            "--build",
            str(args.build_dir),
            "--parallel",
            str(args.jobs),
            "--target",
            *targets,
        ]
    if "native" in selected:
        yield "native", [
            ctest,
            "--test-dir",
            str(args.build_dir),
            "--verbose",
            "--output-on-failure",
            "--no-tests=error",
            "--parallel",
            "1",
            "-L",
            "^musa$",
            "--repeat",
            f"until-fail:{args.native_repeats}",
            "--output-junit",
            str(args.output_dir / "native.xml"),
        ]
    for suite, module in (("export", "test_musa_aoti"), ("runner", "test_musa_runner")):
        if suite in selected:
            yield suite, [
                sys.executable,
                "-c",
                UNITTEST_DRIVER,
                f"backends.musa.tests.{module}",
            ]
    if "deployment" in selected:
        for module in ("test_musa_package", "test_musa_benchmark", "test_musa_deployment"):
            yield module, [
                sys.executable,
                "-c",
                UNITTEST_DRIVER,
                f"backends.musa.tests.{module}",
            ]
    if "models" in selected:
        yield "artifact-tests", [
            sys.executable,
            str(ROOT / ".ci/scripts/tests/test_musa_artifacts.py"),
            "-v",
        ]
        for model in dict.fromkeys(args.models):
            for dtype in dict.fromkeys(args.dtypes):
                command = [
                    sys.executable,
                    str(ROOT / "examples/musa/aoti_model_acceptance.py"),
                    "--model",
                    model,
                    "--dtype",
                    dtype,
                    "--batch-size",
                    "2",
                    "--dynamic-batch-max",
                    "3",
                    "--runner-repeats",
                    str(args.runner_repeats),
                    "--runner",
                    str(args.build_dir / "executor_runner"),
                    "--output-dir",
                    str(args.output_dir / "models" / f"{model}-{dtype}"),
                ]
                if model in {
                    "resnet18",
                    "mobilenet_v2",
                    "efficientnet_b0",
                    "densenet121",
                }:
                    command += ["--dynamic-spatial-max", "80"]
                if model in args.weights:
                    command += ["--weights", str(args.weights[model])]
                yield f"model-{model}-{dtype}", command
                yield f"verify-model-{model}-{dtype}", [
                    sys.executable,
                    str(ROOT / "examples/musa/acceptance_artifacts.py"),
                    str(
                        args.output_dir / "models" / f"{model}-{dtype}" / "result.json"
                    ),
                ]


def run_step(name, command, args, env, report):
    log_path = args.output_dir / f"{name}.log"
    step = {"name": name, "command": command, "log": str(log_path), "status": "running"}
    report["steps"].append(step)
    started = time.monotonic()
    print(f"[{name}] log: {log_path}", flush=True)
    try:
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
        step["returncode"] = completed.returncode
        if completed.returncode:
            raise RuntimeError(
                f"{name} failed with exit code {completed.returncode}; see {log_path}"
            )
        if name == "native" and re.search(
            r"\[\s*SKIPPED\s*\]", log_path.read_text(errors="replace")
        ):
            raise RuntimeError(f"Native tests skipped GPU execution; see {log_path}")
        step["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        step["status"] = "failed"
        step["error"] = str(error)
        raise
    finally:
        step["seconds"] = round(time.monotonic() - started, 3)
        (args.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )


def main(argv=None):
    args = parse_args(argv)
    args.build_dir = args.build_dir.resolve()
    args.musa_home = args.musa_home.resolve()
    args.output_dir = (args.output_dir or args.build_dir / "musa-reports").resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "suite": args.suite,
        "build_dir": str(args.build_dir),
        "steps": [],
    }
    summary = args.output_dir / "summary.json"
    summary.write_text(json.dumps(report, indent=2) + "\n")
    try:
        import torch
        import torch_musa

        if not torch.musa.is_available() or torch.musa.device_count() < 1:
            raise RuntimeError(
                "MUSA CI requires a usable MUSA GPU; skipped tests are not success"
            )
        for model, path in args.weights.items():
            if not path.is_file():
                raise RuntimeError(f"Weights for {model} are missing: {path}")
        report["weights"] = {model: str(path) for model, path in args.weights.items()}
        cmake = require_tool(args.cmake)
        ctest = require_tool(args.ctest or Path(cmake).with_name("ctest"))
        flatc = require_tool(args.flatc)
        if not (args.musa_home / "cmake/FindMUSAToolkit.cmake").is_file():
            raise RuntimeError(f"MUSA SDK CMake module missing under {args.musa_home}")
        report.update(
            {
                "python": sys.version,
                "torch": str(torch.__version__),
                "torch_musa": str(torch_musa.__version__),
                "device": torch.musa.get_device_name(0),
                "musa_home": str(args.musa_home),
                "source_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "source_dirty": bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"], cwd=ROOT, text=True
                    ).strip()
                ),
            }
        )
        env = environment(
            args.build_dir,
            args.musa_home,
            flatc,
            Path(torch.__file__).parent,
            Path(torch_musa.__file__).parent,
        )
        run_step("flatc-version", [flatc, "--version"], args, env, report)
        if args.suite in ("all", "deployment"):
            env["EXECUTORCH_MUSA_PATCHELF"] = require_tool(args.patchelf)
            require_tool("readelf")
        for name, command in commands(
            args, cmake, ctest, flatc, torch.utils.cmake_prefix_path
        ):
            run_step(name, command, args, env, report)
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["error"] = str(error)
        print(f"MUSA regression failed: {error}", file=sys.stderr)
    finally:
        summary.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {summary}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
