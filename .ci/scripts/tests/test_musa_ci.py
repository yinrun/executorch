import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "test_musa.py"
SPEC = importlib.util.spec_from_file_location("musa_ci", SCRIPT)
musa_ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(musa_ci)


class MusaCiTest(unittest.TestCase):
    def arguments(self, *extra):
        return musa_ci.parse_args(
            [
                "--build-dir",
                "/tmp/musa-build",
                "--output-dir",
                "/tmp/musa-reports",
                *extra,
            ]
        )

    def test_native_build_is_musa_only_and_ctest_is_serial(self):
        commands = dict(
            musa_ci.commands(
                self.arguments("--suite", "native"),
                "cmake",
                "ctest",
                "flatc",
                "/torch/cmake",
            )
        )
        self.assertIn("-DEXECUTORCH_BUILD_MUSA=ON", commands["configure"])
        self.assertIn("-DEXECUTORCH_BUILD_CUDA=OFF", commands["configure"])
        self.assertIn("-DEXECUTORCH_BUILD_ROCM=OFF", commands["configure"])
        self.assertIn("-DBUILD_TESTING=ON", commands["configure"])
        self.assertEqual(
            commands["build"][-len(musa_ci.NATIVE_TARGETS) :],
            list(musa_ci.NATIVE_TARGETS),
        )
        self.assertIn("--no-tests=error", commands["native"])
        self.assertEqual(
            commands["native"][commands["native"].index("--parallel") + 1], "1"
        )
        self.assertIn("until-fail:3", commands["native"])
        self.assertNotIn("export", commands)

    def test_all_includes_export_runner_and_requested_model_matrix(self):
        args = self.arguments(
            "--models", "mlp", "densenet121", "--dtypes", "fp32", "bf16"
        )
        commands = dict(
            musa_ci.commands(args, "cmake", "ctest", "flatc", "/torch/cmake")
        )
        self.assertIn("executor_runner", commands["build"])
        self.assertIn("export", commands)
        self.assertIn("runner", commands)
        models = {
            name: command
            for name, command in commands.items()
            if name.startswith("model-")
        }
        self.assertEqual(
            set(models),
            {
                "model-mlp-fp32",
                "model-mlp-bf16",
                "model-densenet121-fp32",
                "model-densenet121-bf16",
            },
        )
        self.assertNotIn("--dynamic-spatial-max", models["model-mlp-fp32"])
        self.assertIn("--dynamic-spatial-max", models["model-densenet121-bf16"])
        self.assertIn("--runner", models["model-mlp-fp32"])
        self.assertIn("artifact-tests", commands)
        names = list(commands)
        for name in models:
            verification = "verify-" + name
            self.assertEqual(names[names.index(name) + 1], verification)
            self.assertTrue(commands[verification][-1].endswith("/result.json"))

    def test_environment_does_not_inherit_test_filters(self):
        with patch.dict(
            os.environ,
            {
                "GTEST_FILTER": "DoesNotExist",
                "GTEST_REPEAT": "0",
                "PYTHONPATH": "/deps",
            },
            clear=True,
        ):
            env = musa_ci.environment(
                Path("/build"),
                Path("/sdk"),
                "/tools/flatc",
                Path("/torch"),
                Path("/torch_musa"),
            )
        self.assertNotIn("GTEST_FILTER", env)
        self.assertNotIn("GTEST_REPEAT", env)
        self.assertEqual(env["EXECUTORCH_MUSA_RUNNER"], "/build/executor_runner")
        self.assertTrue(env["PYTHONPATH"].endswith(os.pathsep + "/deps"))
        self.assertTrue(env["LD_LIBRARY_PATH"].startswith("/build/backends/cuda"))

    def test_weights_are_forwarded_to_each_selected_dtype(self):
        args = self.arguments(
            "--suite",
            "models",
            "--models",
            "mobilenet_v2",
            "--weights",
            "mobilenet_v2",
            "/checkpoints/mobile.pth",
        )
        commands = dict(
            musa_ci.commands(args, "cmake", "ctest", "flatc", "/torch/cmake")
        )
        for dtype in ("fp32", "fp16", "bf16"):
            command = commands[f"model-mobilenet_v2-{dtype}"]
            self.assertEqual(
                command[command.index("--weights") + 1], "/checkpoints/mobile.pth"
            )
        for options in (
            ("--suite", "native", "--weights", "mobilenet_v2", "/weights"),
            ("--models", "mlp", "--weights", "mobilenet_v2", "/weights"),
            ("--weights", "mlp", "/weights"),
            (
                "--weights",
                "mobilenet_v2",
                "/first",
                "--weights",
                "mobilenet_v2",
                "/second",
            ),
        ):
            with self.subTest(options=options), patch(
                "sys.stderr", new_callable=io.StringIO
            ), self.assertRaises(SystemExit):
                self.arguments(*options)

    def test_failed_and_skipped_commands_fail_the_report(self):
        for name, code in (
            ("build", "raise SystemExit(7)"),
            ("native", "print('[  SKIPPED ] no GPU')"),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                args = self.arguments()
                args.output_dir = Path(directory)
                report = {"status": "running", "steps": []}
                with self.assertRaises(RuntimeError):
                    musa_ci.run_step(
                        name,
                        [sys.executable, "-c", code],
                        args,
                        os.environ.copy(),
                        report,
                    )
                saved = json.loads((args.output_dir / "summary.json").read_text())
                self.assertEqual(saved["steps"][0]["status"], "failed")
                self.assertTrue((args.output_dir / f"{name}.log").exists())

    def test_timeout_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments()
            args.output_dir = Path(directory)
            report = {"status": "running", "steps": []}
            with patch.object(
                musa_ci.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired("test", 1),
            ):
                with self.assertRaises(subprocess.TimeoutExpired):
                    musa_ci.run_step("native", ["test"], args, {}, report)
            self.assertEqual(report["steps"][0]["status"], "failed")

    def test_missing_device_overwrites_stale_success(self):
        torch = types.SimpleNamespace(
            musa=types.SimpleNamespace(is_available=lambda: False)
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            summary.write_text('{"status": "passed"}')
            with patch.dict(
                sys.modules, {"torch": torch, "torch_musa": types.SimpleNamespace()}
            ):
                result = musa_ci.main(["--output-dir", directory, "--suite", "native"])
            self.assertEqual(result, 1)
            self.assertEqual(json.loads(summary.read_text())["status"], "failed")

    def test_unittest_driver_rejects_empty_or_skipped_suites(self):
        modules = {
            "passing": "import unittest\nclass Test(unittest.TestCase):\n    def test_ok(self): pass\n",
            "skipped": "import unittest\nclass Test(unittest.TestCase):\n    @unittest.skip('no device')\n    def test_skip(self): pass\n",
            "empty": "",
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, source in modules.items():
                (Path(directory) / f"{name}.py").write_text(source)
                completed = subprocess.run(
                    [sys.executable, "-c", musa_ci.UNITTEST_DRIVER, name],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                )
                with self.subTest(module=name):
                    self.assertEqual(
                        completed.returncode,
                        0 if name == "passing" else 1,
                        completed.stderr,
                    )


if __name__ == "__main__":
    unittest.main()
