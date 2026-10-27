import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from examples.musa.benchmark_aoti import (
    fingerprint,
    host_environment,
    main,
    summarize,
    validate_samples,
    verify_package,
)


class TestMusaBenchmark(unittest.TestCase):
    def test_failed_runner_retains_host_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            (package / "manifest.json").write_text("{}")
            source = root / "input.bin"
            source.write_bytes(b"input")
            reference = root / "reference.bin"
            reference.write_bytes(b"reference")
            args = SimpleNamespace(
                package=package,
                inputs=[source],
                input_shapes="1x1",
                reference_output=[reference],
                output_dir=root / "output",
                warmup=1,
                iterations=1,
                repeats=1,
                timeout=10,
                musa_home=root / "musa",
            )
            before = {"launcher_cpu_affinity": [0, 1]}
            after = {"launcher_cpu_affinity": [1]}
            with (
                mock.patch("argparse.ArgumentParser.parse_args", return_value=args),
                mock.patch(
                    "examples.musa.benchmark_aoti.verify_package", return_value={}
                ),
                mock.patch(
                    "examples.musa.benchmark_aoti.host_environment",
                    side_effect=[before, after],
                ),
                mock.patch(
                    "subprocess.run",
                    side_effect=subprocess.CalledProcessError(3, ["runner"]),
                ),
                mock.patch("builtins.print"),
            ):
                self.assertEqual(main(), 1)
            report = json.loads((args.output_dir / "summary.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["last_process_context"]["host_before"], before)
            self.assertEqual(report["last_process_context"]["host_after"], after)
            self.assertEqual(report["processes"], [])

    def test_host_environment_records_affinity_and_sysfs_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpu = root / "devices/system/cpu/cpufreq/policy0"
            device = root / "class/devfreq/test-device"
            cpu.mkdir(parents=True)
            (cpu.parent / "ondemand").mkdir()
            device.mkdir(parents=True)
            (cpu / "affected_cpus").write_text("0 1 2 3\n")
            (cpu / "scaling_governor").write_text("ondemand\n")
            (device / "cur_freq").write_text("730000000\n")
            (root / "class/devfreq/not-a-policy").write_text("ignored")
            with mock.patch(
                "os.sched_getaffinity", return_value={3, 0, 2, 1}, create=True
            ):
                context = host_environment(root)
            self.assertEqual(context["launcher_cpu_affinity"], [0, 1, 2, 3])
            cpus = context["cpu_frequency_policies"]["entries"]
            self.assertEqual(len(cpus), 1)
            self.assertEqual(cpus[0]["values"]["affected_cpus"], "0 1 2 3")
            self.assertEqual(cpus[0]["values"]["scaling_governor"], "ondemand")
            self.assertIn("scaling_driver", cpus[0]["errors"])
            devices = context["device_frequency_policies"]["entries"]
            self.assertEqual(len(devices), 1)
            self.assertEqual(devices[0]["values"]["cur_freq"], "730000000")

    def test_host_environment_unavailable_sources_are_not_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "os.sched_getaffinity", side_effect=OSError("unavailable"), create=True
            ):
                context = host_environment(Path(directory))
            self.assertIn("unavailable", context["launcher_cpu_affinity_error"])
            self.assertNotIn("launcher_cpu_affinity", context)
            for key in ("cpu_frequency_policies", "device_frequency_policies"):
                self.assertEqual(context[key]["entries"], [])
                self.assertIn("error", context[key])

    def test_host_environment_preserves_individual_read_errors(self):
        original = Path.read_text

        def read_text(path, *args, **kwargs):
            if path.name == "scaling_cur_freq":
                raise OSError("Device or resource busy")
            return original(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "devices/system/cpu/cpufreq/policy0"
            policy.mkdir(parents=True)
            (policy / "scaling_governor").write_text("ondemand")
            with mock.patch.object(Path, "read_text", read_text):
                context = host_environment(root)
            record = context["cpu_frequency_policies"]["entries"][0]
            self.assertEqual(record["values"]["scaling_governor"], "ondemand")
            self.assertIn("busy", record["errors"]["scaling_cur_freq"])

    def test_statistics_and_nearest_rank_percentile(self):
        stats = summarize([4, 1, 3, 2])
        self.assertEqual(
            stats,
            {
                "count": 4,
                "min_ms": 1,
                "median_ms": 2.5,
                "mean_ms": 2.5,
                "p90_ms": 4,
                "max_ms": 4,
            },
        )

    def test_nonfinite_or_empty_samples_rejected(self):
        for values in ([], [float("nan")], [float("inf")], [-1], [0], [True]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                summarize(values)

    def test_sample_counts_sync_and_inclusive_timing(self):
        data = {
            "version": 1,
            "scope": "host_synchronized_method",
            "synchronization": "musaDeviceSynchronize",
            "device_index": 0,
            "warmup_executions": 3,
            "num_executions": 2,
            "execute_ms": [1, 2],
            "prepare_execute_ms": [2, 3],
        }
        validate_samples(data, 3, 2)
        for key, value in (
            ("num_executions", 1),
            ("warmup_executions", 0),
            ("execute_ms", [1]),
            ("prepare_execute_ms", [0.5, 3]),
            ("synchronization", "none"),
            ("scope", "enqueue"),
            ("device_index", -1),
        ):
            changed = copy.deepcopy(data)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_samples(changed, 3, 2)

    def test_package_checksums_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = (
                "run.sh",
                "bin/executor_runner",
                "model/model.pte",
                "model/model.ptd",
                "lib/libaoti_cuda_shims.so",
                "lib/libextension_cuda.so",
            )
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"original")
            manifest = {
                "version": 1,
                "files": {name: fingerprint(root / name) for name in names},
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verify_package(root), manifest)
            (root / "run.sh").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_package(root)
            (root / "run.sh").write_bytes(b"original")
            manifest["files"]["../escape"] = {}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Invalid package path"):
                verify_package(root)


if __name__ == "__main__":
    unittest.main()
