import copy
import json
import tempfile
import unittest
from pathlib import Path

from examples.musa.benchmark_aoti import (
    fingerprint,
    summarize,
    validate_samples,
    verify_package,
)


class TestMusaBenchmark(unittest.TestCase):
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
