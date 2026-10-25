import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "examples/musa/acceptance_artifacts.py"
SPEC = importlib.util.spec_from_file_location("musa_artifacts", SCRIPT)
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)


class MusaArtifactsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name, payload in (
            ("input.bin", bytes(range(8)) + bytes(8)),
            ("expected.bin", bytes(range(8))),
            ("output.bin", bytes(range(8))),
            ("output-repeat.bin", bytes(range(8))),
        ):
            (self.root / name).write_bytes(payload)
        self.report = {
            "artifact_version": 1,
            "dtype": "bf16",
            "input_dtype": "bf16",
            "runner": "/runner",
            "runner_repeats": 2,
            "artifact_cases": [
                {
                    "input_shape": [1, 4],
                    "expected_shape": [1, 4],
                    "input": artifacts.record_artifact(self.root / "input.bin"),
                    "expected": artifacts.record_artifact(self.root / "expected.bin"),
                    "outputs": [
                        artifacts.record_artifact(self.root / "output.bin"),
                        artifacts.record_artifact(self.root / "output-repeat.bin"),
                    ],
                }
            ],
        }

    def test_valid_padded_case_and_export_only(self):
        artifacts.verify_artifacts(self.root, self.report)
        report = copy.deepcopy(self.report)
        report["runner"] = None
        report["artifact_cases"][0]["outputs"] = []
        artifacts.verify_artifacts(self.root, report)

    def test_snapshot_bytes_are_verified_before_recording(self):
        with self.assertRaisesRegex(ValueError, "tensor snapshot"):
            artifacts.record_artifact(self.root / "expected.bin", bytes(8))

    def test_same_size_tail_corruption_is_rejected(self):
        path = self.root / "expected.bin"
        path.write_bytes(path.read_bytes()[:-2] + bytes(2))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            artifacts.verify_artifacts(self.root, self.report)

    def test_truncated_and_missing_files_are_rejected(self):
        path = self.root / "output.bin"
        path.write_bytes(bytes(2))
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            artifacts.verify_artifacts(self.root, self.report)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            artifacts.verify_artifacts(self.root, self.report)

    def test_nonzero_padding_and_shape_mismatch_are_rejected(self):
        path = self.root / "input.bin"
        path.write_bytes(bytes(range(16)))
        case = self.report["artifact_cases"][0]
        case["input"] = artifacts.record_artifact(path)
        with self.assertRaisesRegex(ValueError, "input padding"):
            artifacts.verify_artifacts(self.root, self.report)
        path.write_bytes(bytes(16))
        case["input"] = artifacts.record_artifact(path)
        case["expected_shape"] = [1, 5]
        with self.assertRaisesRegex(ValueError, "shape/size mismatch"):
            artifacts.verify_artifacts(self.root, self.report)

    def test_old_or_incomplete_reports_are_rejected(self):
        for change in ("legacy", "empty", "repeat", "shapes", "batches", "spatial"):
            with self.subTest(change=change):
                report = copy.deepcopy(self.report)
                if change == "legacy":
                    del report["artifact_version"]
                elif change == "empty":
                    report["artifact_cases"] = []
                elif change == "repeat":
                    report["artifact_cases"][0]["outputs"].pop()
                elif change == "shapes":
                    report["dynamic_shape_results"] = [{"shape": [2, 4]}]
                elif change == "batches":
                    report["dynamic_batch_results"] = [{"batch_size": 2}]
                else:
                    report["dynamic_spatial_results"] = [{"shape": [2, 4]}]
                with self.assertRaises(ValueError):
                    artifacts.verify_artifacts(self.root, report)

    def test_outside_paths_and_duplicates_are_rejected(self):
        for filename in ("../expected.bin", "/tmp/expected.bin", "input.bin"):
            with self.subTest(filename=filename):
                report = copy.deepcopy(self.report)
                report["artifact_cases"][0]["expected"]["file"] = filename
                with self.assertRaisesRegex(ValueError, "artifact path"):
                    artifacts.verify_artifacts(self.root, report)

    def test_cli_requires_no_framework_and_rejects_corruption(self):
        path = self.root / "result.json"
        path.write_text(json.dumps(self.report))
        command = [sys.executable, "-S", str(SCRIPT), str(path)]
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
        (self.root / "expected.bin").write_bytes(bytes(8))
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
