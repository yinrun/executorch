import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from examples.musa.aoti_qwen35_benchmark_report import (
    main,
    summarize_generation,
    summarize_vision,
)


class TestQwen35BenchmarkReport(unittest.TestCase):
    def generation(self):
        return {
            "version": 1,
            "status": "completed",
            "model": "Qwen3.5-4B-GGUF",
            "context_capacity": 4096,
            "prefill_mode": "serial_token",
            "controller": "native_cpp",
            "scope": "synchronized_prepare_execute_output_validation",
            "vision_encoding_included": False,
            "logits_dump_enabled": False,
            "warmup_runs": 1,
            "measured_runs": 3,
            "maximum_generated_tokens": 160,
            "runs": [
                {
                    "run": index,
                    "warmup": index == 0,
                    "prompt_tokens": 4,
                    "decode_steps": 2,
                    "steps": 6,
                    "prefill_ms": duration,
                    "decode_ms": duration / 2,
                    "loop_ms": duration * 2,
                    "stop_reason": "eos",
                    "generated_token_ids": [10, 20, 248046],
                }
                for index, duration in enumerate((10000.0, 100.0, 200.0, 400.0))
            ],
        }

    def vision(self):
        return {
            "version": 1,
            "scope": "host_synchronized_method",
            "synchronization": "musaDeviceSynchronize",
            "device_index": 0,
            "warmup_executions": 1,
            "num_executions": 3,
            "execute_ms": [10.0, 30.0, 20.0],
            "prepare_execute_ms": [11.0, 32.0, 21.0],
        }

    def test_rates_exclude_warmup_and_first_generated_token_from_decode(self):
        report = summarize_generation(self.generation())
        self.assertEqual(report["decode_steps"], 2)
        for name in ("serial_prompt_tokens_per_second", "decode_tokens_per_second"):
            self.assertEqual(report[name]["samples"], [40.0, 20.0, 10.0])
            self.assertEqual(report[name]["median"], 20.0)
            self.assertEqual(report[name]["minimum"], 10.0)
        self.assertFalse(report["batched_prefill"])
        self.assertFalse(report["independent_accuracy_validated"])
        self.assertFalse(report["pure_gpu_time"])

    def test_rejects_generation_diagnostic_or_wrong_scope(self):
        for field, value in (
            ("logits_dump_enabled", True),
            ("warmup_runs", 0),
            ("measured_runs", 2),
            ("prefill_mode", "batch"),
            ("status", "running"),
            ("vision_encoding_included", True),
            ("warmup_runs", True),
            ("version", True),
        ):
            with self.subTest(field=field, value=value):
                data = self.generation()
                data[field] = value
                with self.assertRaises(ValueError):
                    summarize_generation(data)

    def test_rejects_invalid_timings_and_counts(self):
        for field, value in (
            ("decode_ms", 0),
            ("prefill_ms", float("nan")),
            ("loop_ms", float("inf")),
            ("decode_ms", True),
            ("loop_ms", 1),
            ("steps", 7),
            ("decode_steps", 3),
            ("run", True),
            ("warmup", True),
            ("stop_reason", "context_limit"),
            ("stop_reason", "max_new_tokens"),
        ):
            with self.subTest(field=field, value=value):
                data = self.generation()
                data["runs"][1][field] = value
                with self.assertRaises(ValueError):
                    summarize_generation(data)

    def test_rejects_changed_greedy_ids_even_in_warmup(self):
        for index in (0, 2):
            data = self.generation()
            data["runs"][index]["generated_token_ids"][0] += 1
            with self.assertRaisesRegex(ValueError, "workload changed"):
                summarize_generation(data)

    def test_valid_length_stops_are_reported_without_eos_claim(self):
        for reason in ("context_limit", "max_new_tokens"):
            with self.subTest(reason=reason):
                data = self.generation()
                data["maximum_generated_tokens"] = 3
                for run in data["runs"]:
                    run["stop_reason"] = reason
                    run["generated_token_ids"] = [10, 20, 30]
                    if reason == "context_limit":
                        run["prompt_tokens"] = 4094
                        run["steps"] = 4096
                report = summarize_generation(data)
                self.assertEqual(report["stop_reason"], reason)
                self.assertFalse(report["independent_accuracy_validated"])

    def test_rejects_missing_run_and_invalid_token(self):
        data = self.generation()
        data["runs"].pop()
        with self.assertRaises(ValueError):
            summarize_generation(data)
        for value in (-1, 248320, True):
            data = self.generation()
            data["runs"][0]["generated_token_ids"][0] = value
            with self.assertRaises(ValueError):
                summarize_generation(data)

    def test_vision_summary(self):
        report = summarize_vision(self.vision())
        self.assertEqual(report["execute_ms"]["median"], 20.0)
        self.assertEqual(report["prepare_execute_ms"]["median"], 21.0)
        self.assertFalse(report["includes_loading_preprocessing"])

    def test_vision_rejects_incomplete_or_invalid_samples(self):
        for field, value in (
            ("warmup_executions", 0),
            ("num_executions", 2),
            ("execute_ms", [10.0]),
            ("execute_ms", [1, 2, float("inf")]),
            ("prepare_execute_ms", [1, 2, 3]),
            ("synchronization", "none"),
            ("device_index", True),
        ):
            with self.subTest(field=field):
                data = self.vision()
                data[field] = value
                with self.assertRaises(ValueError):
                    summarize_vision(data)

    def test_cli_preserves_sources_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "native.json"
            output = root / "summary.json"
            original = json.dumps(self.generation()).encode()
            source.write_bytes(original)
            arguments = [
                "benchmark_report",
                "--generation",
                str(source),
                "--output",
                str(output),
            ]
            with mock.patch("sys.argv", arguments):
                main()
                result = output.read_bytes()
                with self.assertRaises(FileExistsError):
                    main()
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(output.read_bytes(), result)
            self.assertFalse(json.loads(result)["speedup_claim"])


if __name__ == "__main__":
    unittest.main()
