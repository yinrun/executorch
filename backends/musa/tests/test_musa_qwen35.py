"""MUSA coverage for Qwen3.5 recurrent state and export dispatch."""

import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import torch

try:
    import torch_musa
except ImportError:
    MUSA_AVAILABLE = False
else:
    MUSA_AVAILABLE = torch_musa.is_available()

from executorch.backends.musa.musa_backend import _decompose_musa_to_copy
from executorch.examples.models.llama import attention
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.norm import RMSNorm
from executorch.examples.models.llama.rope import Rope
from examples.musa import aoti_qwen35_compare as qwen35_compare
from examples.musa.aoti_qwen35_compare import (
    compare_distributions,
    musa_cache_arguments,
    musa_cache_report,
    native_eos_reached,
    select_fixtures,
    select_input_token,
    summarize_comparison,
    validated_context_length,
)
from examples.musa.aoti_qwen35_validate import (
    compare_logits,
    file_hash,
    read_greedy_token,
    runtime_libraries,
    tensor_values,
    VOCAB_SIZE,
)


class TestQwen35LogitValidation(unittest.TestCase):
    def test_identical_distribution_fast_path_matches_full_metrics(self):
        for values in (
            [0.0],
            [0.0, -0.0, 1.0, 1.0, -1.0],
            [-1000.0, 1000.0, 0.0, 999.0],
            [index / 32 for index in range(-200, 200)],
        ):
            for top_k in (1, 10, 1000):
                with self.subTest(values=values, top_k=top_k):
                    self.assertEqual(
                        compare_distributions(values, values, top_k),
                        compare_distributions(values, list(values), top_k),
                    )

    def test_identical_payloads_still_validate_size_and_finiteness(self):
        for payload in (
            b"",
            b"\0" * 2,
            struct.pack("<H", 0x7FC0) * VOCAB_SIZE,
            struct.pack("<H", 0x7F80) * VOCAB_SIZE,
        ):
            with self.assertRaises(ValueError):
                qwen35_compare.compare_payloads(payload, payload, "bf16")

    def test_payload_identity_does_not_hide_sign_bits_or_one_logit_changes(self):
        original = struct.pack("<H", 0) * VOCAB_SIZE
        for first, different in ((0x8000, 0), (0x3F80, 1)):
            changed = struct.pack("<H", first) + original[2:]
            result = qwen35_compare.compare_payloads(changed, original, "bf16")
            self.assertFalse(result["byte_equal"])
            self.assertEqual(result["different"], different)
        with mock.patch.object(qwen35_compare, "tensor_values", wraps=tensor_values) as decode:
            result = qwen35_compare.compare_payloads(original, original, "bf16")
            self.assertTrue(result["byte_equal"])
            self.assertEqual(decode.call_count, 1)

    def test_exact_comparison_cli_preserves_diagnostics_and_checks_every_session(self):
        exact = {
            "session": 0,
            "fixture": "example",
            "position": 0,
            "native_vs_hf": {
                "byte_equal": True,
                "outside_legacy_tolerance": 0,
                "predicted_token": 1,
                "reference_token": 1,
            },
        }
        different = {
            **exact,
            "session": 1,
            "native_vs_hf": {**exact["native_vs_hf"], "byte_equal": False},
        }
        for strict, steps, passes in (
            (True, [exact], True),
            (True, [exact, different], False),
            (True, [], False),
            (False, [exact, different], True),
        ):
            with (
                self.subTest(strict=strict, steps=steps),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory) / "output"
                arguments = ["compare"]
                for name in ("references", "pte", "ptd", "runner"):
                    arguments.extend([f"--{name}", str(Path(directory) / name)])
                arguments.extend(["--output-dir", str(output)])
                if strict:
                    arguments.append("--require-byte-equal")

                def completed_comparison(args, report):
                    report.update(status="comparison_complete", steps=steps)

                with mock.patch("sys.argv", arguments), mock.patch.object(
                    qwen35_compare, "compare", side_effect=completed_comparison
                ):
                    if passes:
                        qwen35_compare.main()
                    else:
                        with self.assertRaisesRegex(AssertionError, "Exact native/HF"):
                            qwen35_compare.main()
                result = json.loads((output / "result.json").read_text())
                self.assertEqual(result["require_byte_equal"], strict)
                self.assertEqual(result["steps"], steps)
                self.assertEqual(result["runtime_validated"], False)
                self.assertEqual(result["all_logits_byte_equal"], steps == [exact])
                self.assertEqual(
                    result["status"], "comparison_complete" if passes else "failed"
                )
                if strict and different in steps:
                    self.assertIn("session=1 fixture=example position=0", result["error"])

    def test_nonlegacy_context_requires_verified_export_metadata(self):
        self.assertEqual(
            validated_context_length({"context_length": 64}, None, None, None), 64
        )
        for length in (128, 256):
            with self.assertRaisesRegex(ValueError, "export-metadata"):
                validated_context_length({"context_length": length}, None, None, None)
        for length in (True, 0, 1, 64.0, None):
            with self.assertRaisesRegex(ValueError, "context length"):
                validated_context_length({"context_length": length}, None, None, None)

    def test_export_metadata_binds_context_outputs_checkpoint_and_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pte, ptd = root / "qwen35.pte", root / "aoti_musa_blob.ptd"
            pte.write_bytes(b"test program")
            ptd.write_bytes(b"test weights")
            path = root / "result.json"
            manifest = {
                "context_length": 128,
                "checkpoint_hashes": {"/checkpoint/weights": "weights hash"},
            }
            metadata = {
                "status": "exported",
                "model": "Qwen3.5-0.8B-text",
                "dtype": "bf16",
                "context_length": 128,
                "delegates": 1,
                "artifacts": {pte.name: file_hash(pte), ptd.name: file_hash(ptd)},
                "checkpoint_hashes": {"weights": "weights hash"},
                "output_indices": {"logits": 0, "greedy_token": 1},
            }
            path.write_text(json.dumps(metadata))
            self.assertEqual(
                validated_context_length(manifest, pte, ptd, path, True), 128
            )
            for change in (
                {"status": "failed"},
                {"dtype": "fp32"},
                {"context_length": 64},
                {"context_length": 128.0},
                {"delegates": 2},
                {"artifacts": {}},
                {"checkpoint_hashes": {"weights": "different"}},
                {"output_indices": {"logits": 0, "greedy_token": None}},
                {"output_indices": {"logits": 1, "greedy_token": 0}},
            ):
                with self.subTest(change=change):
                    path.write_text(json.dumps({**metadata, **change}))
                    with self.assertRaises(ValueError):
                        validated_context_length(manifest, pte, ptd, path, True)
            path.write_text(json.dumps(metadata))
            ptd.write_bytes(b"changed weights")
            with self.assertRaisesRegex(ValueError, "package hash"):
                validated_context_length(manifest, pte, ptd, path, True)

    def test_native_eos_stops_only_after_prompt_at_matching_reference_end(self):
        fixture = dict(token_ids=[1, 2, 3], prompt_token_count=2, stop_reason="eos")
        self.assertFalse(native_eos_reached(fixture, 0, 9, 9))
        self.assertFalse(native_eos_reached(fixture, 1, 3, 9))
        self.assertTrue(native_eos_reached(fixture, 2, 9, 9))
        with self.assertRaisesRegex(ValueError, "stopping point"):
            native_eos_reached(fixture, 1, 9, 9)
        with self.assertRaisesRegex(ValueError, "did not reach"):
            native_eos_reached(fixture, 2, 3, 9)
        limited = {**fixture, "stop_reason": "context_limit"}
        self.assertFalse(native_eos_reached(limited, 2, 3, 9))
        with self.assertRaisesRegex(ValueError, "stopping point"):
            native_eos_reached(limited, 2, 9, 9)

    def test_greedy_token_requires_exact_int64_and_vocabulary_range(self):
        for token in (0, 31, VOCAB_SIZE - 1):
            self.assertEqual(read_greedy_token(struct.pack("<q", token)), token)
        for payload in (b"", b"\x00" * 4, b"\x00" * 16):
            with self.assertRaises(ValueError):
                read_greedy_token(payload)
        for token in (-1, VOCAB_SIZE, 2**63 - 1):
            with self.assertRaises(ValueError):
                read_greedy_token(struct.pack("<q", token))

    def test_cache_teardown_report_ignores_samples_and_rejects_leaks(self):
        counts = dict(
            cached_bytes=0, cached_blocks=0, live_blocks=0, reuses=9, allocations=2
        )
        final = "MUSA ALLOCATION CACHE " + json.dumps(counts)
        sample = "MUSA ALLOCATION CACHE SAMPLE " + json.dumps(
            {**counts, "cached_bytes": 4096}
        )
        self.assertEqual(musa_cache_report(sample + "\n" + final), counts)
        for log in (sample, final + "\n" + final):
            with self.assertRaises(AssertionError):
                musa_cache_report(log)
        for name in ("cached_bytes", "cached_blocks", "live_blocks"):
            with self.assertRaises(AssertionError):
                musa_cache_report(
                    "MUSA ALLOCATION CACHE " + json.dumps({**counts, name: 1})
                )
        for value in (-1, True, 1.0, None):
            with self.assertRaises(AssertionError):
                musa_cache_report(
                    "MUSA ALLOCATION CACHE " + json.dumps({**counts, "reuses": value})
                )

    def test_native_cache_arguments_are_explicit_and_bounded(self):
        self.assertEqual(musa_cache_arguments(0), [])
        self.assertEqual(musa_cache_arguments(4096), ["--musa_cache_bytes=4096"])
        self.assertEqual(
            musa_cache_arguments(2**64 - 1),
            ["--musa_cache_bytes=18446744073709551615"],
        )
        for value in (-1, 2**64, True, 4096.0, "4096", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                musa_cache_arguments(value)

    def test_fixture_selection_preserves_references_and_manifest_order(self):
        fixtures = [{"name": name} for name in ("trust", "fairness", "context_limit")]
        self.assertIs(select_fixtures(fixtures, None), fixtures)
        selected = select_fixtures(fixtures, ["context_limit", "trust"])
        self.assertEqual(selected, [fixtures[0], fixtures[2]])
        self.assertIs(selected[0], fixtures[0])
        self.assertEqual(len(fixtures), 3)
        for requested in (["missing"], ["trust", "trust"]):
            with self.assertRaises(ValueError):
                select_fixtures(fixtures, requested)
        with self.assertRaises(ValueError):
            select_fixtures([fixtures[0], fixtures[0]], ["trust"])

    def test_bf16_decode_preserves_all_finite_bit_patterns(self):
        words = [value for value in range(65536) if value & 0x7F80 != 0x7F80]
        words = (words * (VOCAB_SIZE // len(words) + 1))[:VOCAB_SIZE]
        payload = struct.pack(f"<{len(words)}H", *words)
        values = tensor_values(payload, "bf16")
        self.assertEqual(
            struct.pack(f"<{len(values)}f", *values),
            b"".join(struct.pack("<I", value << 16) for value in words),
        )
        for word in (0x7F80, 0xFF80, 0x7F81, 0xFFC0):
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                tensor_values(struct.pack("<H", word) + payload[2:], "bf16")

    def test_free_generation_uses_native_feedback_and_rejects_divergence(self):
        fixture = {"name": "test", "token_ids": [1, 2, 3], "prompt_token_count": 2}
        self.assertEqual(select_input_token(fixture, 0, None, True), 1)
        self.assertEqual(select_input_token(fixture, 1, 99, True), 2)
        self.assertEqual(select_input_token(fixture, 2, 3, True), 3)
        self.assertEqual(select_input_token(fixture, 2, 99, False), 3)
        with self.assertRaisesRegex(ValueError, "trajectory diverged"):
            select_input_token(fixture, 2, 99, True)
        for prediction in (None, -1, 248320, True):
            with self.assertRaisesRegex(ValueError, "valid native prediction"):
                select_input_token(fixture, 2, prediction, True)

    def test_comparison_summary_separates_numeric_and_ranking_failures(self):
        steps = []
        for position, phase in enumerate(("prompt", "generation")):
            steps.append(
                {
                    "session": 0,
                    "fixture": "example",
                    "position": position,
                    "phase": phase,
                    "native_vs_hf": {
                        "byte_equal": False,
                        "max_error": 0.5 if position == 0 else 0.001,
                        "mean_error": 0.25 if position == 0 else 0.001,
                        "total_variation": 0.1 if position == 0 else 0.0001,
                        "outside_legacy_tolerance": 1 if position == 0 else 0,
                        "reference_token": 0,
                        "predicted_token": position,
                    },
                    "et_vs_hf": {"byte_equal": True},
                }
            )
        summary = summarize_comparison(steps)
        self.assertEqual(summary["scope"], "teacher_forced_fixed_trajectories")
        self.assertIsNone(summary["reset_byte_equal"])
        fixture = summary["fixtures"][0]
        self.assertEqual(fixture["steps"], 2)
        self.assertEqual(fixture["byte_equal_steps"], 0)
        self.assertEqual(fixture["et_hf_byte_equal_steps"], 2)
        self.assertEqual(fixture["steps_outside_legacy_tolerance"], 1)
        self.assertEqual(fixture["max_error"], 0.5)
        self.assertAlmostEqual(fixture["mean_error"], 0.1255)
        self.assertEqual(fixture["max_total_variation"], 0.1)
        self.assertEqual(fixture["first_difference_position"], 0)
        self.assertEqual(fixture["phases"]["prompt"]["argmax_mismatch_positions"], [])
        self.assertEqual(
            fixture["phases"]["generation"]["argmax_mismatch_positions"], [1]
        )
        steps.extend(
            {**row, "session": 1, "reset_byte_equal": True} for row in steps.copy()
        )
        summary = summarize_comparison(steps)
        self.assertEqual(summary["native_calls"], 4)
        self.assertEqual(summary["reset_checks"], 2)
        self.assertTrue(summary["reset_byte_equal"])
        self.assertEqual(summary["fixtures"][0]["steps"], 2)
        steps[-1]["reset_byte_equal"] = False
        self.assertFalse(summarize_comparison(steps)["reset_byte_equal"])

    def test_comparison_summary_exact_and_empty(self):
        row = {
            "session": 0,
            "fixture": "first",
            "position": 0,
            "phase": "generation",
            "native_vs_hf": {
                "byte_equal": True,
                "max_error": 0.0,
                "mean_error": 0.0,
                "total_variation": 0.0,
                "outside_legacy_tolerance": 0,
                "reference_token": 1,
                "predicted_token": 1,
            },
            "et_vs_hf": {"byte_equal": True},
        }
        summary = summarize_comparison([row, {**row, "fixture": "second"}])
        self.assertEqual(
            [item["name"] for item in summary["fixtures"]], ["first", "second"]
        )
        for fixture in summary["fixtures"]:
            self.assertEqual(fixture["byte_equal_steps"], 1)
            self.assertIsNone(fixture["first_difference_position"])
            self.assertEqual(fixture["phases"]["prompt"]["steps"], 0)
        for steps in ([], [{**row, "session": 1}]):
            with self.assertRaises(ValueError):
                summarize_comparison(steps)

    def test_distribution_comparison_exact_and_shifted_logits(self):
        reference = [-2.0, 0.0, 1.0, 3.0]
        exact = compare_distributions(reference, reference)
        self.assertEqual(exact["different"], 0)
        self.assertEqual(exact["max_error"], 0)
        self.assertEqual(exact["total_variation"], 0)
        self.assertEqual(exact["kl_reference_actual"], 0)
        self.assertEqual(exact["top_k_overlap"], 4)
        shifted = compare_distributions(
            [value + 1000 for value in reference], reference
        )
        self.assertEqual(shifted["max_error"], 1000)
        self.assertEqual(shifted["total_variation"], 0)
        self.assertEqual(shifted["kl_reference_actual"], 0)

    def test_distribution_comparison_rank_changes_and_underflow(self):
        row = compare_distributions([0.0, 1.0, -10000.0], [1.0, 0.0, -10000.0], top_k=1)
        self.assertEqual(row["reference_token"], 0)
        self.assertEqual(row["predicted_token"], 1)
        self.assertEqual(row["top_k_overlap"], 0)
        self.assertAlmostEqual(row["total_variation"], 0.4621171572600098)
        self.assertAlmostEqual(row["kl_reference_actual"], 0.4621171572600098)
        self.assertAlmostEqual(row["mean_error"], 2 / 3)
        self.assertAlmostEqual(row["rms_error"], (2 / 3) ** 0.5)
        self.assertLess(
            row["actual_probability_of_reference_top"], row["reference_top_probability"]
        )
        row = compare_distributions([2.0, 2.0], [2.0, 2.0])
        self.assertEqual(row["predicted_token"], 0)

    def test_distribution_comparison_invalid_inputs(self):
        for actual, reference, top_k in (
            ([], [], 10),
            ([1.0], [], 10),
            ([1.0], [1.0], 0),
            ([1.0], [1.0], 1.5),
            ([float("nan")], [1.0], 1),
            ([1.0], [float("inf")], 1),
        ):
            with self.subTest(
                actual=actual, reference=reference, top_k=top_k
            ), self.assertRaises(ValueError):
                compare_distributions(actual, reference, top_k)

    def test_runtime_library_audit_preserves_spaces(self):
        path = "/tmp/relocated package/lib/libextension_cuda.so"
        self.assertEqual(runtime_libraries(f"123: calling init: {path}\n"), [path])
        for name in (
            "libtorch_cpu.so",
            "libc10.so",
            "libpython3.12.so",
            "libmusa_python.so",
        ):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                runtime_libraries(f"123: calling init: /tmp/relocated package/{name}\n")
        with self.assertRaises(AssertionError):
            runtime_libraries("")

    def test_logits_require_full_finite_vocabulary(self):
        for dtype, word in (("fp32", struct.pack("<f", 1)), ("bf16", b"\x80\x3f")):
            with self.subTest(dtype=dtype):
                payload = word * VOCAB_SIZE
                row = compare_logits(payload, payload, dtype)
                self.assertEqual(row["outside_tolerance"], 0)
                self.assertEqual(row["max_error"], 0)
                self.assertEqual(row["predicted_token"], row["reference_token"])
                with self.assertRaises(ValueError):
                    tensor_values(payload[:-1], dtype)
                nan = (
                    struct.pack("<f", float("nan")) if dtype == "fp32" else b"\xc0\x7f"
                )
                with self.assertRaises(ValueError):
                    tensor_values(nan + payload[len(word) :], dtype)

    def test_token_and_numeric_failures_are_independent(self):
        reference = struct.pack("<f", 1) * VOCAB_SIZE
        changed = struct.pack("<f", 1.0001) + reference[4:]
        row = compare_logits(reference, changed, "fp32")
        self.assertEqual(row["outside_tolerance"], 0)
        changed = reference[:4] + struct.pack("<f", 1.0001) + reference[8:]
        row = compare_logits(changed, reference, "fp32")
        self.assertNotEqual(row["predicted_token"], row["reference_token"])
        self.assertEqual(row["outside_tolerance"], 0)
        changed = struct.pack("<f", 2) + reference[4:]
        row = compare_logits(changed, reference, "fp32")
        self.assertEqual(row["predicted_token"], row["reference_token"])
        self.assertEqual(row["outside_tolerance"], 1)


@unittest.skipUnless(MUSA_AVAILABLE, "MUSA is not available")
class TestMusaQwen35(unittest.TestCase):
    def test_gpu_greedy_wrapper_retains_logits_and_first_tie(self):
        from examples.musa.aoti_qwen35_export import Qwen35GreedyOutput

        class IdentityLogits(torch.nn.Module):
            def forward(self, token_ids, attention_options):
                return token_ids

        wrapper = Qwen35GreedyOutput(IdentityLogits()).eval()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for values, expected in (([-2, -1, -1, -3], 1), ([-0.0, 0.0, -1], 0)):
                logits = torch.tensor(values, dtype=dtype, device="musa").reshape(
                    1, 1, -1
                )
                output, token = wrapper(logits, {})
                self.assertIs(output, logits)
                self.assertEqual(token.device.type, "musa")
                self.assertEqual(token.dtype, torch.int64)
                self.assertEqual(tuple(token.shape), (1,))
                self.assertEqual(token.item(), expected)
        for shape in ((), (0,), (2, 3)):
            with self.subTest(shape=shape), self.assertRaisesRegex(
                ValueError, "one nonempty logit row"
            ):
                wrapper(torch.zeros(shape, device="musa"), {})

    def test_qwen_sdpa_preserves_hf_bf16_rounding(self):
        from examples.musa.aoti_qwen35_export import Qwen35MusaSDPA

        generator = torch.Generator().manual_seed(42)
        query = torch.randn(1, 8, 1, 256, generator=generator).to(
            device="musa", dtype=torch.bfloat16
        )
        key = torch.randn(1, 2, 64, 256, generator=generator).to(
            device="musa", dtype=torch.bfloat16
        )
        value = torch.randn(1, 2, 64, 256, generator=generator).to(
            device="musa", dtype=torch.bfloat16
        )
        attention = Qwen35MusaSDPA(2048, 256, 4, 64)
        position = torch.tensor([43], device="musa")
        boolean_mask = torch.arange(64, device="musa").unsqueeze(0) <= 43
        additive_mask = torch.zeros(1, 64, device="musa", dtype=torch.bfloat16)
        additive_mask.masked_fill_(~boolean_mask, float("-inf"))
        for mask in (boolean_mask, additive_mask):
            scores = query @ key.repeat_interleave(4, dim=1).transpose(-2, -1)
            scores = scores * (256**-0.5)
            scores = scores + additive_mask
            probabilities = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
            expected = (
                (probabilities @ value.repeat_interleave(4, dim=1))
                .transpose(1, 2)
                .reshape(1, 1, 2048)
            )
            actual = attention(position, query, key, value, 1, 1, mask)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.no_grad()
    def test_hf_reference_preserves_fp32_rope(self):
        try:
            from safetensors.torch import save_file
            from transformers.models.qwen3_5.configuration_qwen3_5 import (
                Qwen3_5TextConfig,
            )
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5ForCausalLM,
                Qwen3_5TextRotaryEmbedding,
            )
        except ImportError:
            self.skipTest("Qwen3.5 Transformers reference dependencies are unavailable")
        from examples.musa.aoti_qwen35_reference import load_hf_model

        config = Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            layer_types=["full_attention"],
            tie_word_embeddings=True,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [6, 5, 5],
            },
        )
        source = Qwen3_5ForCausalLM(config).to(dtype=torch.bfloat16)
        weights = {
            name.replace("model.", "model.language_model.", 1): value.contiguous()
            for name, value in source.state_dict().items()
            if name.startswith("model.")
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "config.json").write_text(
                json.dumps({"text_config": config.to_dict()})
            )
            save_file(
                weights,
                str(checkpoint / "model.safetensors-00001-of-00001.safetensors"),
            )
            model = load_hf_model(checkpoint)
        self.assertTrue(
            all(value.dtype == torch.bfloat16 for value in model.parameters())
        )
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        rotary = model.model.rotary_emb
        self.assertEqual(rotary.inv_freq.dtype, torch.float32)
        self.assertEqual(rotary.original_inv_freq.dtype, torch.float32)
        reference = Qwen3_5TextRotaryEmbedding(config, device="musa")
        inputs = torch.zeros(1, 1, 32, device="musa", dtype=torch.bfloat16)
        positions = torch.tensor([[63]], device="musa")
        expected = reference(inputs, positions)
        actual = rotary(inputs, positions)
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target, atol=0, rtol=0)
        rounded = reference.to(dtype=torch.bfloat16)(inputs, positions)
        self.assertTrue(
            any(
                not torch.equal(value, target)
                for value, target in zip(rounded, expected)
            )
        )

    @torch.no_grad()
    def test_low_precision_cast_preserves_memory_format(self):
        value = torch.linspace(-2, 2, 120, device="musa").reshape(2, 3, 4, 5)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for memory_format in (
                torch.preserve_format,
                torch.contiguous_format,
                torch.channels_last,
            ):
                with self.subTest(dtype=dtype, memory_format=memory_format):
                    source = value.transpose(2, 3)
                    expected = torch.ops.aten._to_copy.default(
                        source, dtype=dtype, memory_format=memory_format
                    )
                    actual = _decompose_musa_to_copy(
                        source, dtype=dtype, memory_format=memory_format
                    )
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertNotEqual(actual.data_ptr(), source.data_ptr())
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @torch.no_grad()
    def test_unit_offset_norm_scales_before_low_precision_cast(self):
        torch.manual_seed(19)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                module = RMSNorm(32, add_unit_offset=True).to("musa", dtype=dtype)
                module.weight.copy_(torch.randn_like(module.weight))
                hidden = torch.randn(2, 3, 32, device="musa", dtype=dtype)
                normalized = hidden.float() * torch.rsqrt(
                    hidden.float().square().mean(-1, keepdim=True) + module.eps
                )
                expected = (normalized * (1 + module.weight.float())).to(dtype)
                torch.testing.assert_close(module(hidden), expected, atol=0, rtol=0)

    def make_attention(self, dtype):
        args = ModelArgs(
            dim=32,
            hidden_dim=64,
            n_heads=2,
            n_kv_heads=1,
            n_layers=1,
            max_seq_len=8,
            max_context_len=8,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
        )
        module = attention.AttentionGatedDeltaNet(args, 0, Rope(args))
        module = module.eval().to(device="musa", dtype=dtype)
        module.recurrent_state = module.recurrent_state.float()
        return module

    @torch.no_grad()
    def test_musa_recurrence_bypasses_portable_op_and_resets(self):
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            for sequence in (1, 3):
                with self.subTest(dtype=dtype, sequence=sequence):
                    module = self.make_attention(dtype)
                    hidden = torch.randn(1, sequence, 32, device="musa", dtype=dtype)
                    frequencies = torch.zeros(1, 1, device="musa", dtype=dtype)
                    positions = torch.arange(sequence, device="musa")
                    with mock.patch.object(
                        attention,
                        "_get_gated_delta_rule_op",
                        side_effect=AssertionError(
                            "MUSA must not resolve the portable op"
                        ),
                    ) as portable:
                        first, _ = module(
                            hidden, frequencies, frequencies, input_pos=positions
                        )
                        state = module.recurrent_state.clone()
                        conv_state = module.conv_state.clone()
                        module(
                            hidden,
                            frequencies,
                            frequencies,
                            input_pos=positions + sequence,
                        )
                        self.assertFalse(torch.equal(state, module.recurrent_state))
                        reset, _ = module(
                            hidden, frequencies, frequencies, input_pos=positions
                        )
                        portable.assert_not_called()
                    torch.testing.assert_close(first, reset, atol=0, rtol=0)
                    torch.testing.assert_close(
                        state, module.recurrent_state, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        conv_state, module.conv_state, atol=0, rtol=0
                    )
                    self.assertEqual(module.recurrent_state.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(first).all().item())

    @torch.no_grad()
    def test_musa_export_preserves_both_mutable_states(self):
        module = self.make_attention(torch.bfloat16)
        hidden = torch.randn(1, 1, 32, device="musa", dtype=torch.bfloat16)
        frequencies = torch.zeros(1, 1, device="musa", dtype=torch.bfloat16)
        positions = torch.zeros(1, device="musa", dtype=torch.int64)
        with mock.patch.object(
            attention,
            "_get_gated_delta_rule_op",
            side_effect=AssertionError("MUSA must not export the portable op"),
        ) as portable:
            exported = torch.export.export(
                module,
                (hidden, frequencies, frequencies),
                {"input_pos": positions},
            ).run_decompositions()
            portable.assert_not_called()
        self.assertEqual(
            set(exported.graph_signature.buffers_to_mutate.values()),
            {"conv_state", "recurrent_state"},
        )
        self.assertFalse(
            any(
                "llama.gated_delta_rule" in str(node.target)
                for node in exported.graph.nodes
            )
        )


if __name__ == "__main__":
    unittest.main()
