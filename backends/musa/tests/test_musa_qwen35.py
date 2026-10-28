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
from examples.musa.aoti_qwen35_compare import (
    compare_distributions,
    select_input_token,
    summarize_comparison,
)
from examples.musa.aoti_qwen35_validate import (
    compare_logits,
    runtime_libraries,
    tensor_values,
    VOCAB_SIZE,
)


class TestQwen35LogitValidation(unittest.TestCase):
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
