"""MUSA coverage for Qwen3.5 recurrent state and export dispatch."""

import struct
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
from examples.musa.aoti_qwen35_validate import (
    compare_logits,
    runtime_libraries,
    tensor_values,
    VOCAB_SIZE,
)


class TestQwen35LogitValidation(unittest.TestCase):
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
