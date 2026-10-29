"""Validate explicit GGUF adapters against Transformers GPU computations."""

import unittest

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from executorch.backends.musa.tests.test_musa_gguf import sample_blocks
from examples.musa.aoti_qwen35_gguf_reference import (
    DirectGgufDeltaNet,
    DirectGgufRMSNorm,
    PermutedProjection,
    ReferenceLinear,
    cache_reference_linears,
    head_permutation,
)


@unittest.skipUnless(
    hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
)
class TestMusaGgufHfReference(unittest.TestCase):
    def test_direct_norm_preserves_hf_equation(self):
        original = hf.Qwen3_5RMSNorm(16).to("musa")
        direct = DirectGgufRMSNorm(16).to("musa")
        with torch.no_grad():
            original.weight.copy_(torch.randn(16, device="musa"))
            direct.weight.copy_(1 + original.weight)
            value = torch.randn((2, 3, 16), device="musa")
            torch.testing.assert_close(direct(value), original(value), atol=0, rtol=0)

    def test_head_permutations_are_inverse_and_projection_correct(self):
        for key_heads, value_heads, width in ((16, 32, 128), (2, 6, 4)):
            forward = head_permutation(
                key_heads, value_heads, width, tiled_to_grouped=False
            )
            backward = head_permutation(
                key_heads, value_heads, width, tiled_to_grouped=True
            )
            original = torch.arange(value_heads * width, device="musa")
            self.assertTrue(torch.equal(original[forward][backward], original))
            identity = torch.nn.Identity()
            before = PermutedProjection(identity, forward, input_order=True)
            after = PermutedProjection(identity, backward)
            self.assertTrue(torch.equal(after(before(original)), original))
        with self.assertRaises(ValueError):
            head_permutation(3, 4, 8, tiled_to_grouped=True)

    def test_direct_decay_matches_hf_chunk_and_recurrent_paths(self):
        torch.manual_seed(1729)
        config = Qwen3_5TextConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=32,
            layer_types=["linear_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_conv_kernel_dim=4,
        )
        original = hf.Qwen3_5GatedDeltaNet(config, 0).to("musa").eval()
        original.causal_conv1d_fn = None
        original.causal_conv1d_update = hf.torch_causal_conv1d_update
        original.chunk_gated_delta_rule = hf.torch_chunk_gated_delta_rule
        original.recurrent_gated_delta_rule = hf.torch_recurrent_gated_delta_rule
        direct = DirectGgufDeltaNet(config, 0).to("musa").eval()
        state = {
            name: value
            for name, value in original.state_dict().items()
            if name != "A_log"
        }
        missing, unexpected = direct.load_state_dict(state, strict=False)
        self.assertEqual(missing, ["negative_a"])
        self.assertFalse(unexpected)
        original_cache, direct_cache = hf.Qwen3_5DynamicCache(
            config
        ), hf.Qwen3_5DynamicCache(config)
        with torch.no_grad():
            direct.negative_a.copy_(-original.A_log.exp())
            for position in range(3):
                value = torch.randn((1, 1, 16), device="musa")
                offset = torch.tensor([position], device="musa")
                expected = original(value, original_cache, offset)
                actual = direct(value, direct_cache, offset)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(
                    direct_cache.recurrent_states[0],
                    original_cache.recurrent_states[0],
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    direct_cache.conv_states[0],
                    original_cache.conv_states[0],
                    atol=0,
                    rtol=0,
                )

    def test_reference_cache_preserves_outputs_and_budget(self):
        for quant_type in ("q4_k", "q5_k", "q6_k"):
            packed = torch.from_numpy(
                sample_blocks(quant_type, rows=3, blocks_per_row=1)
            ).to("musa")
            layer = ReferenceLinear(packed, quant_type)
            value = torch.randn((2, 256), device="musa")
            with torch.no_grad():
                expected = layer(value)
                rejected = cache_reference_linears(layer, layer.decoded_nbytes() - 1)
                self.assertEqual(rejected["cached_linears"], 0)
                result = cache_reference_linears(layer, layer.decoded_nbytes())
                self.assertEqual(result["cached_linears"], 1)
                self.assertEqual(result["decoded_bytes"], 3 * 256 * 4)
                torch.testing.assert_close(layer(value), expected, atol=0, rtol=0)
                with self.assertRaises(ValueError):
                    cache_reference_linears(layer, 100000)
        with self.assertRaises(ValueError):
            cache_reference_linears(torch.nn.Identity(), -1)


if __name__ == "__main__":
    unittest.main()
