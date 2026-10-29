"""Check fixed-grid vision equations and exact mixed-precision storage."""

import unittest

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

from examples.musa.aoti_qwen35_gguf_vision import (
    GgufVision,
    VisionConfig,
    vision_geometry,
    vision_tensor_mapping,
)


class TestGgufVisionGeometry(unittest.TestCase):
    def test_complete_mapping(self):
        mapping = vision_tensor_mapping()
        self.assertEqual(len(mapping), 298)
        self.assertEqual(len(set(mapping.values())), 298)
        with torch.device("meta"):
            model = GgufVision(VisionConfig(), ((1, 2, 2),))
        self.assertEqual(set(mapping.values()), set(dict(model.named_parameters())))
        counts = {}
        for parameter in model.parameters():
            counts[parameter.dtype] = counts.get(parameter.dtype, 0) + 1
        self.assertEqual(counts, {torch.float32: 200, torch.bfloat16: 98})

    def test_geometry_order_and_weights(self):
        indices, weights, coordinates = vision_geometry(VisionConfig(), ((1, 4, 6),))
        self.assertEqual(indices.shape, (4, 24))
        self.assertTrue(torch.equal(weights.sum(0), torch.ones(24)))
        self.assertEqual(
            coordinates[:8].tolist(),
            [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]],
        )
        self.assertEqual(indices.min().item(), 0)
        self.assertEqual(indices.max().item(), 2303)

    def test_reject_invalid_grids(self):
        for grids in ((), ((2, 4, 4),), ((1, 3, 4),), ((1, 0, 4),), ((1, 4.0, 4),)):
            with self.assertRaises(ValueError):
                vision_geometry(VisionConfig(), grids)


@unittest.skipUnless(
    hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
)
class TestGgufVisionGpu(unittest.TestCase):
    def make_models(self, grids):
        torch.manual_seed(1731)
        config = VisionConfig(
            width=32,
            intermediate=64,
            depth=2,
            heads=4,
            patch_size=2,
            position_side=8,
            output_width=16,
        )
        actual = GgufVision(config, grids).eval().to("musa")
        hf_config = Qwen3_5VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            depth=2,
            num_heads=4,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
            num_position_embeddings=64,
            out_hidden_size=16,
            hidden_act="gelu_pytorch_tanh",
        )
        hf_config._attn_implementation = "eager"
        reference = hf.Qwen3_5VisionModel(hf_config).eval().to("musa")
        with torch.no_grad():
            for name, value in actual.named_parameters():
                value.copy_(torch.randn_like(value.float()).mul(0.05).to(value.dtype))
                if "norm" in name and name.endswith("weight"):
                    value.add_(1)
            state = {
                "patch_embed.proj.weight": torch.stack(
                    (actual.patch_embed.weight_first, actual.patch_embed.weight_second),
                    dim=2,
                ),
                "patch_embed.proj.bias": actual.patch_embed.bias,
                "pos_embed.weight": actual.position_weight,
                "merger.norm.weight": actual.merger_norm.weight,
                "merger.norm.bias": actual.merger_norm.bias,
                "merger.linear_fc1.weight": actual.merger_fc1.weight.float(),
                "merger.linear_fc1.bias": actual.merger_fc1.bias,
                "merger.linear_fc2.weight": actual.merger_fc2.weight.float(),
                "merger.linear_fc2.bias": actual.merger_fc2.bias,
            }
            for index, block in enumerate(actual.blocks):
                for source, target in (
                    ("norm1", "norm1"),
                    ("norm2", "norm2"),
                    ("attn.qkv", "attn.qkv"),
                    ("attn.proj", "attn.proj"),
                    ("fc1", "mlp.linear_fc1"),
                    ("fc2", "mlp.linear_fc2"),
                ):
                    for suffix, value in block.get_submodule(source).named_parameters():
                        state[f"blocks.{index}.{target}.{suffix}"] = value.float()
            reference.load_state_dict(state, strict=True)
        return actual, reference

    def test_single_and_multiple_images_against_hf(self):
        for grids in (((1, 4, 6),), ((1, 4, 6), (1, 2, 4))):
            actual, reference = self.make_models(grids)
            patches = torch.randn(
                sum(height * width for _, height, width in grids), 24, device="musa"
            )
            grid_tensor = torch.tensor(grids, device="musa")
            with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
                expected = reference(patches, grid_tensor).pooler_output
                result = actual(patches)
            self.assertEqual(result.shape, (patches.shape[0] // 4, 16))
            self.assertEqual(result.device.type, "musa")
            torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_export_and_input_guard(self):
        model, _ = self.make_models(((1, 4, 6),))
        patches = torch.randn(24, 24, device="musa")
        with torch.no_grad():
            exported = torch.export.export(model, (patches,), strict=True)
            torch.testing.assert_close(
                exported.module()(patches), model(patches), atol=0, rtol=0
            )
            with self.assertRaises(ValueError):
                model(patches.bfloat16())
            with self.assertRaises(ValueError):
                model(patches[:4])

    def test_patch_embedding_preserves_conv3d_accumulation(self):
        model, reference = self.make_models(((1, 4, 6),))
        patches = torch.randn(24, 24, device="musa")
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            torch.testing.assert_close(
                model.patch_embed(patches),
                reference.patch_embed(patches),
                atol=0,
                rtol=0,
            )


if __name__ == "__main__":
    unittest.main()
