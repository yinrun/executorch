"""Validate image positions, interleaved RoPE and token/embedding selection."""

from types import SimpleNamespace
import unittest

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from examples.musa.aoti_qwen35_export import Qwen35MusaSDPA, STATE_SUFFIXES
from examples.musa.aoti_qwen35_gguf_multimodal import (
    GgufMultimodalStep,
    InterleavedMRope,
    image_token_positions,
)


class HfPositionReference:
    get_rope_index = hf.Qwen3_5Model.get_rope_index
    get_vision_position_ids = hf.Qwen3_5Model.get_vision_position_ids

    def __init__(self):
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2)
        )


class TestImagePositions(unittest.TestCase):
    def test_text_single_and_multiple_images(self):
        image = 248056
        cases = (
            ([11, 12, 13], ()),
            ([11, 12] + [image] * 6 + [13], ((1, 4, 6),)),
            (
                [11, 12] + [image] * 6 + [13, 14] + [image] * 2 + [15],
                ((1, 4, 6), (1, 2, 4)),
            ),
        )
        reference = HfPositionReference()
        for tokens, grids in cases:
            actual = image_token_positions(tokens, grids)
            token_tensor = torch.tensor([tokens])
            expected, delta = reference.get_rope_index(
                token_tensor,
                token_tensor.eq(image).int(),
                image_grid_thw=torch.tensor(grids).reshape(-1, 3),
            )
            self.assertTrue(torch.equal(actual["position_ids"], expected[:, 0]))
            self.assertEqual(actual["rope_delta"], delta.item())
            self.assertEqual(actual["next_position"], expected.max().item() + 1)
            rows = [row for row in actual["image_rows"] if row >= 0]
            self.assertEqual(rows, list(range(tokens.count(image))))

    def test_rejects_malformed_inputs(self):
        for tokens, grids in (
            ([], ()),
            ([True], ()),
            ([248320], ()),
            ([1] * 4097, ()),
            ([248056], ()),
            ([248056], ((1, 4, 4),)),
            ([1], ((1, 2, 2),)),
            ([248056] * 2, ((1, 2, 2),)),
            ([248056], ((2, 2, 2),)),
            ([248056], ((1, 3, 2),)),
        ):
            with self.assertRaises(ValueError):
                image_token_positions(tokens, grids)


@unittest.skipUnless(
    hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
)
class TestMultimodalStepGpu(unittest.TestCase):
    def test_interleaved_rope_against_hf(self):
        config = Qwen3_5TextConfig(
            hidden_size=256,
            head_dim=256,
            num_attention_heads=1,
            num_key_value_heads=1,
            rope_parameters=dict(
                rope_type="default",
                rope_theta=10000000.0,
                partial_rotary_factor=0.25,
                mrope_section=[11, 11, 10],
                mrope_interleaved=True,
            ),
        )
        reference = hf.Qwen3_5TextRotaryEmbedding(config, device="musa")
        actual = InterleavedMRope("musa")
        positions = torch.tensor(
            [[0, 17, 4095], [0, 25, 4093], [0, 36, 4094]], device="musa"
        )
        hidden = torch.zeros(1, 3, 256, device="musa")
        expected = reference(hidden, positions[:, None, :])
        for result, wanted in zip(actual(positions), expected):
            torch.testing.assert_close(result, wanted[0], atol=0, rtol=0)
        exported = torch.export.export(actual, (positions,))
        for result, wanted in zip(exported.module()(positions), expected):
            torch.testing.assert_close(result, wanted[0], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            actual(positions.float())

    def test_single_step_text_and_image_parity(self):
        torch.manual_seed(1732)
        args = ModelArgs(
            dim=32,
            hidden_dim=64,
            n_heads=1,
            n_kv_heads=1,
            head_dim=256,
            n_layers=1,
            vocab_size=32,
            max_seq_len=8,
            max_context_len=8,
            use_kv_cache=True,
            use_hf_rope=True,
            rope_theta=10000000.0,
            partial_rotary_factor=0.25,
        )
        model = construct_transformer(args).to("musa").eval()
        original_sdpa = model.layers[0].attention.SDPA
        model.layers[0].attention.SDPA = Qwen35MusaSDPA(
            original_sdpa.dim,
            original_sdpa.head_dim,
            original_sdpa.n_rep,
            original_sdpa.max_context_len,
        )
        step = GgufMultimodalStep(model).eval()
        token = torch.tensor([[5]], device="musa")
        position = torch.tensor([0], device="musa")
        rope_positions = torch.zeros(3, 1, dtype=torch.int64, device="musa")
        embedding = torch.randn(1, 1, 32, device="musa")
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            for is_image in (False, True):
                for name, value in model.named_buffers():
                    if name.endswith(STATE_SUFFIXES):
                        value.zero_()
                expected = (
                    model(h=embedding, attn_options={"input_pos": position})
                    if is_image
                    else model(token, {"input_pos": position})
                )
                for name, value in model.named_buffers():
                    if name.endswith(STATE_SUFFIXES):
                        value.zero_()
                flag = torch.tensor([is_image], device="musa")
                actual, greedy = step(token, position, rope_positions, embedding, flag)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                self.assertEqual(greedy.item(), expected.argmax().item())
                exported = torch.export.export(
                    step, (token, position, rope_positions, embedding, flag)
                )
                functionalized = exported.run_decompositions({})
                self.assertEqual(
                    len(functionalized.graph_signature.buffers_to_mutate), 2
                )


if __name__ == "__main__":
    unittest.main()
