"""Bounded inference-only Qwen projection fusion and state preservation."""

import copy
import json
from pathlib import Path
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

from executorch.examples.models.llama.attention import AttentionGatedDeltaNet
from executorch.examples.models.llama.feed_forward import FeedForward
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from examples.musa import aoti_qwen35_export as exporter
from examples.musa.aoti_qwen35_fusion import (
    fuse_qwen35_projections,
    Qwen35MusaFusedFeedForward,
)


@unittest.skipUnless(MUSA_AVAILABLE, "MUSA is not available")
class TestMusaQwen35Fusion(unittest.TestCase):
    def make_model(self):
        configuration = json.loads(
            (
                Path(__file__).parents[3]
                / "examples/models/qwen3_5/config/0_8b_config.json"
            ).read_text()
        )
        configuration.update(
            n_layers=2,
            vocab_size=64,
            layer_types=["linear_attention", "full_attention"],
            max_batch_size=1,
            max_seq_len=64,
            max_context_len=64,
            use_kv_cache=True,
        )
        model = construct_transformer(ModelArgs(**configuration))
        model.output.weight = model.tok_embeddings.weight
        model = model.eval().to(device="musa", dtype=torch.bfloat16)
        model.layers[0].attention.recurrent_state = (
            model.layers[0].attention.recurrent_state.float()
        )
        previous = model.layers[1].attention.SDPA
        model.layers[1].attention.SDPA = exporter.Qwen35MusaSDPA(
            previous.dim, previous.head_dim, previous.n_rep, previous.max_context_len
        ).eval()
        return model

    @torch.no_grad()
    def test_fused_model_preserves_weights_outputs_state_and_reset(self):
        torch.manual_seed(41)
        original = self.make_model()
        model = copy.deepcopy(original)
        full_attention = model.layers[1].attention
        tied_weight = model.output.weight
        parameter_count = sum(value.numel() for value in model.parameters())
        fuse_qwen35_projections(model)
        self.assertIs(model.layers[1].attention, full_attention)
        self.assertIs(model.output.weight, tied_weight)
        self.assertIs(model.output.weight, model.tok_embeddings.weight)
        self.assertEqual(
            sum(value.numel() for value in model.parameters()), parameter_count
        )
        self.assertFalse(
            any("in_proj_" in name for name, _ in model.named_parameters())
        )
        fused_attention = model.layers[0].attention.attention
        first_output = None
        with torch.backends.mudnn.flags(allow_tf32=False):
            for position in (0, 1, 2, 0, 1, 2):
                tokens = torch.tensor([[position + 1]], device="musa")
                options = {"input_pos": torch.tensor([position], device="musa")}
                expected = original(tokens, options)
                actual = model(tokens, options)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(
                    fused_attention.recurrent_state,
                    original.layers[0].attention.recurrent_state,
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    fused_attention.conv_state,
                    original.layers[0].attention.conv_state,
                    atol=0,
                    rtol=0,
                )
                if position == 0:
                    if first_output is None:
                        first_output = actual.clone()
                    else:
                        torch.testing.assert_close(actual, first_output, atol=0, rtol=0)
        self.assertEqual(fused_attention.recurrent_state.dtype, torch.float32)
        with self.assertRaises(ValueError):
            fuse_qwen35_projections(model)
        self.assertIs(model.layers[1].attention, full_attention)

    @torch.no_grad()
    def test_fused_export_retains_all_mutated_buffers(self):
        model = fuse_qwen35_projections(self.make_model())
        tokens = torch.ones((1, 1), dtype=torch.int64, device="musa")
        options = {"input_pos": torch.zeros(1, dtype=torch.int64, device="musa")}
        exported = torch.export.export(model, (tokens, options)).run_decompositions({})
        states = set(exported.graph_signature.buffers_to_mutate.values())
        self.assertEqual(len(states), 4)
        self.assertEqual(
            states,
            {
                name
                for name, _ in model.named_buffers()
                if name.endswith(exporter.STATE_SUFFIXES)
            },
        )

    @torch.no_grad()
    def test_fused_feed_forward_preserves_offset_input_and_weight_order(self):
        original = FeedForward(1024, 3584).eval().to("musa", torch.bfloat16)
        fused = Qwen35MusaFusedFeedForward(original)
        torch.testing.assert_close(fused.gate_up.weight[:3584], original.w1.weight)
        torch.testing.assert_close(fused.gate_up.weight[3584:], original.w3.weight)
        self.assertNotEqual(
            fused.gate_up.weight.data_ptr(), original.w1.weight.data_ptr()
        )
        storage = torch.randn(1, 1, 1027, device="musa", dtype=torch.bfloat16)
        hidden = storage[..., 3:]
        before = storage.clone()
        with torch.backends.mudnn.flags(allow_tf32=False):
            torch.testing.assert_close(fused(hidden), original(hidden), atol=0, rtol=0)
        torch.testing.assert_close(storage, before, atol=0, rtol=0)

    @torch.no_grad()
    def test_unsupported_inputs_fail_before_state_reset(self):
        model = fuse_qwen35_projections(self.make_model())
        attention = model.layers[0].attention
        attention.attention.conv_state.fill_(3)
        attention.attention.recurrent_state.fill_(7)
        invalid = [
            torch.zeros(2, 1, 1024, device="musa", dtype=torch.bfloat16),
            torch.zeros(1, 2, 1024, device="musa", dtype=torch.bfloat16),
            torch.zeros(1, 1, 1024, device="musa"),
            torch.zeros(1, 1, 1024, dtype=torch.bfloat16),
            torch.zeros(1, 1, 2048, device="musa", dtype=torch.bfloat16)[..., ::2],
        ]
        position = torch.zeros(1, dtype=torch.int64, device="musa")
        for hidden in invalid:
            with self.assertRaisesRegex(ValueError, "one contiguous MUSA BF16 token"):
                attention(hidden, None, None, input_pos=position)
            with self.assertRaises(ValueError):
                model.layers[0].feed_forward(hidden)
        self.assertTrue((attention.attention.conv_state == 3).all().item())
        self.assertTrue((attention.attention.recurrent_state == 7).all().item())
        hidden = torch.zeros(1, 1, 1024, device="musa", dtype=torch.bfloat16)
        attention.train()
        with self.assertRaises(ValueError):
            attention(hidden, None, None, input_pos=position)
        attention.eval().to(torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "FP32 recurrent"):
            attention(hidden, None, None, input_pos=position)

    def test_invalid_linears_activation_and_layout_are_rejected(self):
        original = FeedForward(1024, 3584).eval().to("musa", torch.bfloat16)
        cases = [
            torch.nn.Linear(
                1024, 3584, bias=True, device="musa", dtype=torch.bfloat16
            ).eval(),
            torch.nn.Linear(1024, 3584, bias=False, device="musa").eval(),
            torch.nn.Linear(1024, 3584, bias=False, dtype=torch.bfloat16).eval(),
            torch.nn.Linear(
                1024, 16, bias=False, device="musa", dtype=torch.bfloat16
            ).eval(),
        ]
        noncontiguous = copy.deepcopy(original.w1)
        noncontiguous.weight = torch.nn.Parameter(
            noncontiguous.weight.T.contiguous().T
        )
        cases.append(noncontiguous)
        for linear in cases:
            with (
                mock.patch.object(original, "w1", linear),
                self.assertRaises(ValueError),
            ):
                Qwen35MusaFusedFeedForward(original)
        with mock.patch.object(original, "act_fn", torch.nn.functional.relu):
            with self.assertRaises(ValueError):
                Qwen35MusaFusedFeedForward(original)

    def test_preflight_rejects_without_partially_replacing_modules(self):
        model = self.make_model()
        first_ffn = model.layers[0].feed_forward
        first_attention = model.layers[0].attention
        with mock.patch.object(
            model.layers[1].feed_forward.w3,
            "bias",
            torch.nn.Parameter(torch.zeros(3584, device="musa", dtype=torch.bfloat16)),
        ), self.assertRaises(ValueError):
            fuse_qwen35_projections(model)
        self.assertIs(model.layers[0].feed_forward, first_ffn)
        self.assertIs(model.layers[0].attention, first_attention)
        self.assertIsInstance(first_attention, AttentionGatedDeltaNet)
        self.assertIsInstance(first_attention.in_proj_qkv, torch.nn.Linear)
        for capacity in (32, 512):
            with mock.patch.object(model, "max_context_len", capacity):
                with self.assertRaisesRegex(ValueError, "contexts 64, 128 or 256"):
                    fuse_qwen35_projections(model)
        with mock.patch.object(model.params, "max_batch_size", 2):
            with self.assertRaises(ValueError):
                fuse_qwen35_projections(model)
        model.train()
        with self.assertRaisesRegex(ValueError, "inference-only"):
            fuse_qwen35_projections(model)

    def test_exporter_fusion_is_opt_in_and_rejects_unsupported_modes(self):
        for extra, valid in (
            ([], True),
            (["--dtype", "bf16", "--fuse-projections"], True),
            (["--fuse-projections"], False),
            (
                ["--dtype", "bf16", "--context-length", "512", "--fuse-projections"],
                False,
            ),
        ):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "export"
                argv = [
                    "export", "--checkpoint-dir", directory,
                    "--output-dir", str(output), *extra,
                ]
                with (
                    mock.patch("sys.argv", argv),
                    mock.patch.object(exporter, "export_model") as export,
                ):
                    if valid:
                        exporter.main()
                        self.assertEqual(
                            export.call_args.args[0].fuse_projections, bool(extra)
                        )
                    else:
                        with self.assertRaises(SystemExit):
                            exporter.main()
                        export.assert_not_called()
                        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
