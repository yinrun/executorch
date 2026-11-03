"""GGUF model mapping and tiled GDN semantics, separate from model acceptance."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from executorch.backends.musa.gguf import dequantize_gguf
from executorch.backends.musa.gguf_packed import GGUF_LINEAR, GgufLinear
from executorch.backends.musa.tests.test_musa_gguf import sample_blocks
from executorch.examples.models.llama.attention import AttentionGatedDeltaNet
from executorch.examples.models.llama.feed_forward import FeedForward
from executorch.examples.models.llama.model_args import ModelArgs
from examples.musa.aoti_qwen35_gguf import (
    GgufGatedDeltaNet,
    GgufMergedFeedForward,
    model_parameters,
    tensor_mapping,
)


def metadata():
    return {
        "general.architecture": "qwen35",
        "qwen35.block_count": 32,
        "qwen35.embedding_length": 2560,
        "qwen35.feed_forward_length": 9216,
        "qwen35.attention.head_count": 16,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "qwen35.rope.dimension_count": 64,
        "qwen35.rope.dimension_sections": [11, 11, 10, 0],
        "qwen35.rope.freq_base": 10000000.0,
        "qwen35.ssm.conv_kernel": 4,
        "qwen35.ssm.state_size": 128,
        "qwen35.ssm.group_count": 16,
        "qwen35.ssm.time_step_rank": 32,
        "qwen35.ssm.inner_size": 4096,
        "qwen35.full_attention_interval": 4,
        "qwen35.context_length": 262144,
        "qwen35.attention.layer_norm_rms_epsilon": 9.999999974752427e-7,
    }


class TestMusaQwen35GgufMetadata(unittest.TestCase):
    def test_merged_ffn_export_and_invalid_projections(self):
        with torch.device("meta"):
            previous = FeedForward(256, 512)
            previous.w1 = GgufLinear(torch.empty((512, 144), dtype=torch.uint8), "q4_k")
            previous.w3 = GgufLinear(torch.empty((512, 144), dtype=torch.uint8), "q4_k")
            previous.w2 = GgufLinear(torch.empty((256, 288), dtype=torch.uint8), "q4_k")
            merged = GgufMergedFeedForward(previous)
            exported = torch.export.export(merged, (torch.empty((1, 1, 256)),))
            calls = [
                node for node in exported.graph.nodes if node.target == GGUF_LINEAR
            ]
            self.assertEqual(len(calls), 2)
            self.assertEqual(merged.gate_up.packed_weight.shape, (1024, 144))
            self.assertIs(merged.w2, previous.w2)
            self.assertIs(merged.act_fn, previous.act_fn)
            with self.assertRaises(ValueError):
                GgufMergedFeedForward(FeedForward(256, 512))
            previous.w3 = GgufLinear(torch.empty((512, 176), dtype=torch.uint8), "q5_k")
            with self.assertRaises(ValueError):
                GgufMergedFeedForward(previous)
            previous.w3 = GgufLinear(torch.empty((511, 144), dtype=torch.uint8), "q4_k")
            with self.assertRaises(ValueError):
                GgufMergedFeedForward(previous)
            previous.w3 = previous.w1
            previous.w1.bias = torch.empty(512)
            with self.assertRaises(ValueError):
                GgufMergedFeedForward(previous)

    def test_full_mapping_and_no_double_norm_offset(self):
        args = model_parameters(metadata(), 4096)
        self.assertFalse(args.rms_norm_add_unit_offset)
        self.assertEqual(args.max_context_len, 4096)
        mapping = tensor_mapping(args)
        self.assertEqual(len(mapping), 425)
        self.assertEqual(len(set(mapping.values())), 425)
        self.assertEqual(mapping["blk.0.ssm_a"], "layers.0.attention.negative_a")
        self.assertEqual(mapping["blk.3.attn_q.weight"], "layers.3.attention.wq")
        self.assertNotIn("blk.3.ssm_a", mapping)
        self.assertEqual(sum(name.endswith("ssm_a") for name in mapping), 24)

    def test_rejects_incompatible_model_and_context(self):
        for name, invalid in (
            ("general.architecture", "qwen3vl"),
            ("qwen35.embedding_length", 1024),
            ("qwen35.ssm.group_count", 32),
            ("qwen35.rope.dimension_sections", [16, 16, 0, 0]),
            ("qwen35.attention.layer_norm_rms_epsilon", 1e-5),
            ("qwen35.attention.layer_norm_rms_epsilon", float("nan")),
        ):
            values = metadata()
            values[name] = invalid
            with self.assertRaises(ValueError):
                model_parameters(values, 4096)
        for context in (0, 1, 4097, 3.5):
            with self.assertRaises(ValueError):
                model_parameters(metadata(), context)


@unittest.skipUnless(
    hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
)
class TestMusaQwen35GgufGpu(unittest.TestCase):
    def test_merged_ffn_preserves_packed_bytes_and_outputs(self):
        with torch.device("meta"):
            previous = FeedForward(256, 4608)
        gate = sample_blocks("q4_k", rows=4608, blocks_per_row=1)
        up = gate[::-1].copy()
        down = sample_blocks("q4_k", rows=256, blocks_per_row=18)
        previous.w1 = GgufLinear(torch.from_numpy(gate).to("musa"), "q4_k")
        previous.w3 = GgufLinear(torch.from_numpy(up).to("musa"), "q4_k")
        previous.w2 = GgufLinear(torch.from_numpy(down).to("musa"), "q4_k")
        merged = GgufMergedFeedForward(previous).eval()
        self.assertTrue(
            torch.equal(merged.gate_up.packed_weight[:4608], previous.w1.packed_weight)
        )
        self.assertTrue(
            torch.equal(merged.gate_up.packed_weight[4608:], previous.w3.packed_weight)
        )
        self.assertNotIn("w1.packed_weight", dict(merged.named_buffers()))
        self.assertNotIn("w3.packed_weight", dict(merged.named_buffers()))
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            for shape in ((1, 1, 256), (2, 256)):
                value = (
                    torch.linspace(-0.01, 0.01, 256, device="musa")
                    .expand(shape)
                    .contiguous()
                )
                actual, expected = merged(value), previous(value)
                self.assertTrue(torch.isfinite(actual).all().item())
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_tiled_gdn_matches_grouped_reference(self):
        torch.manual_seed(1729)
        args = ModelArgs(
            dim=16,
            hidden_dim=32,
            n_heads=2,
            n_kv_heads=2,
            head_dim=8,
            n_layers=1,
            vocab_size=32,
            max_seq_len=8,
            max_context_len=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_conv_kernel_dim=4,
        )
        grouped = AttentionGatedDeltaNet(args, 0, None).to("musa").eval()
        tiled = GgufGatedDeltaNet(args, 0, None).to("musa").eval()
        head_order = torch.tensor([0, 2, 1, 3], device="musa")
        column_order = (
            torch.arange(16, device="musa").reshape(2, 2, 4).transpose(0, 1).reshape(-1)
        )
        with torch.no_grad():
            tiled.dt_bias.copy_(grouped.dt_bias[head_order])
            tiled.negative_a.copy_(-grouped.A_log.exp()[head_order])
            tiled.norm.weight.copy_(grouped.norm.weight)
            tiled.out_proj.weight.copy_(grouped.out_proj.weight[:, column_order])
            tiled.conv1d.weight[:16].copy_(grouped.conv1d.weight[:16])
            tiled.conv1d.weight[16:].copy_(grouped.conv1d.weight[16:][column_order])
            for step in range(3):
                mixed = torch.randn((1, 1, 32), device="musa")
                gate = torch.randn((1, 1, 4, 4), device="musa")
                beta = torch.randn((1, 1, 4), device="musa")
                decay = torch.randn((1, 1, 4), device="musa")
                expected, _ = grouped._forward_projected(mixed, gate, beta, decay)
                tiled_mixed = torch.cat(
                    (mixed[..., :16], mixed[..., 16:][..., column_order]), dim=-1
                )
                actual, _ = tiled._forward_projected(
                    tiled_mixed,
                    gate[:, :, head_order],
                    beta[..., head_order],
                    decay[..., head_order],
                )
                torch.testing.assert_close(
                    actual, expected, atol=1e-6, rtol=1e-6, msg=f"step {step}"
                )
                torch.testing.assert_close(
                    tiled.recurrent_state,
                    grouped.recurrent_state[:, head_order],
                    atol=1e-6,
                    rtol=1e-6,
                )

    def test_large_eager_linear_chunks_preserve_rows(self):
        data = sample_blocks("q6_k", rows=4097, blocks_per_row=1)
        data[:, 208] = 0
        data[:, 209] = 0x24
        packed = torch.from_numpy(data).to("musa")
        value = torch.ones((2, 256), device="musa")
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            actual = GgufLinear(packed, "q6_k")(value)
            dense = dequantize_gguf(packed, "q6_k")
            expected = torch.nn.functional.linear(value, dense)
        self.assertEqual(actual.shape, (2, 4097))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@unittest.skipUnless(
    hasattr(torch, "musa")
    and torch.musa.is_available()
    and os.environ.get("EXECUTORCH_MUSA_GGUF_RUNNER"),
    "requires a native packed GGUF runner",
)
class TestMusaQwen35MergedFfnRunner(unittest.TestCase):
    def test_merged_ffn_matches_separate_native_outputs(self):
        from executorch.backends.musa import MusaBackend, MusaPartitioner
        from executorch.exir import to_edge_transform_and_lower
        from executorch.extension.export_util.utils import save_pte_program
        from examples.musa.aoti_model_acceptance import (
            _serialization_schema,
            _write_tensor,
        )

        class Comparison(torch.nn.Module):
            def __init__(self):
                super().__init__()
                with torch.device("meta"):
                    previous = FeedForward(256, 512)
                gate = sample_blocks("q4_k", rows=512, blocks_per_row=1)
                previous.w1 = GgufLinear(torch.from_numpy(gate).to("musa"), "q4_k")
                previous.w3 = GgufLinear(
                    torch.from_numpy(gate[::-1].copy()).to("musa"), "q4_k"
                )
                previous.w2 = GgufLinear(
                    torch.from_numpy(
                        sample_blocks("q4_k", rows=256, blocks_per_row=2)
                    ).to("musa"),
                    "q4_k",
                )
                self.previous = previous
                self.merged = GgufMergedFeedForward(previous)

            def forward(self, value):
                return self.previous(value), self.merged(value)

        model = Comparison().eval()
        inputs = (torch.linspace(-0.01, 0.01, 256, device="musa").reshape(1, 1, 256),)
        runner = Path(os.environ["EXECUTORCH_MUSA_GGUF_RUNNER"]).resolve()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with torch.no_grad(), _serialization_schema(), torch._inductor.config.patch(
                memory_planning=True, memory_pool="combined"
            ):
                exported = torch.export.export(model, inputs)
                specs = [MusaBackend.generate_method_name_compile_spec("forward")]
                program = to_edge_transform_and_lower(
                    exported, partitioner=[MusaPartitioner(specs)]
                ).to_executorch()
                self.assertEqual(
                    len(program.executorch_program.execution_plan[0].delegates), 1
                )
                pte = Path(save_pte_program(program, "merged_ffn", directory))
            _write_tensor(inputs[0], output / "input.bin")
            process = subprocess.run(
                [
                    str(runner),
                    f"--model_path={pte}",
                    f"--data_path={output}/aoti_musa_blob.ptd",
                    f"--inputs={output}/input.bin",
                    f"--output_file={output}/output",
                    "--print_output=none",
                    "--num_executions=3",
                ],
                env=dict(
                    PATH="/usr/bin:/bin",
                    LD_LIBRARY_PATH=os.environ.get("LD_LIBRARY_PATH", ""),
                ),
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            previous = (output / "output-0.bin").read_bytes()
            merged = (output / "output-1.bin").read_bytes()
            self.assertEqual(len(merged), 256 * 4)
            self.assertEqual(previous, merged)
            self.assertTrue(
                torch.isfinite(torch.frombuffer(bytearray(merged), dtype=torch.float32))
                .all()
                .item()
            )


if __name__ == "__main__":
    unittest.main()
