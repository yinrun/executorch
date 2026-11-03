"""Lossless packed GDN gate merging, including native row-dispatch boundaries."""

import copy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from executorch.backends.musa.gguf_packed import GGUF_LINEAR, GgufLinear
from executorch.backends.musa.tests.test_musa_gguf import sample_blocks
from executorch.examples.models.llama.model_args import ModelArgs
from examples.musa.aoti_qwen35_gguf import GgufGatedDeltaNet, GgufMergedGdnProjection


def packed_projection(rows, columns, encoding="q4_k"):
    data = sample_blocks(encoding, rows=rows, blocks_per_row=columns // 256)
    blocks = data.reshape(-1, 144 if encoding == "q4_k" else 176)
    for offset in (0, 2):
        blocks[:, offset : offset + 2] = np.array([1 / 1024], dtype=np.float16).view(
            np.uint8
        )
    return GgufLinear(torch.from_numpy(data).to("musa"), encoding)


class TestGdnMergeMetadata(unittest.TestCase):
    def test_export_has_one_packed_projection(self):
        with torch.device("meta"):
            layers = [
                GgufLinear(torch.empty((rows, 144), dtype=torch.uint8), "q4_k")
                for rows in (512, 4, 4)
            ]
            merged = GgufMergedGdnProjection(*layers)
            exported = torch.export.export(merged, (torch.empty((1, 1, 256)),))
            self.assertEqual(
                sum(node.target == GGUF_LINEAR for node in exported.graph.nodes), 1
            )
            self.assertEqual(merged.sizes, (512, 4, 4))
            self.assertEqual(merged.projection.packed_weight.shape, (520, 144))
            self.assertEqual(len(exported.state_dict), 1)

    def test_rejects_incompatible_projections(self):
        with torch.device("meta"):
            gate = GgufLinear(torch.empty((512, 144), dtype=torch.uint8), "q4_k")
            beta = GgufLinear(torch.empty((4, 144), dtype=torch.uint8), "q4_k")
            for decay in (
                torch.nn.Linear(256, 4, bias=False),
                GgufLinear(torch.empty((4, 176), dtype=torch.uint8), "q5_k"),
                GgufLinear(torch.empty((4, 288), dtype=torch.uint8), "q4_k"),
                GgufLinear(torch.empty((3, 144), dtype=torch.uint8), "q4_k"),
                GgufLinear(
                    torch.empty((4, 144), dtype=torch.uint8), "q4_k", torch.empty(4)
                ),
            ):
                with self.assertRaises(ValueError):
                    GgufMergedGdnProjection(gate, beta, decay)
        with self.assertRaises(ValueError):
            GgufMergedGdnProjection(
                gate, beta, GgufLinear(torch.empty((4, 144), dtype=torch.uint8), "q4_k")
            )


@unittest.skipUnless(
    hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
)
class TestGdnMergeGpu(unittest.TestCase):
    def test_projection_bytes_and_eager_outputs(self):
        projections = [packed_projection(rows, 256) for rows in (512, 4, 4)]
        merged = GgufMergedGdnProjection(*projections)
        start = 0
        for projection in projections:
            rows = projection.packed_weight.shape[0]
            self.assertTrue(
                torch.equal(
                    merged.projection.packed_weight[start : start + rows],
                    projection.packed_weight,
                )
            )
            start += rows
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            for shape in ((1, 1, 256), (2, 256)):
                value = (
                    torch.linspace(-0.1, 0.1, 256, device="musa")
                    .expand(shape)
                    .contiguous()
                )
                for actual, projection in zip(merged(value), projections):
                    torch.testing.assert_close(
                        actual, projection(value), atol=1e-6, rtol=1e-6
                    )

    def test_forward_preserves_state_updates_and_reset(self):
        torch.manual_seed(1729)
        args = ModelArgs(
            dim=256,
            hidden_dim=512,
            n_heads=2,
            n_kv_heads=2,
            head_dim=128,
            n_layers=1,
            vocab_size=256,
            max_seq_len=8,
            max_context_len=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )
        previous = GgufGatedDeltaNet(args, 0, None).to("musa").eval()
        previous.negative_a.fill_(-0.1)
        previous.in_proj_qkv = packed_projection(1024, 256, "q5_k")
        previous.in_proj_z = packed_projection(512, 256)
        previous.in_proj_b = packed_projection(4, 256)
        previous.in_proj_a = packed_projection(4, 256)
        previous.out_proj = packed_projection(256, 512)
        merged = copy.deepcopy(previous)
        merged.merge_gate_projections()
        for name in ("in_proj_z", "in_proj_b", "in_proj_a"):
            self.assertNotIn(name, dict(merged.named_children()))
        with self.assertRaises(ValueError):
            merged.merge_gate_projections()
        first = None
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            for position in (0, 1, 2, 0):
                value = torch.linspace(-0.01, 0.01, 256, device="musa").reshape(
                    1, 1, 256
                ) * (position + 1)
                options = dict(input_pos=torch.tensor([position], device="musa"))
                actual, _ = merged(value, None, None, **options)
                expected, _ = previous(value, None, None, **options)
                self.assertTrue(torch.isfinite(actual).all().item())
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(
                    merged.conv_state, previous.conv_state, atol=1e-5, rtol=1e-5
                )
                torch.testing.assert_close(
                    merged.recurrent_state,
                    previous.recurrent_state,
                    atol=1e-5,
                    rtol=1e-5,
                )
                if first is None:
                    first = actual.clone()
                elif position == 0:
                    torch.testing.assert_close(actual, first, atol=0, rtol=0)
            with self.assertRaises(ValueError):
                merged(value.expand(2, 1, 256), None, None, **options)


@unittest.skipUnless(
    hasattr(torch, "musa")
    and torch.musa.is_available()
    and os.environ.get("EXECUTORCH_MUSA_GGUF_RUNNER"),
    "requires a native packed GGUF runner",
)
class TestGdnMergeRunner(unittest.TestCase):
    def test_native_projection_outputs_are_byte_exact(self):
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
                self.separate = torch.nn.ModuleList(
                    packed_projection(rows, 256) for rows in (512, 4, 4)
                )
                self.merged = GgufMergedGdnProjection(*self.separate)

            def forward(self, value):
                return tuple(layer(value) for layer in self.separate) + self.merged(
                    value
                )

        model = Comparison().eval()
        value = torch.sin(
            torch.arange(256, dtype=torch.float32, device="musa")
        ).reshape(1, 1, 256)
        runner = Path(os.environ["EXECUTORCH_MUSA_GGUF_RUNNER"]).resolve()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with torch.no_grad(), _serialization_schema(), torch._inductor.config.patch(
                memory_planning=True, memory_pool="combined"
            ):
                exported = torch.export.export(model, (value,))
                specs = [MusaBackend.generate_method_name_compile_spec("forward")]
                program = to_edge_transform_and_lower(
                    exported, partitioner=[MusaPartitioner(specs)]
                ).to_executorch()
                self.assertEqual(
                    len(program.executorch_program.execution_plan[0].delegates), 1
                )
                pte = Path(save_pte_program(program, "gdn_merge", directory))
            _write_tensor(value, output / "input.bin")
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
            for index, rows in enumerate((512, 4, 4)):
                previous = (output / f"output-{index}.bin").read_bytes()
                merged = (output / f"output-{index + 3}.bin").read_bytes()
                self.assertEqual(len(merged), rows * 4)
                self.assertEqual(merged, previous)
                self.assertTrue(
                    np.isfinite(np.frombuffer(merged, dtype=np.float32)).all()
                )


if __name__ == "__main__":
    unittest.main()
