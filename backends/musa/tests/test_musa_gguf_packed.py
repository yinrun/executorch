"""Packed GGUF contracts and native export without dense weight constants."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from executorch.backends.musa import MusaBackend, MusaPartitioner
from executorch.backends.musa.gguf import BLOCK_BYTES, GGUF_DECODE, dequantize_gguf
from executorch.backends.musa.gguf_packed import (
    GGUF_EMBEDDING,
    GGUF_LINEAR,
    GgufEmbedding,
    GgufLinear,
    QUANT_IDS,
)
from executorch.backends.musa.tests.test_musa_gguf import sample_blocks
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program
from examples.musa.aoti_model_acceptance import _serialization_schema, _write_tensor
from examples.musa.aoti_qwen35_validate import runtime_libraries


class TestMusaGgufPackedMetadata(unittest.TestCase):
    def test_export_keeps_packed_operations_opaque(self):
        for quant_type, block_bytes in BLOCK_BYTES.items():
            packed = torch.empty(
                (3, 10 * block_bytes), dtype=torch.uint8, device="meta"
            )
            for model, value, operator in (
                (
                    GgufLinear(packed, quant_type),
                    torch.empty((2, 2560), device="meta"),
                    GGUF_LINEAR,
                ),
                (
                    GgufEmbedding(packed, quant_type),
                    torch.empty((2, 1), dtype=torch.int64, device="meta"),
                    GGUF_EMBEDDING,
                ),
            ):
                exported = torch.export.export(model, (value,))
                targets = [node.target for node in exported.graph.nodes]
                self.assertIn(operator, targets)
                self.assertNotIn(GGUF_DECODE, targets)
                self.assertNotIn(operator, MusaBackend.get_decomposition_table())
                self.assertEqual(
                    exported.state_dict["packed_weight"].dtype, torch.uint8
                )
                self.assertTrue(to_edge(exported).exported_program().graph)

    def test_rejects_invalid_shapes_dtypes_and_devices(self):
        packed = torch.empty((3, 1440), dtype=torch.uint8, device="meta")
        for value in (
            torch.empty((2, 2559), device="meta"),
            torch.empty((0, 2560), device="meta"),
            torch.empty((2, 2560), dtype=torch.bfloat16, device="meta"),
            torch.empty((2, 2560), device="cpu"),
            torch.empty((2560, 2), device="meta").t(),
        ):
            with self.assertRaises(RuntimeError):
                GGUF_LINEAR(value, packed, 12)
        with self.assertRaises(RuntimeError):
            GGUF_EMBEDDING(torch.empty((1,), device="meta"), packed, 12)
        with self.assertRaises(ValueError):
            GGUF_LINEAR(torch.empty((1, 2560), device="meta"), packed, 11)
        with self.assertRaises(ValueError):
            GgufLinear(packed, "q4_k_m")


@unittest.skipUnless(
    hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
)
class TestMusaGgufPackedEager(unittest.TestCase):
    def test_embedding_selects_only_requested_rows(self):
        indices = torch.tensor([[2, 0], [1, 2]], device="musa")
        for quant_type in BLOCK_BYTES:
            packed = torch.from_numpy(sample_blocks(quant_type)).to("musa")
            before = packed.clone()
            actual = GgufEmbedding(packed, quant_type)(indices)
            expected = dequantize_gguf(packed, quant_type)[indices]
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(before, packed))
            for invalid in (-1, 3):
                with self.assertRaises(IndexError):
                    GgufEmbedding(packed, quant_type)(
                        torch.tensor([invalid], device="musa")
                    )

    def test_out_rejects_aliases_and_wrong_shapes(self):
        packed = torch.from_numpy(sample_blocks("q4_k")).to("musa")
        value = torch.ones((1, 2560), device="musa")
        with self.assertRaises(ValueError):
            torch.ops.executorch_musa.gguf_linear.out(
                value, packed, 12, out=value[:, :3]
            )
        with self.assertRaises(RuntimeError):
            torch.ops.executorch_musa.gguf_linear.out(value, packed, 12, out=value)


@unittest.skipUnless(
    hasattr(torch, "musa")
    and torch.musa.is_available()
    and os.environ.get("EXECUTORCH_MUSA_GGUF_RUNNER"),
    "requires a runner built with native packed GGUF shims",
)
class TestMusaGgufPackedRunner(unittest.TestCase):
    def test_shared_raw_constants_and_exact_native_outputs(self):
        class Packed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList()
                self.lookups = torch.nn.ModuleList()
                for quant_type, block_bytes in BLOCK_BYTES.items():
                    data = sample_blocks(quant_type).reshape(-1, block_bytes)
                    for offset in (208,) if quant_type == "q6_k" else (0, 2):
                        data[:, offset : offset + 2] = np.array(
                            [1 / 64], dtype=np.float16
                        ).view(np.uint8)
                    packed = torch.from_numpy(data.reshape(3, -1)).to("musa")
                    self.layers.append(GgufLinear(packed, quant_type))
                    self.lookups.append(GgufEmbedding(packed, quant_type))

            def forward(self, value, indices):
                return tuple(
                    output
                    for layer, lookup in zip(self.layers, self.lookups)
                    for output in (layer(value), lookup(indices))
                )

        model = Packed().eval()
        inputs = (
            torch.ones((2, 2560), device="musa"),
            torch.tensor([2, 0, 1], device="musa"),
        )
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            expected = model(*inputs)
        runner = Path(os.environ["EXECUTORCH_MUSA_GGUF_RUNNER"]).resolve()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with torch.no_grad(), _serialization_schema():
                exported = torch.export.export(model, inputs)
                specs = [MusaBackend.generate_method_name_compile_spec("forward")]
                manager = to_edge_transform_and_lower(
                    exported, partitioner=[MusaPartitioner(specs)]
                ).to_executorch()
                pte = Path(save_pte_program(manager, "gguf_packed", directory))
            self.assertEqual(
                len(manager.executorch_program.execution_plan[0].delegates), 1
            )
            payload = (output / "aoti_musa_blob.ptd").read_bytes()
            for layer in model.layers:
                raw = layer.packed_weight.cpu().numpy().tobytes()
                self.assertEqual(payload.count(raw), 1)
                dense = dequantize_gguf(
                    layer.packed_weight,
                    {value: name for name, value in QUANT_IDS.items()}[
                        layer.quant_type
                    ],
                )
                self.assertNotIn(dense.cpu().numpy().tobytes(), payload)
            paths = []
            for index, value in enumerate(inputs):
                path = output / f"input-{index}.bin"
                _write_tensor(value, path)
                paths.append(path)
            result = subprocess.run(
                [
                    str(runner),
                    f"--model_path={pte}",
                    f"--data_path={output}/aoti_musa_blob.ptd",
                    f"--inputs={','.join(map(str, paths))}",
                    f"--output_file={output}/output",
                    "--print_output=none",
                    "--num_executions=3",
                ],
                env=dict(
                    PATH="/usr/bin:/bin",
                    LD_DEBUG="libs",
                    LD_BIND_NOW="1",
                    LD_LIBRARY_PATH=os.environ.get("LD_LIBRARY_PATH", ""),
                ),
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for index, value in enumerate(expected):
                _write_tensor(value, output / f"expected-{index}.bin")
                self.assertEqual(
                    (output / f"output-{index}.bin").read_bytes(),
                    (output / f"expected-{index}.bin").read_bytes(),
                )
            libraries = runtime_libraries(result.stderr)
            self.assertTrue(any("libaoti_cuda_shims.so" in path for path in libraries))
            self.assertFalse(
                any("libtorch" in path or "libpython" in path for path in libraries)
            )


if __name__ == "__main__":
    unittest.main()
