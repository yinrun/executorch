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
    GGUF_PARTS,
    GgufEmbedding,
    GgufLinear,
    QUANT_IDS,
)
from executorch.backends.musa.tests.test_musa_gguf import sample_blocks
from executorch.backends.musa.q6_predecoded import Q6_PREDECODED, predecode_q6
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program
from examples.musa.aoti_model_acceptance import _serialization_schema, _write_tensor
from examples.musa.aoti_qwen35_validate import runtime_libraries


class TestMusaGgufPackedMetadata(unittest.TestCase):
    def test_split_k_is_opt_in_and_keeps_partial_output_opaque(self):
        for encoding, block_bytes in (("q4_k", 144), (Q6_PREDECODED, 320)):
            packed = torch.empty(
                (129, 17 * block_bytes), dtype=torch.uint8, device="meta"
            )
            value = torch.empty((1, 1, 1, 4352), device="meta")
            default = torch.export.export(GgufLinear(packed, encoding), (value,))
            self.assertNotIn(GGUF_PARTS, [node.target for node in default.graph.nodes])
            model = GgufLinear(packed, encoding, split_k=True)
            exported = torch.export.export(model, (value,))
            partials = [
                node for node in exported.graph.nodes if node.target == GGUF_PARTS
            ]
            self.assertEqual(len(partials), 1)
            self.assertEqual(tuple(partials[0].meta["val"].shape), (4, 1, 1, 1, 129))
            self.assertNotIn(
                GGUF_LINEAR, [node.target for node in exported.graph.nodes]
            )
            self.assertNotIn(
                GGUF_DECODE, [node.target for node in exported.graph.nodes]
            )
            self.assertNotIn(GGUF_PARTS, MusaBackend.get_decomposition_table())
            self.assertEqual(exported.state_dict["packed_weight"].dtype, torch.uint8)
            self.assertTrue(to_edge(exported).exported_program().graph)
            self.assertIn(
                "aoti_torch_musa_gguf_parts_out",
                MusaBackend.get_supported_fallback_kernels(),
            )

    def test_parts_reject_unsupported_geometry_and_output_metadata(self):
        packed = torch.empty((128, 16 * 144), dtype=torch.uint8, device="meta")
        value = torch.empty((1, 4096), device="meta")
        for invalid in (
            torch.empty((2, 4096), device="meta"),
            torch.empty((1, 4096), dtype=torch.bfloat16, device="meta"),
        ):
            with self.assertRaises(RuntimeError):
                GGUF_PARTS(invalid, packed, 12)
        for rows, blocks, quant_type, block_bytes in (
            (127, 16, 12, 144),
            (128, 10, 12, 144),
            (128, 16, 13, 176),
            (128, 16, 14, 210),
        ):
            with self.assertRaises(RuntimeError):
                GGUF_PARTS(
                    torch.empty((1, blocks * 256), device="meta"),
                    torch.empty(
                        (rows, blocks * block_bytes), dtype=torch.uint8, device="meta"
                    ),
                    quant_type,
                )
        for output in (
            torch.empty((1, 128), device="meta"),
            torch.empty((4, 1, 128), dtype=torch.bfloat16, device="meta"),
            torch.empty((4, 128, 2), device="meta")[:, :, 0].unsqueeze(1),
        ):
            with self.assertRaises(RuntimeError):
                torch.ops.executorch_musa.gguf_parts.out(value, packed, 12, out=output)

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
    def test_split_k_outputs_bias_and_weights_are_preserved(self):
        for encoding in ("q4_k", "q6_k"):
            raw = sample_blocks(encoding, rows=129, blocks_per_row=17)
            block_bytes = 144 if encoding == "q4_k" else 210
            blocks = raw.reshape(-1, block_bytes)
            for offset in (0, 2) if encoding == "q4_k" else (208,):
                blocks[:, offset : offset + 2] = np.array(
                    [1 / 64], dtype=np.float16
                ).view(np.uint8)
            packed = torch.from_numpy(raw).to("musa")
            if encoding == "q6_k":
                packed = predecode_q6(packed)
                encoding = Q6_PREDECODED
                block_bytes = 320
            before = packed.clone()
            bias = torch.arange(129, dtype=torch.float32, device="musa") / 128
            model = GgufLinear(packed, encoding, bias, split_k=True)
            value = torch.linspace(-0.125, 0.125, 4352, device="musa").reshape(
                1, 1, 4352
            )
            with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
                partials = GGUF_PARTS(value, packed, model.quant_type)
                for part in range(4):
                    first, last = 17 * part // 4, 17 * (part + 1) // 4
                    expected = GGUF_LINEAR(
                        value[..., first * 256 : last * 256].contiguous(),
                        packed[
                            :, first * block_bytes : last * block_bytes
                        ].contiguous(),
                        model.quant_type,
                    )
                    self.assertTrue(torch.equal(partials[part], expected))
                expected = (
                    ((partials[0] + partials[1]) + partials[2]) + partials[3] + bias
                )
                self.assertTrue(torch.equal(model(value), expected))
                out = torch.empty_like(partials)
                torch.ops.executorch_musa.gguf_parts.out(
                    value, packed, model.quant_type, out=out
                )
                self.assertTrue(torch.equal(out, partials))
                self.assertTrue(torch.equal(before, packed))

    def test_partial_output_tail_aliases_and_unaligned_inputs_are_rejected(self):
        packed = torch.zeros((128, 16 * 144), dtype=torch.uint8, device="musa")
        owner = torch.zeros(4096 + 128, device="musa")
        value = owner[128:].reshape(1, 4096)
        output = owner[:512].reshape(4, 1, 128)
        with self.assertRaises(ValueError):
            torch.ops.executorch_musa.gguf_parts.out(value, packed, 12, out=output)
        with self.assertRaises(ValueError):
            GGUF_PARTS(owner[1:4097].reshape(1, 4096), packed, 12)
        raw = torch.zeros(128 * 16 * 144 + 1, dtype=torch.uint8, device="musa")
        with self.assertRaises(ValueError):
            GGUF_PARTS(value, raw[1:].reshape(128, 16 * 144), 12)

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
