"""K-quant bit layouts and same-block MUSA decoding, not model acceptance."""

import unittest
from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from executorch.backends.musa import MusaBackend, MusaPartitioner
from executorch.backends.musa.gguf import (
    BLOCK_BYTES,
    GGUF_DECODE,
    decode_fp16_scale,
    dequantize_gguf,
)
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program
from examples.musa.aoti_gguf_inventory import tensor_inventory
from examples.musa.aoti_model_acceptance import _serialization_schema, _write_tensor
from examples.musa.aoti_qwen35_validate import runtime_libraries


def sample_blocks(quant_type, rows=3, blocks_per_row=10):
    generator = np.random.default_rng(1729)
    block_bytes = BLOCK_BYTES[quant_type]
    blocks = generator.integers(
        0, 256, (rows * blocks_per_row, block_bytes), dtype=np.uint8
    )
    scales = np.array(
        [0, 2**-24, 2**-14, 0.03125, -0.125, 2, 127, 0.0078125], dtype=np.float16
    )
    offsets = (208,) if quant_type == "q6_k" else (0, 2)
    for offset in offsets:
        values = np.resize(scales, blocks.shape[0]).view(np.uint8).reshape(-1, 2)
        blocks[:, offset : offset + 2] = values
    return blocks.reshape(rows, -1)


class TestMusaGgufMetadata(unittest.TestCase):
    def test_export_retains_decoder_until_backend_decomposition(self):
        class Decode(torch.nn.Module):
            def __init__(self, quant_type):
                super().__init__()
                self.quant_type = quant_type

            def forward(self, raw):
                return dequantize_gguf(raw, self.quant_type)

        for quant_type, block_bytes in BLOCK_BYTES.items():
            raw = torch.empty((3, 10 * block_bytes), device="meta", dtype=torch.uint8)
            exported = torch.export.export(Decode(quant_type), (raw,))
            self.assertIn(GGUF_DECODE, [node.target for node in exported.graph.nodes])
            self.assertIn(GGUF_DECODE, MusaBackend.get_decomposition_table())
            manager = to_edge(exported)
            self.assertTrue(manager.exported_program().graph)

    def test_shapes_and_dtypes(self):
        for quant_type, block_bytes in BLOCK_BYTES.items():
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                raw = torch.empty(
                    (3, 10 * block_bytes), device="meta", dtype=torch.uint8
                )
                output = dequantize_gguf(raw, quant_type, dtype)
                self.assertEqual(output.shape, (3, 2560))
                self.assertEqual(output.dtype, dtype)

    def test_rejects_invalid_contracts(self):
        for shape, dtype, device in (
            ((3, 143), torch.uint8, "meta"),
            ((0, 144), torch.uint8, "meta"),
            ((3, 0), torch.uint8, "meta"),
            ((144,), torch.uint8, "meta"),
            ((3, 144), torch.int8, "meta"),
            ((3, 144), torch.uint8, "cpu"),
        ):
            with self.assertRaises(RuntimeError):
                dequantize_gguf(torch.empty(shape, dtype=dtype, device=device), "q4_k")
        raw = torch.empty((3, 144), dtype=torch.uint8, device="meta")
        with self.assertRaises(ValueError):
            dequantize_gguf(raw, "q4_k_m")
        with self.assertRaises(ValueError):
            dequantize_gguf(raw, "q4_k", torch.int32)
        with self.assertRaises(RuntimeError):
            dequantize_gguf(
                torch.empty((144, 3), dtype=torch.uint8, device="meta").t(), "q4_k"
            )

    def test_inventory_preserves_distinct_mixed_formats(self):
        tensors = [
            SimpleNamespace(
                name=quant_type,
                tensor_type=SimpleNamespace(name=quant_type.upper()),
                shape=[2560, 3],
                n_elements=7680,
                n_bytes=30 * block_bytes,
                data=sample_blocks(quant_type),
            )
            for quant_type, block_bytes in BLOCK_BYTES.items()
        ]
        result = tensor_inventory(tensors)
        self.assertEqual(
            [tensor["encoding"] for tensor in result], ["Q4_K", "Q5_K", "Q6_K"]
        )
        self.assertTrue(all(tensor["torch_shape"] == [3, 2560] for tensor in result))
        with self.assertRaises(ValueError):
            tensor_inventory([tensors[0], tensors[0]])
        tensors[0].n_bytes -= 1
        with self.assertRaises(ValueError):
            tensor_inventory(tensors)


@unittest.skipUnless(
    hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
)
class TestMusaGgufEager(unittest.TestCase):
    def test_changed_quantized_bit_is_detectable(self):
        for quant_type, block_bytes in BLOCK_BYTES.items():
            data = np.zeros((1, block_bytes), dtype=np.uint8)
            scale_offset = 208 if quant_type == "q6_k" else 0
            data[:, scale_offset : scale_offset + 2] = np.array(
                [1], dtype=np.float16
            ).view(np.uint8)
            if quant_type == "q6_k":
                data[:, 192:208] = 1
                quant_offset = 0
            else:
                data[:, 4:8] = 1
                quant_offset = 16 if quant_type == "q4_k" else 48
            raw = torch.from_numpy(data).to("musa")
            before = dequantize_gguf(raw, quant_type)
            changed = raw.clone()
            changed[0, quant_offset] = 1
            after = dequantize_gguf(changed, quant_type)
            self.assertEqual(torch.count_nonzero(before != after).item(), 1)
            self.assertEqual(torch.max(torch.abs(before - after)).item(), 1)

    def test_all_fp16_scale_encodings(self):
        bits = np.arange(65536, dtype=np.uint16)
        raw = torch.from_numpy(bits.view(np.uint8).reshape(-1, 2)).to("musa")
        expected = torch.from_numpy(bits.view(np.float16).astype(np.float32))
        actual = decode_fp16_scale(raw, 0).reshape(-1).cpu()
        finite = torch.isfinite(expected)
        self.assertTrue(
            torch.equal(
                actual[finite].view(torch.int32), expected[finite].view(torch.int32)
            )
        )
        self.assertTrue(torch.equal(torch.isnan(actual), torch.isnan(expected)))
        self.assertTrue(
            torch.equal(actual[torch.isinf(expected)], expected[torch.isinf(expected)])
        )

    def test_matches_gguf_reference_without_changing_raw_blocks(self):
        import gguf

        for quant_type in BLOCK_BYTES:
            for columns in (256, 2560, 4096, 9216, 12288):
                data = sample_blocks(quant_type, blocks_per_row=columns // 256)
                expected = torch.from_numpy(
                    gguf.dequantize(data, gguf.GGMLQuantizationType[quant_type.upper()])
                )
                storage = torch.empty(data.size + 4, dtype=torch.uint8, device="musa")
                raw = storage[2:-2].reshape(data.shape)
                raw.copy_(torch.from_numpy(data))
                before = raw.clone()
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    output = dequantize_gguf(raw, quant_type, dtype)
                    self.assertEqual(output.device.type, "musa")
                    self.assertTrue(
                        torch.equal(output.cpu(), expected.to(dtype)),
                        (quant_type, columns, dtype),
                    )
                self.assertTrue(torch.equal(raw, before))


@unittest.skipUnless(
    hasattr(torch, "musa")
    and torch.musa.is_available()
    and os.environ.get("EXECUTORCH_MUSA_RUNNER"),
    "requires MUSA and EXECUTORCH_MUSA_RUNNER",
)
class TestMusaGgufRunner(unittest.TestCase):
    def test_three_formats_execute_as_native_delegate(self):
        class Decode(torch.nn.Module):
            def forward(self, q4, q5, q6):
                return tuple(
                    dequantize_gguf(raw, quant_type)
                    for raw, quant_type in zip((q4, q5, q6), BLOCK_BYTES)
                )

        inputs = tuple(
            torch.from_numpy(sample_blocks(quant_type)).to("musa")
            for quant_type in BLOCK_BYTES
        )
        expected = Decode()(*inputs)
        runner = Path(os.environ["EXECUTORCH_MUSA_RUNNER"]).resolve()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with torch.no_grad(), _serialization_schema():
                exported = torch.export.export(Decode(), inputs)
                specs = [MusaBackend.generate_method_name_compile_spec("forward")]
                manager = to_edge_transform_and_lower(
                    exported, partitioner=[MusaPartitioner(specs)]
                ).to_executorch()
                pte = Path(save_pte_program(manager, "gguf_decode", directory))
            self.assertEqual(
                len(manager.executorch_program.execution_plan[0].delegates), 1
            )
            input_paths = []
            for index, value in enumerate(inputs):
                path = output / f"input-{index}.bin"
                _write_tensor(value, path)
                input_paths.append(path)
            environment = dict(
                PATH="/usr/bin:/bin",
                LD_DEBUG="libs",
                LD_BIND_NOW="1",
                LD_LIBRARY_PATH=os.environ.get("LD_LIBRARY_PATH", ""),
            )
            result = subprocess.run(
                [
                    str(runner),
                    f"--model_path={pte}",
                    f"--data_path={output}/aoti_musa_blob.ptd",
                    f"--inputs={','.join(map(str, input_paths))}",
                    f"--output_file={output}/output",
                    "--print_output=none",
                    "--num_executions=3",
                ],
                env=environment,
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
