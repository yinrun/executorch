import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from executorch.backends.musa import MusaBackend, MusaPartitioner
from executorch.backends.musa.vision_layer_norm import (
    VISION_LAYER_NORM,
    VISION_LAYER_NORM_OUT,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program
from examples.musa.aoti_model_acceptance import _serialization_schema, _write_tensor
from examples.musa.aoti_qwen35_validate import runtime_libraries


class LayerNormModule(torch.nn.Module):
    def forward(self, value, weight, bias):
        return VISION_LAYER_NORM(value, weight, bias, 1e-6)


class TestVisionLayerNormMetadata(unittest.TestCase):
    def test_shapes_and_output(self):
        weight = torch.empty(1024, device="meta")
        bias = torch.empty_like(weight)
        for rows in (800, 1200):
            value = torch.empty((rows, 1024), device="meta")
            output = VISION_LAYER_NORM(value, weight, bias, 1e-6)
            self.assertEqual(output.shape, value.shape)
            self.assertEqual(output.dtype, torch.float32)
            self.assertTrue(output.is_contiguous())
            self.assertIs(
                VISION_LAYER_NORM_OUT(value, weight, bias, 1e-6, out=output), output
            )

    def test_invalid_scope(self):
        weight = torch.empty(1024, device="meta")
        value = torch.empty((800, 1024), device="meta")
        for invalid in (
            torch.empty((0, 1024), device="meta"),
            torch.empty((799, 1024), device="meta"),
            torch.empty((800, 512), device="meta"),
            torch.empty((800, 1024), device="meta", dtype=torch.bfloat16),
            torch.empty((800, 2048), device="meta")[:, ::2],
        ):
            with self.assertRaises(RuntimeError):
                VISION_LAYER_NORM(invalid, weight, weight, 1e-6)
        for epsilon in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(RuntimeError):
                VISION_LAYER_NORM(value, weight, weight, epsilon)
        with self.assertRaises(RuntimeError):
            VISION_LAYER_NORM(value, weight[:1023], weight, 1e-6)
        with self.assertRaises(RuntimeError):
            VISION_LAYER_NORM_OUT(value, weight, weight, 1e-6, out=value[:, :1])


@unittest.skipUnless(
    hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
)
class TestVisionLayerNormEager(unittest.TestCase):
    def test_same_input_and_alias_guards(self):
        torch.manual_seed(1771)
        value = torch.randn((800, 1024), device="musa")
        weight = torch.randn(1024, device="musa")
        bias = torch.randn_like(weight)
        originals = [tensor.clone() for tensor in (value, weight, bias)]
        expected = torch.nn.functional.layer_norm(value, (1024,), weight, bias, 1e-6)
        actual = VISION_LAYER_NORM(value, weight, bias, 1e-6)
        self.assertTrue(
            torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            VISION_LAYER_NORM_OUT(value, weight, bias, 1e-6, out=value)
        output = torch.empty_like(value)
        with self.assertRaisesRegex(ValueError, "overlap"):
            VISION_LAYER_NORM_OUT(value, output[0], bias, 1e-6, out=output)
        for tensor, original in zip((value, weight, bias), originals):
            self.assertTrue(torch.equal(tensor, original))


@unittest.skipUnless(
    hasattr(torch, "musa")
    and torch.musa.is_available()
    and os.environ.get("EXECUTORCH_MUSA_RUNNER"),
    "requires MUSA and EXECUTORCH_MUSA_RUNNER",
)
class TestVisionLayerNormRunner(unittest.TestCase):
    def test_native_matches_torch_musa_for_both_grids(self):
        runner = Path(os.environ["EXECUTORCH_MUSA_RUNNER"]).resolve()
        environment = dict(
            PATH="/usr/bin:/bin",
            LD_BIND_NOW="1",
            LD_DEBUG="libs",
            LD_LIBRARY_PATH=f"{runner.parent}/backends/cuda:{runner.parent}/extension/cuda:/usr/local/musa/lib",
        )
        torch.manual_seed(1771)
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            for rows in (800, 1200):
                value = torch.randn((rows, 1024), device="musa")
                value[:16] = 17 + value[:16] * 1e-5
                weight = torch.randn(1024, device="musa")
                bias = torch.randn_like(weight)
                tensors = (value, weight, bias)
                expected = torch.nn.functional.layer_norm(
                    value, (1024,), weight, bias, 1e-6
                )
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    exported = torch.export.export(LayerNormModule(), tensors)
                    with _serialization_schema(), torch._inductor.config.patch(
                        memory_planning=True, memory_pool="combined"
                    ):
                        specifications = [
                            MusaBackend.generate_method_name_compile_spec("forward")
                        ]
                        manager = to_edge_transform_and_lower(
                            exported, partitioner=[MusaPartitioner(specifications)]
                        ).to_executorch()
                        pte = Path(save_pte_program(manager, "norm", directory))
                    self.assertIn(
                        b"aoti_torch_musa_vision_layer_norm_out\0", pte.read_bytes()
                    )
                    paths = [root / f"input-{index}.bin" for index in range(3)]
                    for tensor, path in zip(tensors, paths):
                        _write_tensor(tensor, path)
                    _write_tensor(expected, root / "expected.bin")
                    for repeat in range(2):
                        result = subprocess.run(
                            [
                                str(runner),
                                f"--model_path={pte}",
                                f"--data_path={root}/aoti_musa_blob.ptd",
                                f"--inputs={','.join(map(str, paths))}",
                                f"--output_file={root}/output-{repeat}",
                                "--print_output=none",
                                "--musa_cache_bytes=536870912",
                            ],
                            env=environment,
                            capture_output=True,
                            text=True,
                            timeout=90,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(
                            (root / f"output-{repeat}-0.bin").read_bytes(),
                            (root / "expected.bin").read_bytes(),
                        )
                        self.assertTrue(runtime_libraries(result.stderr))


if __name__ == "__main__":
    unittest.main()
