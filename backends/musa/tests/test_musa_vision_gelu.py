import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from executorch.backends.musa import MusaBackend, MusaPartitioner
from executorch.backends.musa.vision_gelu import VISION_GELU, VISION_GELU_OUT
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program
from examples.musa.aoti_model_acceptance import _serialization_schema, _write_tensor
from examples.musa.aoti_qwen35_validate import runtime_libraries


class GeluModule(torch.nn.Module):
    def __init__(self, mode, scale=False):
        super().__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, value):
        if self.scale:
            value = value * 0.125
        return VISION_GELU(value, self.mode)


class TestVisionGeluMetadata(unittest.TestCase):
    def test_shapes_and_output(self):
        for length in (200, 300, 800, 1200):
            value = torch.empty((length, 4096), device="meta")
            output = VISION_GELU(value, 0)
            self.assertEqual(output.shape, value.shape)
            self.assertEqual(output.dtype, torch.float32)
            self.assertTrue(output.is_contiguous())
            self.assertIs(VISION_GELU_OUT(value, 0, out=output), output)

    def test_invalid_scope(self):
        for shape in ((800,), (1, 800, 4096), (800, 4095), (799, 4096)):
            with self.assertRaises(RuntimeError):
                VISION_GELU(torch.empty(shape, device="meta"), 0)
        value = torch.empty((800, 4096), device="meta")
        for invalid in (value.to(torch.bfloat16), value.transpose(0, 1)):
            with self.assertRaises(RuntimeError):
                VISION_GELU(invalid, 1)
        with self.assertRaises(RuntimeError):
            VISION_GELU_OUT(value, 0, out=value[:, :1])
        for mode in (-1, 2):
            with self.assertRaises(RuntimeError):
                VISION_GELU(value, mode)


@unittest.skipUnless(
    hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
)
class TestVisionGeluEager(unittest.TestCase):
    def test_same_input_and_alias_guards(self):
        torch.manual_seed(1773)
        value = torch.randn((800, 4096), device="musa")
        original = value.clone()
        expected = torch.nn.functional.gelu(value)
        output = torch.empty_like(value)
        self.assertIs(VISION_GELU_OUT(value, 0, out=output), output)
        self.assertTrue(
            torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
        )
        self.assertTrue(torch.equal(value, original))
        with self.assertRaisesRegex(ValueError, "overlap"):
            VISION_GELU_OUT(value, 0, out=value)


@unittest.skipUnless(
    hasattr(torch, "musa")
    and torch.musa.is_available()
    and os.environ.get("EXECUTORCH_MUSA_RUNNER"),
    "requires MUSA and EXECUTORCH_MUSA_RUNNER",
)
class TestVisionGeluRunner(unittest.TestCase):
    def test_native_matches_torch_musa_for_both_grids(self):
        runner = Path(os.environ["EXECUTORCH_MUSA_RUNNER"]).resolve()
        environment = dict(
            PATH="/usr/bin:/bin",
            LD_BIND_NOW="1",
            LD_DEBUG="libs",
            LD_LIBRARY_PATH=f"{runner.parent}/backends/cuda:{runner.parent}/extension/cuda:/usr/local/musa/lib",
        )
        torch.manual_seed(1773)
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            for length, mode, scale in (
                (200, 0, False),
                (300, 0, True),
                (800, 1, False),
                (1200, 1, True),
            ):
                value = torch.randn((length, 4096), device="musa") * 9
                value[:, 0] = 0
                value[:, 1] = 1000
                value[:, 2] = -1000
                value[:, 3] = 1e-6
                value[:, 4] = -1e-6
                expected = torch.nn.functional.gelu(
                    value * 0.125 if scale else value,
                    approximate="tanh" if mode else "none",
                )
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    exported = torch.export.export(GeluModule(mode, scale), (value,))
                    with _serialization_schema(), torch._inductor.config.patch(
                        memory_planning=True, memory_pool="combined"
                    ):
                        specifications = [
                            MusaBackend.generate_method_name_compile_spec("forward")
                        ]
                        manager = to_edge_transform_and_lower(
                            exported, partitioner=[MusaPartitioner(specifications)]
                        ).to_executorch()
                        pte = Path(save_pte_program(manager, "gelu", directory))
                    self.assertIn(
                        b"aoti_torch_musa_vision_gelu_out\0", pte.read_bytes()
                    )
                    _write_tensor(value, root / "input.bin")
                    _write_tensor(expected, root / "expected.bin")
                    for repeat in range(2):
                        result = subprocess.run(
                            [
                                str(runner),
                                f"--model_path={pte}",
                                f"--data_path={root}/aoti_musa_blob.ptd",
                                f"--inputs={root}/input.bin",
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
                        libraries = runtime_libraries(result.stderr)
                        self.assertIn(
                            str(runner.parent / "backends/cuda/libaoti_cuda_shims.so"),
                            libraries,
                        )
                        self.assertFalse(
                            any(
                                "libtorch" in name or "libpython" in name
                                for name in libraries
                            )
                        )


if __name__ == "__main__":
    unittest.main()
