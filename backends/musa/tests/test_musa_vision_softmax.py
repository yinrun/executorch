import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from executorch.backends.musa import MusaBackend, MusaPartitioner
from executorch.backends.musa.vision_softmax import VISION_SOFTMAX, VISION_SOFTMAX_OUT
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program
from examples.musa.aoti_model_acceptance import _serialization_schema, _write_tensor
from examples.musa.aoti_qwen35_validate import runtime_libraries


class SoftmaxModule(torch.nn.Module):
    def __init__(self, scale=False):
        super().__init__()
        self.scale = scale

    def forward(self, value):
        if self.scale:
            value = value * 0.125
        return VISION_SOFTMAX(value)


class TestVisionSoftmaxMetadata(unittest.TestCase):
    def test_shapes_and_output(self):
        for length in (800, 1200):
            value = torch.empty((16, length, length), device="meta")
            output = VISION_SOFTMAX(value)
            self.assertEqual(output.shape, value.shape)
            self.assertEqual(output.dtype, torch.float32)
            self.assertTrue(output.is_contiguous())
            self.assertIs(VISION_SOFTMAX_OUT(value, out=output), output)

    def test_invalid_scope(self):
        for shape in ((800, 800), (1, 800, 800), (16, 800, 801), (16, 799, 799)):
            with self.assertRaises(RuntimeError):
                VISION_SOFTMAX(torch.empty(shape, device="meta"))
        value = torch.empty((16, 800, 800), device="meta")
        for invalid in (value.to(torch.bfloat16), value.transpose(1, 2)):
            with self.assertRaises(RuntimeError):
                VISION_SOFTMAX(invalid)
        with self.assertRaises(RuntimeError):
            VISION_SOFTMAX_OUT(value, out=value[:, :, :1])


@unittest.skipUnless(
    hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
)
class TestVisionSoftmaxEager(unittest.TestCase):
    def test_same_input_and_alias_guards(self):
        torch.manual_seed(1773)
        value = torch.randn((16, 800, 800), device="musa")
        original = value.clone()
        expected = torch.softmax(value, dim=-1)
        output = torch.empty_like(value)
        self.assertIs(VISION_SOFTMAX_OUT(value, out=output), output)
        self.assertTrue(
            torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
        )
        self.assertTrue(torch.equal(value, original))
        with self.assertRaisesRegex(ValueError, "overlap"):
            VISION_SOFTMAX_OUT(value, out=value)


@unittest.skipUnless(
    hasattr(torch, "musa")
    and torch.musa.is_available()
    and os.environ.get("EXECUTORCH_MUSA_RUNNER"),
    "requires MUSA and EXECUTORCH_MUSA_RUNNER",
)
class TestVisionSoftmaxRunner(unittest.TestCase):
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
            for length, scale in ((800, False), (1200, False), (1200, True)):
                value = torch.randn((16, length, length), device="musa") * 9
                value[:, 0] = 0
                value[:, 1] += 1000
                value[:, 2] -= 1000
                value[:, 3, 0] = 1000
                value[:, 4, 0] = -1000
                expected = torch.softmax(value * 0.125 if scale else value, dim=-1)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    exported = torch.export.export(SoftmaxModule(scale), (value,))
                    with _serialization_schema(), torch._inductor.config.patch(
                        memory_planning=True, memory_pool="combined"
                    ):
                        specifications = [
                            MusaBackend.generate_method_name_compile_spec("forward")
                        ]
                        manager = to_edge_transform_and_lower(
                            exported, partitioner=[MusaPartitioner(specifications)]
                        ).to_executorch()
                        pte = Path(save_pte_program(manager, "softmax", directory))
                    self.assertIn(
                        b"aoti_torch_musa_vision_softmax_out\0", pte.read_bytes()
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
