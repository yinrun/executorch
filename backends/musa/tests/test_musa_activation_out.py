"""Activation output reuse regressions; requires EXECUTORCH_MUSA_RUNNER."""

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.export import Dim, export

from executorch.backends.musa import MusaBackend, MusaPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program

from examples.musa.aoti_model_acceptance import (
    _serialization_schema,
    _snapshot_tensor,
    _write_input,
)


class ActivationChain(nn.Module):
    def __init__(self, strided: bool) -> None:
        super().__init__()
        self.strided = strided

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.strided:
            value = value.transpose(1, 2)
        intermediate = torch.nn.functional.hardswish(value)
        return intermediate, torch.nn.functional.hardsigmoid(intermediate)


class MathPair(nn.Module):
    def __init__(self, strided: bool) -> None:
        super().__init__()
        self.strided = strided

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.strided:
            value = value.transpose(1, 2)
        return torch.exp(value), torch.nn.functional.softplus(value)


@unittest.skipUnless(
    hasattr(torch, "musa")
    and torch.musa.is_available()
    and os.environ.get("EXECUTORCH_MUSA_RUNNER"),
    "requires MUSA and EXECUTORCH_MUSA_RUNNER",
)
class TestMusaActivationOutRunner(unittest.TestCase):
    def _check_chain(self, strided: bool, native_math: bool = False) -> None:
        runner = Path(os.environ["EXECUTORCH_MUSA_RUNNER"]).resolve()
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": os.environ["HOME"],
            "LD_BIND_NOW": "1",
            "LD_DEBUG": "libs",
            "LD_LIBRARY_PATH": ":".join(
                str(path)
                for path in (
                    runner.parent / "backends/cuda",
                    runner.parent / "extension/cuda",
                    Path(os.environ.get("MUSA_HOME", "/usr/local/musa")) / "lib",
                )
            ),
        }
        dtypes = (torch.float32,) if native_math else (
            torch.float32, torch.float16, torch.bfloat16
        )
        for dtype in dtypes:
            for planned in (False, True):
                with self.subTest(dtype=dtype, memory_planning=planned):
                    values = (
                        [
                            -104, -90, -20, -1, -0.0, 0, 1,
                            19.999998, 20, 20.000002, 80, 88,
                        ]
                        if native_math
                        else [
                            -4,
                            -3,
                            -2,
                            -0.310546875,
                            0,
                            0.85546875,
                            0.99609375,
                            0.66796875,
                            1,
                            2,
                            3,
                            4,
                        ]
                    )
                    row = torch.tensor(
                        values,
                        device="musa",
                        dtype=dtype,
                    ).reshape(1, 3, 4)
                    model = (
                        MathPair(strided) if native_math else ActivationChain(strided)
                    ).eval()
                    exported = export(
                        model,
                        (row.repeat(2, 1, 1),),
                        dynamic_shapes=({0: Dim("batch", min=1, max=3)},),
                    )
                    specs = [MusaBackend.generate_method_name_compile_spec("forward")]
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        with _serialization_schema(), torch._inductor.config.patch(
                            memory_planning=planned, memory_pool="combined"
                        ):
                            manager = to_edge_transform_and_lower(
                                exported, partitioner=[MusaPartitioner(specs)]
                            ).to_executorch()
                            pte = Path(
                                save_pte_program(manager, "activations", directory)
                            )
                        payload = pte.read_bytes()
                        names = ("exp", "softplus") if native_math else (
                            "hardswish", "hardsigmoid"
                        )
                        for name in names:
                            self.assertIn(
                                f"aoti_torch_musa_{name}_out\0".encode(), payload
                            )
                            self.assertNotIn(
                                f"aoti_torch_musa_{name}\0".encode(), payload
                            )
                        if planned:
                            self.assertIn(b"aoti_torch__alloc_from_pool\0", payload)
                        for batch in (1, 2, 3):
                            with self.subTest(batch=batch), torch.no_grad():
                                value = row.repeat(batch, 1, 1)
                                expected = tuple(
                                    _snapshot_tensor(tensor).flatten()
                                    for tensor in model(value)
                                )
                                _write_input(value, root / "input.bin", (3, 3, 4))
                                result = subprocess.run(
                                    [
                                        str(runner),
                                        f"--model_path={pte}",
                                        f"--data_path={root / 'aoti_musa_blob.ptd'}",
                                        f"--inputs={root / 'input.bin'}",
                                        f"--input_shapes={batch}x3x4",
                                        f"--output_file={root / 'output'}",
                                        "--print_output=none",
                                        "--num_executions=5",
                                    ],
                                    env=environment,
                                    capture_output=True,
                                    text=True,
                                    timeout=90,
                                )
                                self.assertEqual(
                                    result.returncode, 0, result.stdout + result.stderr
                                )
                                loaded = re.findall(
                                    r"calling init: (.+)", result.stderr
                                )
                                self.assertTrue(loaded)
                                self.assertTrue(
                                    any(
                                        "libaoti_cuda_shims.so" in path
                                        for path in loaded
                                    )
                                )
                                for path in loaded:
                                    self.assertFalse(
                                        any(
                                            name in path
                                            for name in (
                                                "libtorch",
                                                "libc10",
                                                "libpython",
                                                "libmusa_python",
                                                "site-packages",
                                                "dist-packages",
                                            )
                                        ),
                                        path,
                                    )
                                for index, reference in enumerate(expected):
                                    actual = torch.frombuffer(
                                        bytearray(
                                            (root / f"output-{index}.bin").read_bytes()
                                        ),
                                        dtype=dtype,
                                    )
                                    torch.testing.assert_close(
                                        actual, reference, atol=0, rtol=0
                                    )

    def test_contiguous_chain(self) -> None:
        self._check_chain(False)

    def test_strided_chain(self) -> None:
        self._check_chain(True)


    def test_contiguous_fp32_math(self) -> None:
        self._check_chain(False, native_math=True)

    def test_strided_fp32_math(self) -> None:
        self._check_chain(True, native_math=True)


if __name__ == "__main__":
    unittest.main()
