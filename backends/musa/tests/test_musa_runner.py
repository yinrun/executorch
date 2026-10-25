"""Runner regressions; set EXECUTORCH_MUSA_RUNNER to a MUSA executor_runner."""

import os
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
    _run_runner,
    _serialization_schema,
    _write_input,
)


@unittest.skipUnless(
    hasattr(torch, "musa")
    and torch.musa.is_available()
    and os.environ.get("EXECUTORCH_MUSA_RUNNER"),
    "requires MUSA and EXECUTORCH_MUSA_RUNNER",
)
class TestMusaRunner(unittest.TestCase):
    def test_hardsigmoid_rounding(self) -> None:
        runner = Path(os.environ["EXECUTORCH_MUSA_RUNNER"])
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                row = torch.tensor(
                    [
                        -4,
                        -3,
                        -0.310546875,
                        0,
                        0.85546875,
                        0.99609375,
                        0.66796875,
                        1,
                        3,
                        4,
                    ],
                    device="musa",
                    dtype=dtype,
                )
                model = nn.Hardsigmoid().eval()
                example = row.repeat(2, 1)
                exported = export(
                    model,
                    (example,),
                    dynamic_shapes=({0: Dim("batch", min=1, max=3)},),
                )
                decomposed = exported.run_decompositions(
                    MusaBackend.get_decomposition_table()
                )
                self.assertTrue(
                    any(
                        node.target == torch.ops.executorch_musa.hardsigmoid.default
                        for node in decomposed.graph.nodes
                    )
                )
                specs = [MusaBackend.generate_method_name_compile_spec("forward")]
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    with _serialization_schema():
                        manager = to_edge_transform_and_lower(
                            exported, partitioner=[MusaPartitioner(specs)]
                        ).to_executorch()
                        pte = Path(save_pte_program(manager, "hardsigmoid", directory))
                    for batch in (1, 2, 3):
                        with self.subTest(batch=batch), torch.no_grad():
                            value = row.repeat(batch, 1)
                            expected = model(value).cpu().flatten()
                            _write_input(value, root / "input.bin", (3, row.numel()))
                            _run_runner(
                                runner,
                                pte,
                                root / "aoti_musa_blob.ptd",
                                root / "input.bin",
                                root / "output",
                                tuple(value.shape),
                            )
                            actual = torch.frombuffer(
                                bytearray((root / "output-0.bin").read_bytes()),
                                dtype=dtype,
                            )
                            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_convolution_bias_rounding(self) -> None:
        runner = Path(os.environ["EXECUTORCH_MUSA_RUNNER"])
        for dimensions, factory in enumerate((nn.Conv1d, nn.Conv2d, nn.Conv3d), 1):
            for dtype, small in (
                (torch.float16, 0.00048828125),
                (torch.bfloat16, 0.00390625),
            ):
                with self.subTest(dimensions=dimensions, dtype=dtype):
                    shape = (2, 2, *([1] * dimensions))
                    model = factory(2, 2, 1).eval().to(device="musa", dtype=dtype)
                    with torch.no_grad():
                        model.weight.fill_(1)
                        model.bias.fill_(-1)
                        example = torch.tensor(
                            [1.0, small, 1.0, small], device="musa", dtype=dtype
                        ).reshape(shape)
                        expected = model(example).cpu().flatten()
                    torch.testing.assert_close(
                        expected, torch.full_like(expected, small), atol=0, rtol=0
                    )
                    exported = export(
                        model,
                        (example,),
                        dynamic_shapes=({0: Dim("batch", min=1, max=3)},),
                    )
                    specs = [MusaBackend.generate_method_name_compile_spec("forward")]
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        with _serialization_schema():
                            manager = to_edge_transform_and_lower(
                                exported, partitioner=[MusaPartitioner(specs)]
                            ).to_executorch()
                            pte = Path(
                                save_pte_program(manager, "biased_conv", directory)
                            )
                        _write_input(example, root / "input.bin", (3, *shape[1:]))
                        _run_runner(
                            runner,
                            pte,
                            root / "aoti_musa_blob.ptd",
                            root / "input.bin",
                            root / "output",
                            shape,
                        )
                        actual = torch.frombuffer(
                            bytearray((root / "output-0.bin").read_bytes()), dtype=dtype
                        )
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
