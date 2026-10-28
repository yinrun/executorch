"""Scoped reduction selection, storage contracts, and standalone execution."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from executorch.backends.musa import MusaBackend, MusaPartitioner
from executorch.backends.musa.reduction import REDUCE, REDUCE_OUT, replace_reductions
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops
from executorch.extension.export_util.utils import save_pte_program
from examples.musa.aoti_model_acceptance import _serialization_schema, _write_tensor
from examples.musa.aoti_qwen35_validate import runtime_libraries


class ReductionModule(torch.nn.Module):
    def forward(self, value, query, state):
        transposed = value.transpose(0, 1)
        return (
            transposed.sum(-1),
            transposed.mean(-1, keepdim=True),
            query.sum(-1, keepdim=True),
            state.sum(2),
        )


class TestMusaReductionMetadata(unittest.TestCase):
    def test_shapes_dtype_keepdim_and_scalar(self):
        for shape, dtype, axis, modes in (
            ((128,), torch.float32, -1, (False, True)),
            ((3, 256), torch.float32, 1, (False, True)),
            ((1, 1, 1, 1, 1, 1, 2, 1024), torch.float32, -1, (False, True)),
            ((1, 16, 128, 128), torch.float32, 2, (False,)),
            ((1, 1, 16, 128), torch.bfloat16, -1, (False,)),
        ):
            value = torch.empty(shape, dtype=dtype, device="meta")
            for mean in modes:
                for keepdim in (False, True):
                    output = REDUCE(value, axis, keepdim, mean)
                    expected = (
                        value.mean(axis, keepdim=keepdim)
                        if mean
                        else value.sum(axis, keepdim=keepdim)
                    )
                    self.assertEqual(output.shape, expected.shape)
                    self.assertEqual(output.dtype, dtype)
                    self.assertTrue(output.is_contiguous())
                    self.assertIsNot(output, value)
                    self.assertIs(
                        REDUCE_OUT(value, axis, keepdim, mean, out=output), output
                    )

    def test_invalid_scope_and_output(self):
        for shape, dtype, axis, mean in (
            ((3, 64), torch.float32, 1, False),
            ((0, 128), torch.float32, 1, False),
            ((128,), torch.float16, 0, False),
            ((1, 1, 16, 128), torch.bfloat16, -1, True),
            ((3, 128), torch.bfloat16, -1, False),
            ((1, 16, 128, 128), torch.float32, 2, True),
            ((3, 128), torch.float32, 2, False),
            ((3, 128), torch.float32, -3, False),
            ((1,) * 8 + (128,), torch.float32, -1, False),
        ):
            with self.subTest(shape=shape, dtype=dtype, axis=axis, mean=mean):
                with self.assertRaises(RuntimeError):
                    REDUCE(
                        torch.empty(shape, dtype=dtype, device="meta"),
                        axis,
                        False,
                        mean,
                    )
        value = torch.empty((3, 128), device="meta")
        for output in (
            torch.empty((3, 1), device="meta"),
            torch.empty(3, dtype=torch.bfloat16, device="meta"),
            torch.empty(6, device="meta")[::2],
        ):
            with self.assertRaises(RuntimeError):
                REDUCE_OUT(value, 1, False, False, out=output)

    def test_pass_scope_edge_ops_dtype_and_idempotence(self):
        for device in ("cpu", "musa"):
            for target in (
                torch.ops.aten.sum.dim_IntList,
                ops.edge.aten.sum.dim_IntList,
                torch.ops.aten.mean.dim,
            ):
                for dimensions, dtype, selected in (
                    ([-1], None, True),
                    ([1], torch.float32, True),
                    ([1], torch.float64, False),
                    ([0, 1], None, False),
                    ([], None, False),
                    ([2], None, False),
                ):
                    with FakeTensorMode():
                        value = torch.empty((3, 128), device=device)
                    graph = torch.fx.Graph()
                    input_node = graph.placeholder("value")
                    input_node.meta["val"] = value
                    output = graph.call_function(
                        target, (input_node, dimensions), {"dtype": dtype}
                    )
                    graph.output(output)
                    module = torch.fx.GraphModule({}, graph)
                    self.assertEqual(
                        replace_reductions(module), int(selected and device == "musa")
                    )
                    self.assertEqual(replace_reductions(module), 0)
                    self.assertEqual(
                        output.target,
                        REDUCE if selected and device == "musa" else target,
                    )


@unittest.skipUnless(
    hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
)
class TestMusaReductionEager(unittest.TestCase):
    def test_strides_and_inputs_unchanged(self):
        torch.manual_seed(1729)
        pool = torch.randn(1024, device="musa")
        before = pool.clone()
        for strides in ((128, 1), (1, 3), (256, 2), (0, 1)):
            value = pool.as_strided((3, 128), strides, 1)
            for mean in (False, True):
                output = REDUCE(value, -1, False, mean)
                expected = value.mean(-1) if mean else value.sum(-1)
                self.assertTrue(
                    torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
                )
        self.assertTrue(torch.equal(pool, before))

    def test_alias_rejection_and_disjoint_shared_pool(self):
        pool = torch.ones(400, device="musa")
        value = pool[1:385].reshape(3, 128)
        with self.assertRaisesRegex(ValueError, "overlap"):
            REDUCE_OUT(value, 1, False, False, out=pool[2:5])
        output = pool[390:393]
        self.assertEqual(
            REDUCE_OUT(value, 1, False, False, out=output).data_ptr(), output.data_ptr()
        )
        expected = torch.ones_like(pool)
        expected[390:393] = 128
        self.assertTrue(torch.equal(pool, expected))


@unittest.skipUnless(
    hasattr(torch, "musa")
    and torch.musa.is_available()
    and os.environ.get("EXECUTORCH_MUSA_RUNNER"),
    "requires MUSA and EXECUTORCH_MUSA_RUNNER",
)
class TestMusaReductionRunner(unittest.TestCase):
    def test_source_only_export_native_byte_equality(self):
        runner = Path(os.environ["EXECUTORCH_MUSA_RUNNER"]).resolve()
        environment = {
            "PATH": "/usr/bin:/bin",
            "LD_DEBUG": "libs",
            "LD_BIND_NOW": "1",
            "LD_LIBRARY_PATH": ":".join(
                str(path)
                for path in (
                    runner.parent / "backends/cuda",
                    runner.parent / "extension/cuda",
                    Path(os.environ.get("MUSA_HOME", "/usr/local/musa")) / "lib",
                )
            ),
        }
        torch.manual_seed(1729)
        tensors = (
            torch.randn((128, 3), device="musa"),
            torch.randn((1, 1, 16, 128), device="musa", dtype=torch.bfloat16),
            torch.randn((1, 16, 128, 128), device="musa"),
        )
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            for planned in (False, True):
                exported = torch.export.export(ReductionModule(), tensors)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    with _serialization_schema(), torch._inductor.config.patch(
                        memory_planning=planned, memory_pool="combined"
                    ):
                        specs = [
                            MusaBackend.generate_method_name_compile_spec("forward")
                        ]
                        manager = to_edge_transform_and_lower(
                            exported, partitioner=[MusaPartitioner(specs)]
                        ).to_executorch()
                        pte = Path(save_pte_program(manager, "reduction", directory))
                    for symbol in (
                        b"aoti_torch_musa_reduce_out\0",
                        b"aoti_torch_musa_bf16_sum_out\0",
                    ):
                        self.assertIn(symbol, pte.read_bytes())
                    paths = [
                        root / f"input-{index}.bin" for index in range(len(tensors))
                    ]
                    for tensor, path in zip(tensors, paths):
                        _write_tensor(tensor, path)
                    result = subprocess.run(
                        [
                            str(runner),
                            f"--model_path={pte}",
                            f"--data_path={root / 'aoti_musa_blob.ptd'}",
                            f"--inputs={','.join(map(str, paths))}",
                            f"--output_file={root / 'output'}",
                            "--print_output=none",
                            "--num_executions=3",
                            "--musa_cache_bytes=536870912",
                        ],
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=90,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    for index, expected in enumerate(ReductionModule()(*tensors)):
                        _write_tensor(expected, root / f"expected-{index}.bin")
                        self.assertEqual(
                            (root / f"output-{index}.bin").read_bytes(),
                            (root / f"expected-{index}.bin").read_bytes(),
                        )
                    libraries = runtime_libraries(result.stderr)
                    self.assertTrue(libraries)
                    self.assertFalse(
                        any(
                            "libtorch" in path or "libpython" in path
                            for path in libraries
                        )
                    )
                    stats = [
                        json.loads(line.removeprefix("MUSA ALLOCATION CACHE "))
                        for line in result.stderr.splitlines()
                        if line.startswith("MUSA ALLOCATION CACHE {")
                    ]
                    self.assertEqual(len(stats), 1)
                    for name in ("cached_bytes", "cached_blocks", "live_blocks"):
                        self.assertEqual(stats[0][name], 0)


if __name__ == "__main__":
    unittest.main()
