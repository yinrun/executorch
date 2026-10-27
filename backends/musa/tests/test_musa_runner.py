"""Runner regressions; set EXECUTORCH_MUSA_RUNNER to a MUSA executor_runner."""

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from torch.export import Dim, export

from executorch.backends.musa import MusaBackend, MusaPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program

from examples.musa.aoti_model_acceptance import (
    _check_predictions,
    _record_tensor_artifact,
    _run_runner,
    _serialization_schema,
    _snapshot_tensor,
    _write_input,
    _write_tensor,
    main as acceptance_main,
)


class TestModelPredictions(unittest.TestCase):
    def test_first_and_repeated_prediction_mismatches_fail_acceptance(self) -> None:
        for model_name in (
            "resnet18",
            "mobilenet_v2",
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "efficientnet_b0",
            "densenet121",
            "convnext_tiny",
            "vit_b_16",
            "token_decoder",
            "deeplabv3_mobilenet_v3_large",
        ):
            for mismatch in (None, 0, 1):
                with self.subTest(
                    model=model_name, mismatch=mismatch
                ), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    shape = (1, 2, 1, 1) if model_name.startswith("deeplab") else (1, 2)
                    expected = _write_tensor(
                        torch.tensor([1.0, 1.00001]).reshape(shape),
                        root / "expected-0.bin",
                    )
                    changed = expected.flip(1)
                    torch.testing.assert_close(changed, expected, atol=1e-4, rtol=1e-4)
                    input_dtype = (
                        torch.int64 if model_name == "token_decoder" else torch.float32
                    )
                    _write_tensor(
                        torch.ones(1, 16, dtype=input_dtype), root / "input-0.bin"
                    )
                    exported = (
                        root / "model.pte",
                        root / "model.ptd",
                        root / "input-0.bin",
                        expected,
                        nn.Identity(),
                        (1, 16),
                    )
                    arguments = [
                        "acceptance",
                        "--model",
                        model_name,
                        "--output-dir",
                        str(root),
                        "--runner",
                        "/runner",
                        "--runner-repeats",
                        "2",
                    ]

                    def run(runner, pte, ptd, input_path, output_prefix, shape):
                        repeat = 1 if "repeat" in output_prefix.name else 0
                        value = changed if mismatch == repeat else expected
                        _write_tensor(value, Path(f"{output_prefix}-0.bin"))
                        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

                    with mock.patch("sys.argv", arguments), mock.patch(
                        "examples.musa.aoti_model_acceptance.export_model",
                        return_value=exported,
                    ), mock.patch(
                        "examples.musa.aoti_model_acceptance._run_runner",
                        side_effect=run,
                    ), mock.patch(
                        "builtins.print"
                    ):
                        if mismatch is None:
                            acceptance_main()
                            report = json.loads((root / "result.json").read_text())
                            predictions = _check_predictions(
                                model_name, expected, expected
                            )
                            self.assertTrue(predictions)
                            self.assertEqual(
                                report["artifact_cases"][0]["predictions"], predictions
                            )
                            for key, value in predictions.items():
                                self.assertEqual(report[key], value)
                                self.assertEqual(
                                    report["runner_repeat_results"][0][key], value
                                )
                        else:
                            with self.assertRaisesRegex(
                                AssertionError, "predictions differ"
                            ):
                                acceptance_main()
                            self.assertFalse((root / "result.json").exists())

    def test_feature_outputs_do_not_receive_classification_checks(self) -> None:
        expected = torch.tensor([[1.0, 2.0]])
        actual = expected.flip(-1)
        self.assertEqual(_check_predictions("vit_b_16", actual, expected, True), {})
        self.assertEqual(_check_predictions("mlp", actual, expected), {})


class TestTensorSnapshots(unittest.TestCase):
    def test_cpu_snapshots_are_owned_and_byte_exact(self) -> None:
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int64):
            for shape in ((), (0,), (2, 4)):
                with self.subTest(
                    dtype=dtype, shape=shape
                ), tempfile.TemporaryDirectory() as directory:
                    source = torch.ones(shape, dtype=dtype)
                    if source.ndim == 2:
                        source = source.transpose(0, 1)
                    if source.is_floating_point():
                        source.requires_grad_()
                    path = Path(directory) / "tensor.bin"
                    snapshot = _write_tensor(source, path)
                    self.assertEqual(snapshot.device.type, "cpu")
                    self.assertTrue(snapshot.is_contiguous())
                    self.assertFalse(snapshot.requires_grad)
                    with torch.no_grad():
                        source.zero_()
                    torch.testing.assert_close(
                        snapshot, torch.ones_like(snapshot), atol=0, rtol=0
                    )
                    record = _record_tensor_artifact(snapshot, path)
                    self.assertEqual(
                        record["nbytes"], snapshot.numel() * snapshot.element_size()
                    )

    def test_snapshot_detects_same_size_tail_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expected.bin"
            snapshot = _write_tensor(torch.ones(2, 1000, dtype=torch.bfloat16), path)
            path.write_bytes(path.read_bytes()[:-32] + bytes(32))
            with self.assertRaisesRegex(ValueError, "tensor snapshot"):
                _record_tensor_artifact(snapshot, path)

    def test_acceptance_rejects_artifact_mutation_during_runner(self) -> None:
        for mutated in (None, "expected-0.bin", "input-0.bin", "runner-output-1-0.bin"):
            with self.subTest(
                mutated=mutated
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                expected = _write_tensor(
                    torch.ones(1, 1000, dtype=torch.bfloat16), root / "expected-0.bin"
                )
                _write_input(
                    torch.ones(1, 16, dtype=torch.bfloat16), root / "input-0.bin", None
                )
                arguments = [
                    "acceptance",
                    "--model",
                    "mlp",
                    "--dtype",
                    "bf16",
                    "--output-dir",
                    str(root),
                    "--runner",
                    "/runner",
                    "--runner-repeats",
                    "2",
                ]
                exported = (
                    root / "model.pte",
                    root / "model.ptd",
                    root / "input-0.bin",
                    expected,
                    nn.Identity(),
                    (1, 16),
                )

                def run(runner, pte, ptd, input_path, output_prefix, shape):
                    _write_tensor(expected, Path(f"{output_prefix}-0.bin"))
                    if mutated is not None and "repeat" in output_prefix.name:
                        path = root / mutated
                        path.write_bytes(path.read_bytes()[:-32] + bytes(32))
                    return subprocess.CompletedProcess([], 0, stdout="", stderr="")

                with mock.patch("sys.argv", arguments), mock.patch(
                    "examples.musa.aoti_model_acceptance.export_model",
                    return_value=exported,
                ), mock.patch(
                    "examples.musa.aoti_model_acceptance._run_runner", side_effect=run
                ), mock.patch(
                    "builtins.print"
                ):
                    if mutated is None:
                        acceptance_main()
                        report = json.loads((root / "result.json").read_text())
                        self.assertEqual(len(report["artifact_cases"][0]["outputs"]), 2)
                    else:
                        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                            acceptance_main()
                        self.assertFalse((root / "result.json").exists())

    @unittest.skipUnless(
        hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
    )
    def test_musa_snapshots_preserve_noncontiguous_tail_values(self) -> None:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                source = torch.arange(2000, device="musa", dtype=torch.float32)
                source = (source * 0.25 + 1).to(dtype).reshape(1000, 2).transpose(0, 1)
                path = Path(directory) / "expected.bin"
                snapshot = _write_tensor(source, path)
                expected = (
                    (torch.arange(2000, dtype=torch.float32) * 0.25 + 1)
                    .to(dtype)
                    .reshape(1000, 2)
                    .transpose(0, 1)
                    .contiguous()
                )
                source.zero_()
                torch.musa.synchronize()
                torch.testing.assert_close(snapshot, expected, atol=0, rtol=0)
                self.assertTrue(bool(snapshot[:, -16:].ne(0).all()))
                _record_tensor_artifact(snapshot, path)

    @unittest.skipUnless(
        hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
    )
    def test_musa_snapshot_pinned_staging_and_owned_result(self) -> None:
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int64):
            for shape in ((), (0,), (1001,), (2, 1999)):
                with self.subTest(dtype=dtype, shape=shape):
                    source = torch.full(shape, 3, dtype=dtype, device="musa")
                    if source.ndim == 2:
                        source = source.transpose(0, 1)
                    if source.is_floating_point():
                        source.requires_grad_()
                    with mock.patch("torch.empty", wraps=torch.empty) as allocate:
                        snapshot = _snapshot_tensor(source)
                    self.assertTrue(
                        any(
                            call.kwargs.get("pin_memory")
                            for call in allocate.call_args_list
                        )
                    )
                    self.assertEqual(snapshot.device.type, "cpu")
                    self.assertFalse(snapshot.requires_grad)
                    self.assertFalse(snapshot.is_pinned())
                    self.assertTrue(snapshot.is_contiguous())
                    with torch.no_grad():
                        source.zero_()
                    torch.musa.synchronize(source.device)
                    torch.testing.assert_close(
                        snapshot,
                        torch.full(source.shape, 3, dtype=dtype),
                        atol=0,
                        rtol=0,
                    )

    @unittest.skipUnless(
        hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
    )
    def test_musa_snapshot_pin_allocation_failure_is_not_hidden(self) -> None:
        source = torch.ones(1001, device="musa")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expected.bin"
            with mock.patch(
                "torch.empty", side_effect=RuntimeError("pinned allocation failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "pinned allocation failed"):
                    _write_tensor(source, path)
            self.assertFalse(path.exists())

    @unittest.skipUnless(
        hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
    )
    def test_musa_weight_storage_uses_pinned_chunks_and_preserves_views(self) -> None:
        from executorch.backends.cuda.cuda_weight_collector import _write_tensor_storage

        for size in (1001, 8 * 1024 * 1024 + 1001):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                source = torch.full((size,), 0x5A, dtype=torch.uint8, device="musa")
                view = source[5::2]
                identity = (
                    view.data_ptr(),
                    view.storage_offset(),
                    view.shape,
                    view.stride(),
                )
                path = Path(directory) / "weights.bin"
                with mock.patch(
                    "executorch.backends.cuda.cuda_weight_collector.snapshot_to_cpu",
                    wraps=_snapshot_tensor,
                ) as snapshot:
                    digest = _write_tensor_storage(view, str(path))
                payload = bytes([0x5A]) * size
                self.assertEqual(path.read_bytes(), payload)
                self.assertEqual(digest, hashlib.sha256(payload).digest())
                self.assertEqual(snapshot.call_count, 1 if size == 1001 else 2)
                self.assertEqual(
                    (view.data_ptr(), view.storage_offset(), view.shape, view.stride()),
                    identity,
                )
                source.zero_()
                torch.musa.synchronize()
                self.assertEqual(path.read_bytes(), payload)

    @unittest.skipUnless(
        hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
    )
    def test_musa_weight_storage_failure_removes_partial_file(self) -> None:
        from executorch.backends.cuda.cuda_weight_collector import _materialize_storage

        source = torch.ones(1001, device="musa")
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "executorch.backends.cuda.cuda_weight_collector.snapshot_to_cpu",
                side_effect=RuntimeError("pinned staging failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "pinned staging failed"):
                    _materialize_storage(source, directory)
            self.assertFalse(list(Path(directory).iterdir()))

    @unittest.skipUnless(
        hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
    )
    def test_prefill_reference_snapshots_keep_device_cache_unchanged(self) -> None:
        from examples.musa.aoti_prefill_decode_acceptance import _check_outputs

        logits = torch.tensor([[1.0, 3.0, 2.0]], device="musa")
        key = torch.ones((1, 2, 3, 8), device="musa")
        value = torch.full_like(key, 2)
        expected = (logits, key, value)
        pointers = tuple(tensor.data_ptr() for tensor in expected)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"output-{index}.bin" for index in range(3)]
            for tensor, path in zip(expected, paths):
                _write_tensor(tensor, path)
            with mock.patch(
                "examples.musa.aoti_prefill_decode_acceptance._snapshot_tensor",
                wraps=_snapshot_tensor,
            ) as snapshot:
                report = _check_outputs(paths, expected, 0)
            self.assertEqual(snapshot.call_count, 3)
            self.assertEqual(report["predicted_token"], [1])
            self.assertEqual(report["max_errors"], [0, 0, 0])
            self.assertEqual(tuple(tensor.data_ptr() for tensor in expected), pointers)
            self.assertTrue(all(tensor.device.type == "musa" for tensor in expected))


@unittest.skipUnless(
    hasattr(torch, "musa")
    and torch.musa.is_available()
    and os.environ.get("EXECUTORCH_MUSA_RUNNER"),
    "requires MUSA and EXECUTORCH_MUSA_RUNNER",
)
class TestMusaRunner(unittest.TestCase):

    def test_native_musa_device_metadata(self) -> None:
        from executorch.exir.schema import DeviceType, Tensor

        runner = Path(os.environ["EXECUTORCH_MUSA_RUNNER"])
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU()).eval().to("musa")
        example = torch.randn(2, 16, device="musa")
        exported = export(
            model, (example,), dynamic_shapes=({0: Dim("batch", min=1, max=3)},)
        )
        specs = [MusaBackend.generate_method_name_compile_spec("forward")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with _serialization_schema():
                manager = to_edge_transform_and_lower(
                    exported, partitioner=[MusaPartitioner(specs)]
                ).to_executorch()
                plan = manager.executorch_program.execution_plan[0]
                self.assertEqual(
                    [delegate.id for delegate in plan.delegates], ["MusaBackend"]
                )
                self.assertTrue(plan.non_const_buffer_device)
                self.assertEqual(
                    {entry.device_type for entry in plan.non_const_buffer_device},
                    {DeviceType.MUSA},
                )
                gpu_tensors = [
                    value.val
                    for value in plan.values
                    if isinstance(value.val, Tensor)
                    and value.val.extra_tensor_info is not None
                    and value.val.extra_tensor_info.device_type != DeviceType.CPU
                ]
                self.assertTrue(gpu_tensors)
                self.assertTrue(
                    all(
                        tensor.extra_tensor_info.device_type == DeviceType.MUSA
                        for tensor in gpu_tensors
                    )
                )
                pte = Path(save_pte_program(manager, "native_musa", directory))
            for batch in (1, 2, 3):
                with self.subTest(batch=batch), torch.no_grad():
                    value = torch.randn(batch, 16, device="musa")
                    expected = _snapshot_tensor(model(value)).flatten()
                    _write_input(value, root / "input.bin", (3, 16))
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
                        dtype=torch.float32,
                    )
                    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

            payload = pte.read_bytes()
            self.assertEqual(payload.count(b"MusaBackend"), 1)
            foreign = root / "cuda_backend.pte"
            foreign.write_bytes(payload.replace(b"MusaBackend", b"CudaBackend"))
            with self.assertRaisesRegex(RuntimeError, "executor_runner failed"):
                _run_runner(
                    runner,
                    foreign,
                    root / "aoti_musa_blob.ptd",
                    root / "input.bin",
                    root / "foreign-output",
                    (3, 16),
                )

    def test_legacy_cuda_tagged_musa_metadata(self) -> None:
        from executorch.backends.cuda.cuda_weight_collector import (
            decode_cuda_aoti_metadata,
        )
        from executorch.exir.backend.compile_spec_schema import CompileSpec
        from executorch.exir.schema import DeviceType

        runner = Path(os.environ["EXECUTORCH_MUSA_RUNNER"])
        model = nn.Linear(8, 4).eval().to("musa")
        example = torch.randn(2, 8, device="musa")
        with torch.no_grad():
            expected = _snapshot_tensor(model(example)).flatten()
        exported = export(model, (example,))
        specs = [
            MusaBackend.generate_method_name_compile_spec("forward"),
            CompileSpec("target_device", b"cuda:0"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with _serialization_schema(), mock.patch(
                "executorch.backends.musa.musa_backend._aoti_device_type_for_weight",
                side_effect=lambda tensor: 0 if tensor.device.type == "cpu" else 1,
            ):
                manager = to_edge_transform_and_lower(
                    exported, partitioner=[MusaPartitioner(specs)]
                ).to_executorch()
                program = manager.executorch_program
                plan = program.execution_plan[0]
                self.assertEqual(
                    [delegate.id for delegate in plan.delegates], ["MusaBackend"]
                )
                self.assertEqual(
                    {entry.device_type for entry in plan.non_const_buffer_device},
                    {DeviceType.CUDA},
                )
                metadata = decode_cuda_aoti_metadata(
                    program.backend_delegate_data[
                        plan.delegates[0].processed.index
                    ].data
                )
                self.assertTrue(metadata.entries)
                self.assertEqual({entry.device_type for entry in metadata.entries}, {1})
                pte = Path(save_pte_program(manager, "legacy_musa", directory))
            _write_input(example, root / "input.bin", None)
            _run_runner(
                runner,
                pte,
                root / "aoti_musa_blob.ptd",
                root / "input.bin",
                root / "output",
            )
            actual = torch.frombuffer(
                bytearray((root / "output-0.bin").read_bytes()), dtype=torch.float32
            )
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_depthwise_convolution_algorithm_rounding(self) -> None:
        runner = Path(os.environ["EXECUTORCH_MUSA_RUNNER"])
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(123)
                model = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
                model = model.eval().to(device="musa", dtype=dtype)
                example = torch.randn(2, 32, 32, 32, device="musa", dtype=dtype)
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
                        pte = Path(save_pte_program(manager, "depthwise", directory))
                    for batch in (1, 2, 3):
                        with self.subTest(
                            batch=batch
                        ), torch.no_grad(), torch.backends.mudnn.flags(
                            allow_tf32=False
                        ):
                            value = torch.randn(
                                batch, 32, 32, 32, device="musa", dtype=dtype
                            )
                            expected = _snapshot_tensor(model(value)).flatten()
                            self.assertGreater(expected.abs().max().item(), 0.1)
                            _write_input(value, root / "input.bin", (3, 32, 32, 32))
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
                            expected = _snapshot_tensor(model(value)).flatten()
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
                        expected = _snapshot_tensor(model(example)).flatten()
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
