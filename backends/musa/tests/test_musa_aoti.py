# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest
from unittest import mock

import torch
from torch import nn
from torch.export import Dim, export

from executorch.backends.cuda.cuda_weight_collector import (
    decode_cuda_aoti_metadata,
)
from executorch.backends.musa import (
    MusaBackend,
    MusaPartitioner,
    export_with_musa_pooling_shapes,
    export_with_musa_sdpa_decomposition,
)
from executorch.backends.musa.musa_backend import (
    _aoti_device_type_for_weight,
    _decompose_scaled_dot_product_attention,
    _normalize_musa_assert_devices,
)
from executorch.exir._serialize._cord import FileBackedData
from examples.musa.aoti_kv_cache_acceptance import (
    GrowingCacheDecoder,
    RollingCacheDecoder,
)
from examples.musa.aoti_model_acceptance import (
    BilinearUNet,
    BroadcastBmm,
    DeepLabV3MobileNet,
    DilatedDecoder,
    SimpleAudioDecoder,
    SimpleVideoDecoder,
    TinyUNet,
    TransposedBmm,
    VitLayerTokens,
    _torchvision_efficientnet_b0,
    _torchvision_vit_b_16,
)
from examples.musa.aoti_prefill_decode_acceptance import PromptPrefill


class TestMusaPoolCodegen(unittest.TestCase):
    def _emit(self, device, offset, replacements, fail=False):
        from types import SimpleNamespace

        import sympy
        from torch._inductor.codegen import cpp_wrapper_cpu, memory_planning
        from torch._inductor.sizevars import SizeVarAllocator
        from torch._inductor.virtualized import V as virtualized
        from executorch.backends.musa.musa_backend import _musa_aoti_compile_context

        wrapper_type = cpp_wrapper_cpu.CppWrapperCpu
        original_method = wrapper_type.codegen_alloc_from_pool
        original_printer = cpp_wrapper_cpu.cexpr
        original_symbolic_size = memory_planning.TemporalSplit.get_symbolic_size
        sizevars = SizeVarAllocator()
        sizevars.replacements.update(replacements)
        graph = SimpleNamespace(sizevars=sizevars, device_types={device})
        wrapper = mock.Mock()
        wrapper.get_codegened_graph.return_value = graph
        probe = sympy.Symbol("unrelated", integer=True) + 1

        def generate(wrapper, name, received_offset, dtype, shape, stride):
            self.assertIs(received_offset, offset)
            rendered = cpp_wrapper_cpu.cexpr(received_offset)
            cpp_wrapper_cpu.cexpr(probe)
            if fail:
                raise RuntimeError("pool codegen failed")
            return rendered

        try:
            with mock.patch.object(
                wrapper_type, "codegen_alloc_from_pool", generate
            ), virtualized.set_graph_handler(graph), mock.patch.object(
                sizevars, "simplify", wraps=sizevars.simplify
            ) as simplify:
                with _musa_aoti_compile_context():
                    rendered = wrapper_type.codegen_alloc_from_pool(
                        wrapper, "pool0", offset, torch.float32, (), ()
                    )
                return (
                    rendered,
                    [call.args[0] for call in simplify.call_args_list],
                    probe,
                )
        finally:
            self.assertIs(wrapper_type.codegen_alloc_from_pool, original_method)
            self.assertIs(cpp_wrapper_cpu.cexpr, original_printer)
            self.assertIs(
                memory_planning.TemporalSplit.get_symbolic_size, original_symbolic_size
            )

    def test_pool_size_max_cache_and_device_isolation(self):
        from types import SimpleNamespace

        import sympy
        from torch._inductor.codegen.memory_planning import TemporalSplit
        from torch._inductor.virtualized import V as virtualized
        from torch.utils._sympy.functions import Max
        from executorch.backends.musa.musa_backend import _musa_aoti_compile_context

        height, width = sympy.symbols("height width", integer=True, positive=True)
        original = TemporalSplit.get_symbolic_size
        for device in ("musa", "cpu", "cuda"):
            with self.subTest(device=device):
                first = mock.Mock()
                second = mock.Mock()
                first.get_symbolic_size.return_value = 64 * height
                second.get_symbolic_size.return_value = 128 * width
                split = TemporalSplit([first, second])
                with virtualized.set_graph_handler(
                    SimpleNamespace(device_types={device})
                ), _musa_aoti_compile_context():
                    expression = split.get_symbolic_size()
                    self.assertIsInstance(
                        expression, Max if device == "musa" else sympy.Max
                    )
                    self.assertIs(split.get_symbolic_size(), expression)
                    first.get_symbolic_size.assert_called_once()
                    second.get_symbolic_size.assert_called_once()
                    first.get_symbolic_size.return_value = 256 * height
                    split.clear_cache()
                    updated = split.get_symbolic_size()
                    self.assertEqual(first.get_symbolic_size.call_count, 2)
                    self.assertEqual(int(updated.subs({height: 3, width: 2})), 768)
                    self.assertEqual(TemporalSplit([]).get_symbolic_size(), 0)
                    self.assertEqual(
                        TemporalSplit([first]).get_symbolic_size(), 256 * height
                    )
                self.assertIs(TemporalSplit.get_symbolic_size, original)

    def test_pool_max_preserves_nested_size_values(self):
        from types import SimpleNamespace

        import sympy
        from torch._inductor.codegen.memory_planning import TemporalSplit
        from torch._inductor.utils import align
        from torch._inductor.virtualized import V as virtualized
        from executorch.backends.musa.musa_backend import _musa_aoti_compile_context

        batch, height, width = sympy.symbols(
            "batch height width", integer=True, positive=True
        )
        sizes = [
            64 * batch * (height + 1),
            128 * (width + 1),
            align(32 * batch * (height + 1) * (width + 1)) + 64 * batch,
        ]
        blocks = []
        for size in sizes:
            block = mock.Mock()
            block.get_symbolic_size.return_value = size
            blocks.append(block)
        inner = TemporalSplit(blocks[:2])
        outer = TemporalSplit([inner, blocks[2]])
        with virtualized.set_graph_handler(
            SimpleNamespace(device_types={"musa"})
        ), _musa_aoti_compile_context():
            expression = outer.get_symbolic_size()
        for batch_value in (1, 2, 3):
            for height_value, width_value in ((48, 54), (64, 64), (79, 71), (80, 80)):
                values = {batch: batch_value, height: height_value, width: width_value}
                self.assertEqual(
                    int(expression.subs(values)),
                    max(int(size.subs(values)) for size in sizes),
                )

    def test_pool_offset_preserves_replacements_without_expansion(self):
        import sympy
        from torch._inductor.codegen.cpp_utils import cexpr
        from torch._inductor.utils import align

        batch, height, width, alias = sympy.symbols(
            "batch height width alias", integer=True, positive=True
        )
        offset = align(sympy.Max(64 * (alias + 1) * (height + 1), 128 * batch * width))
        replacements = {alias: batch + 1}
        rendered, calls, probe = self._emit("musa", offset, replacements)
        self.assertEqual(rendered, cexpr(offset.xreplace(replacements), simplify=False))
        self.assertNotIn("alias", rendered)
        self.assertEqual(calls, [probe])

    def test_other_devices_keep_original_pool_printing(self):
        import sympy

        extent = sympy.Symbol("extent", integer=True, positive=True)
        offset = (extent + 1) * (extent + 2)
        for device in ("cpu", "cuda"):
            with self.subTest(device=device):
                _, calls, probe = self._emit(device, offset, {})
                self.assertEqual(calls, [offset, probe])

    def test_pool_printer_restores_after_failure(self):
        import sympy

        with self.assertRaisesRegex(RuntimeError, "pool codegen failed"):
            self._emit("musa", sympy.Symbol("offset", integer=True), {}, fail=True)

    def test_factored_pool_offsets_match_compiled_cpp(self):
        import shutil
        import subprocess
        import tempfile
        from pathlib import Path

        import sympy
        from torch._inductor.utils import align
        from torch.utils._sympy.functions import Max

        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("requires a C++ compiler")
        batch, height, width, alias = sympy.symbols(
            "batch height width alias", integer=True, positive=True
        )
        offset = align(
            Max(
                64 * (alias + 1) * (height + 1),
                128 * batch * width,
                align(32 * batch * (height + 1) * (width + 1)) + 64 * batch,
            )
        )
        replacements = {alias: batch + 1}
        rendered, calls, probe = self._emit("musa", offset, replacements)
        self.assertEqual(calls, [probe])
        cases = [
            (batch_value, height_value, width_value)
            for batch_value in (1, 2, 3)
            for height_value in (48, 54, 64, 71, 79, 80)
            for width_value in (48, 54, 64, 71, 79, 80)
        ]
        expected = [
            int(
                offset.xreplace(replacements).subs(
                    {batch: batch_value, height: height_value, width: width_value}
                )
            )
            for batch_value, height_value, width_value in cases
        ]
        source = (
            """#include <algorithm>
#include <cstdint>
#include <iostream>
int64_t align(int64_t value) { return (value + 63) & -64; }
int main() {
    int64_t batch, height, width;
    while (std::cin >> batch >> height >> width) {
        std::cout << """
            + rendered
            + """ << '\\n';
    }
}
"""
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "offset.cpp"
            path.write_text(source)
            compiled = subprocess.run(
                [compiler, "-std=c++17", str(path), "-o", str(root / "offset")],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run(
                [str(root / "offset")],
                input="\n".join(" ".join(map(str, case)) for case in cases),
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([int(value) for value in result.stdout.split()], expected)


class TestMusaAotiFallbacks(unittest.TestCase):

    def test_runtime_link_libraries_are_scoped_and_restored(self) -> None:
        from torch._inductor.cpp_builder import CppTorchDeviceOptions
        from executorch.backends.musa.musa_backend import _musa_aoti_compile_context

        original = CppTorchDeviceOptions.get_libraries
        libraries = ["torch", "torch_cpu", "c10", "omp", "musa", "musa_python"]
        for aot_mode, compile_only, device in (
            (True, False, "MUSA"),
            (False, False, "MUSA"),
            (True, True, "MUSA"),
            (True, False, "CUDA"),
            (True, False, "CPU"),
        ):
            options = object.__new__(CppTorchDeviceOptions)
            options._libraries = list(libraries)
            options._aot_mode = aot_mode
            options._compile_only = compile_only
            options._definitions = [f" USE_{device}"]
            with self.subTest(
                aot_mode=aot_mode, compile_only=compile_only, device=device
            ):
                with _musa_aoti_compile_context():
                    expected = (
                        ["omp", "musa", "musart"]
                        if aot_mode and not compile_only and device == "MUSA"
                        else libraries
                    )
                    self.assertEqual(options.get_libraries(), expected)
                    self.assertEqual(options._libraries, libraries)
                self.assertIs(CppTorchDeviceOptions.get_libraries, original)
        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            with _musa_aoti_compile_context():
                raise RuntimeError("compile failed")
        self.assertIs(CppTorchDeviceOptions.get_libraries, original)

    def test_native_device_compile_spec_and_weight_mapping(self) -> None:
        from executorch.exir.passes.propagate_device_pass import (
            _parse_device_spec_value,
        )
        from executorch.exir.schema import DeviceType

        partitioner = MusaPartitioner([])
        self.assertEqual(_parse_device_spec_value(b"musa:12"), (DeviceType.MUSA, 12))
        self.assertEqual(_aoti_device_type_for_weight(torch.ones(1)), 0)
        for name, expected in (("musa", 2), ("cuda", None)):
            tensor = mock.Mock()
            tensor.device = mock.Mock(type=name)
            if expected is None:
                with self.assertRaisesRegex(
                    RuntimeError, "Unsupported AOTI constant device"
                ):
                    _aoti_device_type_for_weight(tensor)
            else:
                self.assertEqual(_aoti_device_type_for_weight(tensor), expected)
        self.assertEqual(partitioner.delegation_spec.backend_id, "MusaBackend")
        targets = [
            spec.value
            for spec in partitioner.delegation_spec.compile_specs
            if spec.key == "target_device"
        ]
        self.assertEqual(targets, [b"musa:0"])

    def test_pooling_shape_division_and_restoration(self) -> None:
        from torch import _meta_registrations

        original = _meta_registrations.div_rtn
        for numerator in range(-9, 10):
            for denominator in (-4, -2, -1, 1, 2, 4):
                self.assertEqual(
                    original(numerator, denominator),
                    operator.floordiv(numerator, denominator),
                )

        def check_patch(*args, **kwargs):
            self.assertIs(_meta_registrations.div_rtn, operator.floordiv)
            raise RuntimeError("export failed")

        with mock.patch.object(torch.export, "export", side_effect=check_patch):
            with self.assertRaisesRegex(RuntimeError, "export failed"):
                export_with_musa_pooling_shapes(nn.Identity(), ())
        self.assertIs(_meta_registrations.div_rtn, original)

    def test_registers_native_slimtensor_fallback(self) -> None:
        fallbacks = MusaBackend.get_supported_fallback_kernels()
        self.assertIn("aoti_torch_musa_addmm_out", fallbacks)
        self.assertIn("aoti_torch_musa_mm_out", fallbacks)
        self.assertIn("aoti_torch_musa_bmm_out", fallbacks)
        self.assertIn("aoti_torch_musa_convolution", fallbacks)
        self.assertIn("aoti_torch_musa_convolution_out", fallbacks)
        self.assertIn("aoti_torch_musa_batch_norm", fallbacks)
        self.assertIn("aoti_torch_musa_batch_norm_out", fallbacks)
        self.assertIn("aoti_torch_musa_hardswish", fallbacks)
        self.assertIn("aoti_torch_musa_hardsigmoid", fallbacks)
        self.assertIn("aoti_torch_musa_hardswish_out", fallbacks)
        self.assertIn("aoti_torch_musa_hardsigmoid_out", fallbacks)


@unittest.skipUnless(torch.musa.is_available(), "MUSA is not available")
class TestMusaAoti(unittest.TestCase):

    def test_convolution_out_memory_planning_compile(self) -> None:
        for convolution, shape, with_bias in (
            (nn.Conv1d, (2, 2, 8), False),
            (nn.Conv2d, (2, 2, 8, 8), True),
            (nn.Conv3d, (2, 2, 5, 5, 5), False),
            (nn.ConvTranspose1d, (2, 2, 8), True),
            (nn.ConvTranspose2d, (2, 2, 8, 8), False),
            (nn.ConvTranspose3d, (2, 2, 5, 5, 5), True),
            (nn.Conv2d, (2, 8, 8), False),
        ):
            with self.subTest(convolution=convolution.__name__, bias=with_bias):
                module = (
                    convolution(2, 4, 3, stride=2, padding=1, groups=2, bias=with_bias)
                    .eval()
                    .to("musa", dtype=torch.float16)
                )
                example = torch.randn(*shape, device="musa", dtype=torch.float16)
                with torch._inductor.config.patch(
                    memory_planning=True, memory_pool="combined"
                ):
                    result = MusaBackend.preprocess(
                        export(module, (example,)),
                        [MusaBackend.generate_method_name_compile_spec("forward")],
                    )
                payloads = [
                    buffer.prefix(len(buffer))
                    if isinstance(buffer, FileBackedData)
                    else bytes(buffer)
                    for buffer in result.data_store_output.buffers
                ]
                for symbol in (
                    b"aoti_torch_musa_convolution_out\0",
                    b"aoti_torch__alloc_from_pool\0",
                ):
                    self.assertTrue(
                        any(symbol in payload for payload in payloads), symbol
                    )
                self.assertFalse(
                    any(
                        b"aoti_torch_musa_convolution\0" in payload
                        for payload in payloads
                    )
                )

    def test_batch_norm_out_memory_planning_compile(self) -> None:
        module = nn.BatchNorm2d(4).eval().to("musa", dtype=torch.bfloat16)
        example = torch.randn(2, 4, 8, 8, device="musa", dtype=torch.bfloat16)
        with torch._inductor.config.patch(memory_planning=True, memory_pool="combined"):
            result = MusaBackend.preprocess(
                export(module, (example,)),
                [MusaBackend.generate_method_name_compile_spec("forward")],
            )
        for symbol in (
            b"aoti_torch_musa_batch_norm_out\0",
            b"aoti_torch__alloc_from_pool\0",
        ):
            self.assertTrue(
                any(
                    symbol in bytes(buffer)
                    for buffer in result.data_store_output.buffers
                ),
                symbol,
            )

    def test_batch_norm_out_python_dispatch(self) -> None:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.no_grad():
                value = torch.randn(2, 4, 3, 3, device="musa", dtype=dtype)
                weight = torch.randn(4, device="musa", dtype=dtype)
                bias = torch.randn_like(weight)
                mean = torch.randn_like(weight)
                variance = torch.rand_like(weight) + 1
                output = torch.empty_like(value)
                expected = torch.ops.executorch_musa.batch_norm.default(
                    value, weight, bias, mean, variance, 1e-5
                )
                actual = torch.ops.executorch_musa.batch_norm.out(
                    value, weight, bias, mean, variance, 1e-5, out=output
                )
                self.assertEqual(actual.data_ptr(), output.data_ptr())
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                with self.assertRaises(RuntimeError):
                    torch.ops.executorch_musa.batch_norm.out(
                        value,
                        weight,
                        bias,
                        mean,
                        variance,
                        1e-5,
                        out=torch.empty(1, device="musa", dtype=dtype),
                    )

    def test_dynamic_pooling_spatial_shapes(self) -> None:
        from torch import _meta_registrations

        model = (
            nn.Sequential(
                nn.Conv2d(3, 4, 7, stride=2, padding=3),
                nn.MaxPool2d(3, stride=2, padding=1),
            )
            .eval()
            .to("musa")
        )
        original = _meta_registrations.div_rtn
        example = torch.randn(2, 3, 64, 64, device="musa")
        exported = export_with_musa_pooling_shapes(
            model,
            (example,),
            dynamic_shapes=(
                {
                    0: Dim("batch", min=1, max=3),
                    2: Dim("height", min=48, max=80),
                    3: Dim("width", min=48, max=80),
                },
            ),
        )
        self.assertIs(_meta_registrations.div_rtn, original)
        with torch.no_grad():
            for shape in ((1, 3, 48, 54), (3, 3, 80, 80), (2, 3, 79, 71)):
                value = torch.randn(shape, device="musa")
                torch.testing.assert_close(exported.module()(value), model(value))
        result = MusaBackend.preprocess(
            exported, [MusaBackend.generate_method_name_compile_spec("forward")]
        )
        self.assertGreater(len(result.processed_bytes), 0)

    def test_activation_out_memory_planning_compile(self) -> None:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                example = torch.randn(2, 8, 4, device="musa", dtype=dtype)
                exported = export(
                    nn.Sequential(nn.Hardswish(), nn.Hardsigmoid()).eval(),
                    (example,),
                    dynamic_shapes=({0: Dim("batch", min=1, max=3)},),
                )
                with torch._inductor.config.patch(
                    memory_planning=True, memory_pool="combined"
                ):
                    result = MusaBackend.preprocess(
                        exported,
                        [MusaBackend.generate_method_name_compile_spec("forward")],
                    )
                payloads = [
                    (
                        buffer.prefix(len(buffer))
                        if isinstance(buffer, FileBackedData)
                        else bytes(buffer)
                    )
                    for buffer in result.data_store_output.buffers
                ]
                for symbol in (
                    b"aoti_torch_musa_hardswish_out\0",
                    b"aoti_torch_musa_hardsigmoid_out\0",
                    b"aoti_torch__alloc_from_pool\0",
                ):
                    self.assertTrue(
                        any(symbol in payload for payload in payloads), symbol
                    )
                for symbol in (
                    b"aoti_torch_musa_hardswish\0",
                    b"aoti_torch_musa_hardsigmoid\0",
                ):
                    self.assertFalse(
                        any(symbol in payload for payload in payloads), symbol
                    )

    def test_hardswish_aoti_compile(self) -> None:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                example = torch.randn(2, 8, 4, device="musa", dtype=dtype)
                exported = export(
                    nn.Hardswish().eval(),
                    (example,),
                    dynamic_shapes=({0: Dim("batch", min=1, max=3)},),
                )
                decomposed = exported.run_decompositions(
                    MusaBackend.get_decomposition_table()
                )
                self.assertTrue(
                    any(
                        node.target == torch.ops.executorch_musa.hardswish.default
                        for node in decomposed.graph.nodes
                    )
                )
                result = MusaBackend.preprocess(
                    exported,
                    [MusaBackend.generate_method_name_compile_spec("forward")],
                )
                self.assertGreater(len(result.processed_bytes), 0)
                self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_broadcast_bmm_dynamic_batch_aoti_compile(self) -> None:
        module = BroadcastBmm().eval().to("musa")
        example = torch.randn(2, 4, 16, device="musa")
        exported = export(
            module,
            (example,),
            dynamic_shapes=({0: Dim("batch", min=1, max=3)},),
        )
        result = MusaBackend.preprocess(
            exported, [MusaBackend.generate_method_name_compile_spec("forward")]
        )

        self.assertEqual(len(exported.range_constraints), 1)
        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_transposed_bmm_aoti_compile(self) -> None:
        example = torch.randn(2, 4, 16, device="musa")
        exported = export(
            TransposedBmm().eval(),
            (example,),
            dynamic_shapes=({0: Dim("batch", min=1, max=3)},),
        )
        result = MusaBackend.preprocess(
            exported, [MusaBackend.generate_method_name_compile_spec("forward")]
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_normalizes_unindexed_musa_device_assertion(self) -> None:
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        tensor.meta["val"] = torch.empty(1, device="musa")
        assertion = graph.call_function(
            torch.ops.aten._assert_tensor_metadata.default,
            args=(tensor,),
            kwargs={"device": "musa"},
        )
        graph.output(tensor)
        module = torch.fx.GraphModule(nn.Module(), graph)

        _normalize_musa_assert_devices(module)

        self.assertEqual(assertion.kwargs["device"], torch.device("musa:0"))

    def test_nested_input_aoti_compile(self) -> None:
        class NestedInput(nn.Module):
            def forward(self, tensor: torch.Tensor, options: dict) -> torch.Tensor:
                return tensor + options["other"]

        example = torch.randn(1, 8, device="musa")
        exported = export(NestedInput().eval(), (example, {"other": example}))
        result = MusaBackend.preprocess(
            exported, [MusaBackend.generate_method_name_compile_spec("forward")]
        )

        self.assertGreater(len(result.processed_bytes), 0)

    def test_prompt_prefill_aoti_compile(self) -> None:
        decoder = GrowingCacheDecoder().eval().to("musa")
        module = PromptPrefill(decoder).eval()
        tokens = torch.tensor([[11, 7, 22]], device="musa", dtype=torch.int64)
        result = MusaBackend.preprocess(
            export(
                module,
                (tokens,),
                dynamic_shapes=({1: Dim("prompt", min=2, max=4)},),
            ),
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_growing_kv_cache_aoti_compile(self) -> None:
        module = GrowingCacheDecoder().eval().to("musa")
        example = (
            torch.tensor([[5]], device="musa", dtype=torch.int64),
            torch.randn(1, 2, 3, 8, device="musa"),
            torch.randn(1, 2, 3, 8, device="musa"),
        )
        dynamic_shapes = (
            {},
            {2: Dim("past", min=2, max=4)},
            {2: Dim("past", min=2, max=4)},
        )
        result = MusaBackend.preprocess(
            export(module, example, dynamic_shapes=dynamic_shapes),
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_rolling_kv_cache_aoti_compile(self) -> None:
        module = RollingCacheDecoder().eval().to("musa")
        example = (
            torch.tensor([[5]], device="musa", dtype=torch.int64),
            torch.randn(1, 2, 4, 8, device="musa"),
            torch.randn(1, 2, 4, 8, device="musa"),
        )
        result = MusaBackend.preprocess(
            export(module, example),
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_pointwise_aoti_compile(self) -> None:
        class AddOne(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        example = torch.randn(2, 8, device="musa")
        exported = export(AddOne().eval(), (example,))
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_native_batch_norm_inference_aoti_compile(self) -> None:
        class NativeBatchNorm(nn.Module):
            def __init__(self, variant: str) -> None:
                super().__init__()
                self.variant = variant
                self.weight = nn.Parameter(torch.ones(4))
                self.bias = nn.Parameter(torch.zeros(4))
                self.register_buffer("running_mean", torch.zeros(4))
                self.register_buffer("running_var", torch.ones(4))

            def forward(self, input_tensor: torch.Tensor):
                if self.variant == "legit":
                    result = torch.ops.aten._native_batch_norm_legit.default(
                        input_tensor,
                        self.weight,
                        self.bias,
                        self.running_mean,
                        self.running_var,
                        False,
                        0.1,
                        1e-5,
                    )
                elif self.variant == "no_training":
                    result = (
                        torch.ops.aten._native_batch_norm_legit_no_training.default(
                            input_tensor,
                            self.weight,
                            self.bias,
                            self.running_mean,
                            self.running_var,
                            0.1,
                            1e-5,
                        )
                    )
                elif self.variant == "native":
                    result = torch.ops.aten.native_batch_norm.default(
                        input_tensor,
                        self.weight,
                        self.bias,
                        self.running_mean,
                        self.running_var,
                        False,
                        0.1,
                        1e-5,
                    )
                elif self.variant == "no_update":
                    result = torch.ops.aten._batch_norm_no_update.default(
                        input_tensor,
                        self.weight,
                        self.bias,
                        self.running_mean,
                        self.running_var,
                        0.1,
                        1e-5,
                    )
                else:
                    result = torch.ops.aten._native_batch_norm_legit_functional.default(
                        input_tensor,
                        self.weight,
                        self.bias,
                        self.running_mean,
                        self.running_var,
                        False,
                        0.1,
                        1e-5,
                    )
                return result

        example = torch.randn(2, 4, 8, 8, device="musa")
        for variant in (
            "native",
            "legit",
            "functional",
            "no_training",
            "no_update",
        ):
            with self.subTest(variant=variant):
                exported = export(
                    NativeBatchNorm(variant).eval().to("musa"), (example,)
                )
                result = MusaBackend.preprocess(
                    exported,
                    [MusaBackend.generate_method_name_compile_spec("forward")],
                )
                self.assertGreater(len(result.processed_bytes), 0)
                self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_weighted_linear_aoti_compile(self) -> None:
        class WeightedLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(8, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        example = torch.randn(2, 8, device="musa")
        module = WeightedLinear().eval().to("musa")
        exported = export(module, (example,))
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        metadata = decode_cuda_aoti_metadata(result.processed_bytes)
        self.assertGreaterEqual(len(metadata.entries), 1)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

        entries = {entry.fqn: entry for entry in metadata.entries}
        self.assertIn("linear.weight", entries)
        self.assertEqual(entries["linear.weight"].sizes, (4, 8))
        self.assertEqual(entries["linear.weight"].device_type, 2)
        self.assertTrue(
            entries["linear.weight"].storage_key.startswith("musa_fqn_weight:musa:")
        )

    def test_dynamic_embedding_aoti_compile(self) -> None:
        class EmbeddingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(64, 16)
                self.classifier = nn.Linear(16, 8)

            def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
                return self.classifier(self.embedding(token_ids).mean(dim=1))

        example = torch.randint(64, (2, 4), device="musa", dtype=torch.int64)
        module = EmbeddingModel().eval().to("musa")
        exported = export(
            module, (example,), dynamic_shapes=({0: Dim("batch", min=1, max=3)},)
        )
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_conv2d_aoti_compile(self) -> None:
        class Conv2dModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        example = torch.randn(1, 3, 8, 8, device="musa")
        module = Conv2dModel().eval().to("musa")
        exported = export(module, (example,))
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_batch_norm2d_aoti_compile(self) -> None:
        module = nn.BatchNorm2d(4).eval().to("musa", dtype=torch.bfloat16)
        example = torch.randn(
            1, 4, 8, 8, device="musa", dtype=torch.bfloat16
        )
        result = MusaBackend.preprocess(
            export(module, (example,)),
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_batch_norm2d_mixed_dtype_aoti_compile(self) -> None:
        module = nn.BatchNorm2d(4).eval().to("musa", dtype=torch.float32)
        example = torch.randn(
            1, 4, 8, 8, device="musa", dtype=torch.bfloat16
        )
        result = MusaBackend.preprocess(
            export(module, (example,)),
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_batch_norm1d_matrix_aoti_compile(self) -> None:
        module = nn.BatchNorm1d(4).eval().to("musa", dtype=torch.bfloat16)
        example = torch.randn(2, 4, device="musa", dtype=torch.bfloat16)
        result = MusaBackend.preprocess(
            export(module, (example,)),
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_batch_norm2d_without_affine_aoti_compile(self) -> None:
        module = nn.BatchNorm2d(4, affine=False).eval().to(
            "musa", dtype=torch.bfloat16
        )
        example = torch.randn(
            1, 4, 8, 8, device="musa", dtype=torch.bfloat16
        )
        result = MusaBackend.preprocess(
            export(module, (example,)),
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_direct_batch_norm_aoti_compile(self) -> None:
        class DirectBatchNorm(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
                self.bias = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
                self.register_buffer(
                    "running_mean", torch.zeros(4, dtype=torch.bfloat16)
                )
                self.register_buffer(
                    "running_var", torch.ones(4, dtype=torch.bfloat16)
                )

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return torch.ops.aten.batch_norm.default(
                    input,
                    self.weight,
                    self.bias,
                    self.running_mean,
                    self.running_var,
                    False,
                    0.1,
                    1e-5,
                    True,
                )

        module = DirectBatchNorm().eval().to("musa")
        example = torch.randn(1, 4, 8, 8, device="musa", dtype=torch.bfloat16)
        result = MusaBackend.preprocess(
            export(module, (example,)),
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_conv3d_aoti_compile(self) -> None:
        class Conv3dModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv3d(2, 3, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        example = torch.randn(1, 2, 4, 4, 4, device="musa")
        module = Conv3dModel().eval().to("musa")
        exported = export(module, (example,))
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_conv_transpose2d_aoti_compile(self) -> None:
        class ConvTranspose2dModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.ConvTranspose2d(
                    2, 3, 3, stride=2, padding=1, output_padding=1
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        example = torch.randn(1, 2, 4, 4, device="musa")
        module = ConvTranspose2dModel().eval().to("musa")
        exported = export(module, (example,))
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_conv_transpose1d_aoti_compile(self) -> None:
        class ConvTranspose1dModel(nn.Module):
            def __init__(self, groups: int) -> None:
                super().__init__()
                self.conv = nn.ConvTranspose1d(
                    2, 4, 3, stride=2, padding=1, output_padding=1, groups=groups
                )

            def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
                return self.conv(input_tensor)

        example = torch.randn(1, 2, 5, device="musa")
        for groups in (1, 2):
            with self.subTest(groups=groups):
                module = ConvTranspose1dModel(groups).eval().to("musa")
                exported = export(module, (example,))
                result = MusaBackend.preprocess(
                    exported,
                    [MusaBackend.generate_method_name_compile_spec("forward")],
                )

                self.assertGreater(len(result.processed_bytes), 0)
                self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_conv_transpose3d_aoti_compile(self) -> None:
        class ConvTranspose3dModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.ConvTranspose3d(
                    2, 3, 3, stride=2, padding=1, output_padding=1
                )

            def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
                return self.conv(input_tensor)

        example = torch.randn(1, 2, 2, 3, 4, device="musa")
        module = ConvTranspose3dModel().eval().to("musa")
        exported = export(module, (example,))
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_dilated_conv_transpose_aoti_compile(self) -> None:
        for dimensions, shape in (
            (1, (1, 2, 5)),
            (2, (1, 2, 5, 4)),
            (3, (1, 2, 5, 4, 3)),
        ):
            with self.subTest(dimensions=dimensions):
                module = DilatedDecoder(dimensions).eval().to("musa")
                exported = export(module, (torch.randn(shape, device="musa"),))
                result = MusaBackend.preprocess(
                    exported,
                    [MusaBackend.generate_method_name_compile_spec("forward")],
                )

                self.assertGreater(len(result.processed_bytes), 0)
                self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_dynamic_conv_transpose_batch_aoti_compile(self) -> None:
        for model_type, shape in (
            (SimpleAudioDecoder, (2, 2, 5)),
            (SimpleVideoDecoder, (2, 1, 2, 3, 4)),
        ):
            with self.subTest(model=model_type.__name__):
                module = model_type().eval().to(device="musa", dtype=torch.bfloat16)
                exported = export(
                    module,
                    (torch.randn(shape, device="musa", dtype=torch.bfloat16),),
                    dynamic_shapes=({0: Dim("batch", min=1, max=3)},),
                )
                result = MusaBackend.preprocess(
                    exported,
                    [MusaBackend.generate_method_name_compile_spec("forward")],
                )

                self.assertGreater(len(result.processed_bytes), 0)
                self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_tiny_unet_aoti_compile(self) -> None:
        for dtype, dynamic_shapes in (
            (torch.float32, None),
            (torch.bfloat16, ({0: Dim("batch", min=1, max=3)},)),
        ):
            with self.subTest(dtype=dtype):
                module = TinyUNet().eval().to(device="musa", dtype=dtype)
                batch = 2 if dynamic_shapes is not None else 1
                exported = export(
                    module,
                    (torch.randn(batch, 3, 32, 32, device="musa", dtype=dtype),),
                    dynamic_shapes=dynamic_shapes,
                )
                result = MusaBackend.preprocess(
                    exported,
                    [MusaBackend.generate_method_name_compile_spec("forward")],
                )

                self.assertGreater(len(result.processed_bytes), 0)
                self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_bilinear_unet_dynamic_spatial_aoti_compile(self) -> None:
        module = BilinearUNet().eval().to(device="musa", dtype=torch.bfloat16)
        exported = export(
            module,
            (torch.randn(1, 3, 32, 40, device="musa", dtype=torch.bfloat16),),
            dynamic_shapes=(
                {2: Dim("height", min=24, max=48), 3: Dim("width", min=24, max=48)},
            ),
        )
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_deeplabv3_mobilenet_combined_dynamic_aoti_compile(self) -> None:
        module = DeepLabV3MobileNet().eval().to("musa")
        exported = export(
            module,
            (torch.randn(2, 3, 64, 64, device="musa"),),
            dynamic_shapes=(
                {
                    0: Dim("batch", min=1, max=3),
                    2: Dim("height", min=48, max=80),
                    3: Dim("width", min=48, max=80),
                },
            ),
        )
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_efficientnet_b0_combined_dynamic_aoti_compile(self) -> None:
        module = _torchvision_efficientnet_b0().eval().to("musa")
        exported = export(
            module,
            (torch.randn(2, 3, 64, 64, device="musa"),),
            dynamic_shapes=(
                {
                    0: Dim("batch", min=1, max=3),
                    2: Dim("height", min=48, max=80),
                    3: Dim("width", min=48, max=80),
                },
            ),
        )
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_vit_b16_aoti_compile(self) -> None:
        module = _torchvision_vit_b_16().eval().to("musa")
        exported = export_with_musa_sdpa_decomposition(
            module, (torch.randn(1, 3, 224, 224, device="musa"),)
        )
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_vit_b16_prefix_dynamic_batch_aoti_compile(self) -> None:
        module = _torchvision_vit_b_16()
        module.encoder.layers = module.encoder.layers[:1]
        module = module.eval().to("musa")
        exported = export_with_musa_sdpa_decomposition(
            module,
            (torch.randn(2, 3, 224, 224, device="musa"),),
            dynamic_shapes=({0: Dim("batch", min=1, max=3)},),
        )
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertEqual(len(exported.range_constraints), 1)
        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_vit_layer_tokens_dynamic_batch_aoti_compile(self) -> None:
        model = _torchvision_vit_b_16()
        model.encoder.layers = model.encoder.layers[:1]
        module = VitLayerTokens(model).eval().to("musa")
        exported = export_with_musa_sdpa_decomposition(
            module,
            (torch.randn(2, 3, 224, 224, device="musa"),),
            dynamic_shapes=({0: Dim("batch", min=1, max=3)},),
        )
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertEqual(len(exported.range_constraints), 1)
        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_vit_attention_stages_dynamic_batch_aoti_compile(self) -> None:
        model = _torchvision_vit_b_16()
        model.encoder.layers = model.encoder.layers[:1]
        module = VitLayerTokens(model, attention_stages=True).eval().to("musa")
        exported = export_with_musa_sdpa_decomposition(
            module,
            (torch.randn(2, 3, 224, 224, device="musa"),),
            dynamic_shapes=({0: Dim("batch", min=1, max=3)},),
        )
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertEqual(len(exported.range_constraints), 1)
        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_transformer_encoder_aoti_compile(self) -> None:
        module = (
            nn.TransformerEncoderLayer(
                d_model=16,
                nhead=4,
                dim_feedforward=32,
                dropout=0.0,
                batch_first=True,
            )
            .eval()
            .to("musa")
        )
        example = torch.randn(1, 4, 16, device="musa")
        exported = export(module, (example,))
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_dynamic_transformer_encoder_aoti_compile(self) -> None:
        module = (
            nn.TransformerEncoderLayer(
                d_model=16,
                nhead=4,
                dim_feedforward=32,
                dropout=0.0,
                batch_first=True,
            )
            .eval()
            .to("musa")
        )
        example = torch.randn(2, 4, 16, device="musa")
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        exported = export_with_musa_sdpa_decomposition(
            module,
            (example,),
            dynamic_shapes=({0: Dim("batch", min=1, max=3)},),
        )
        self.assertIs(torch.nn.functional.scaled_dot_product_attention, original_sdpa)
        self.assertEqual(len(exported.range_constraints), 1)
        result = MusaBackend.preprocess(
            exported, [MusaBackend.generate_method_name_compile_spec("forward")]
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_dynamic_causal_attention_sequence_export(self) -> None:
        class DynamicCausalAttention(nn.Module):
            def forward(self, tensor: torch.Tensor) -> torch.Tensor:
                query = tensor.reshape(1, tensor.shape[1], 4, 4).transpose(1, 2)
                output = torch.nn.functional.scaled_dot_product_attention(
                    query, query, query, is_causal=True
                )
                return output.transpose(1, 2).reshape(tensor.shape)

        example = torch.randn(1, 3, 16, device="musa")
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        exported = export_with_musa_sdpa_decomposition(
            DynamicCausalAttention().eval(),
            (example,),
            dynamic_shapes=({1: Dim("sequence", min=1, max=7)},),
        )
        self.assertIs(torch.nn.functional.scaled_dot_product_attention, original_sdpa)
        self.assertEqual(len(exported.range_constraints), 1)
        self.assertFalse(
            any(
                "scaled_dot_product_attention" in str(node.target)
                for node in exported.graph.nodes
            )
        )

    def test_causal_gqa_aoti_compile(self) -> None:
        class CausalGqa(nn.Module):
            def forward(
                self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> torch.Tensor:
                return torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, is_causal=True, enable_gqa=True
                )

        query = torch.randn(1, 4, 4, 8, device="musa")
        key = torch.randn(1, 2, 4, 8, device="musa")
        value = torch.randn(1, 2, 4, 8, device="musa")
        exported = export(CausalGqa().eval(), (query, key, value))
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_sdpa_decomposition(self) -> None:
        mask = torch.tensor(
            [[True, True, False, True, False]] * 3, device="musa"
        )
        options = (
            {"attn_mask": mask},
            {"attn_mask": mask.to(torch.float32) * -0.5},
            {"is_causal": True},
            {"scale": 0.3},
            {"enable_gqa": True},
        )
        for dtype, tolerance in (
            (torch.float32, 1e-4),
            (torch.float16, 1e-2),
            (torch.bfloat16, 4e-2),
        ):
            query = torch.randn(1, 4, 3, 8, device="musa", dtype=dtype)
            key = torch.randn(1, 4, 5, 8, device="musa", dtype=dtype)
            value = torch.randn(1, 4, 5, 8, device="musa", dtype=dtype)
            for kwargs in options:
                with self.subTest(dtype=dtype, kwargs=kwargs):
                    kwargs = dict(kwargs)
                    mask_tensor = kwargs.get("attn_mask")
                    if mask_tensor is not None and mask_tensor.dtype != torch.bool:
                        kwargs["attn_mask"] = mask_tensor.to(dtype)
                    current_key = key[:, :2] if kwargs.get("enable_gqa") else key
                    current_value = value[:, :2] if kwargs.get("enable_gqa") else value
                    expected = torch.nn.functional.scaled_dot_product_attention(
                        query, current_key, current_value, **kwargs
                    )
                    actual = _decompose_scaled_dot_product_attention(
                        query, current_key, current_value, **kwargs
                    )
                    torch.testing.assert_close(
                        actual, expected, atol=tolerance, rtol=tolerance
                    )
