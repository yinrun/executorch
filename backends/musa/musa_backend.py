# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import hashlib
import math
import operator
import os
import typing
from unittest import mock
from typing import Any, Dict, List

import torch
from torch._inductor import ir, lowering
from executorch.backends.aoti.aoti_backend import AotiBackend
from executorch.backends.cuda.cuda_weight_collector import (
    AOTI_DEVICE_TYPE_CPU,
    SERIALIZED_DEVICE_TYPE_MUSA,
    CudaWeightCollector,
)
from executorch.exir._warnings import experimental
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.graph_module import contains_any_call_fn_target_op


def _aoti_device_type_for_weight(tensor: torch.Tensor) -> int:
    if tensor.device.type == "musa":
        return SERIALIZED_DEVICE_TYPE_MUSA
    if tensor.device.type == "cpu":
        return AOTI_DEVICE_TYPE_CPU
    raise RuntimeError(f"Unsupported AOTI constant device for MUSA export: {tensor.device}")


def _normalize_musa_assert_devices(graph_module: torch.fx.GraphModule) -> None:
    for node in graph_module.graph.nodes:
        if (
            node.op != "call_function"
            or node.target != torch.ops.aten._assert_tensor_metadata.default
        ):
            continue
        expected_device = node.kwargs.get("device")
        if isinstance(expected_device, str):
            expected_device = torch.device(expected_device)
        tensor_meta = node.args[0].meta.get("val")
        if (
            isinstance(expected_device, torch.device)
            and expected_device.type == "musa"
            and expected_device.index is None
            and isinstance(tensor_meta, torch.Tensor)
            and tensor_meta.device.type == "musa"
            and tensor_meta.device.index is not None
        ):
            node.kwargs = {**node.kwargs, "device": tensor_meta.device}


def _make_musa_addmm_inputs_contiguous(
    graph_module: torch.fx.GraphModule,
) -> None:
    graph = graph_module.graph
    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or node.target
            not in (torch.ops.aten.addmm.default, torch.ops.aten.linear.default)
        ):
            continue
        input_index = 0 if node.target == torch.ops.aten.linear.default else 1
        with graph.inserting_before(node):
            contiguous_mat1 = graph.call_function(
                torch.ops.aten.clone.default,
                (node.args[input_index],),
                {"memory_format": torch.contiguous_format},
            )
        args = list(node.args)
        args[input_index] = contiguous_mat1
        node.args = tuple(args)
    graph.lint()
    graph_module.recompile()


_musa_batch_norm_library = torch.library.Library("executorch_musa", "FRAGMENT")


def _musa_batch_norm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.nn.functional.batch_norm(
        input, running_mean, running_var, weight, bias, False, 0.1, eps
    )


def _musa_batch_norm_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(input)


try:
    _MUSA_BATCH_NORM = torch.ops.executorch_musa.batch_norm.default
except AttributeError:
    _musa_batch_norm_library.define(
        "batch_norm(Tensor input, Tensor weight, Tensor bias, "
        "Tensor running_mean, Tensor running_var, float eps) -> Tensor"
    )
    _musa_batch_norm_library.impl(
        "batch_norm", _musa_batch_norm_impl, "PrivateUse1"
    )
    _musa_batch_norm_library.impl("batch_norm", _musa_batch_norm_meta, "Meta")
    _MUSA_BATCH_NORM = torch.ops.executorch_musa.batch_norm.default


def _musa_batch_norm_out_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    torch._check(out.shape == input.shape)
    torch._check(out.dtype == input.dtype)
    torch._check(out.device == input.device)
    torch._check(out.is_contiguous())
    return out


def _musa_batch_norm_out_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    _musa_batch_norm_out_meta(
        input, weight, bias, running_mean, running_var, eps, out=out
    )
    return out.copy_(
        _musa_batch_norm_impl(input, weight, bias, running_mean, running_var, eps)
    )


try:
    _MUSA_BATCH_NORM_OUT = torch.ops.executorch_musa.batch_norm.out
except AttributeError:
    _musa_batch_norm_library.define(
        "batch_norm.out(Tensor input, Tensor weight, Tensor bias, "
        "Tensor running_mean, Tensor running_var, float eps, "
        "*, Tensor(a!) out) -> Tensor(a!)"
    )
    _musa_batch_norm_library.impl(
        "batch_norm.out", _musa_batch_norm_out_impl, "PrivateUse1"
    )
    _musa_batch_norm_library.impl("batch_norm.out", _musa_batch_norm_out_meta, "Meta")
    _MUSA_BATCH_NORM_OUT = torch.ops.executorch_musa.batch_norm.out


_MUSA_BATCH_NORM_HEADER = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "backends",
        "cuda",
        "runtime",
        "shims",
        "musa_batch_norm.h",
    )
)

_musa_activation_library = torch.library.Library("executorch_musa", "FRAGMENT")
try:
    _MUSA_HARDSWISH = torch.ops.executorch_musa.hardswish.default
except AttributeError:
    _musa_activation_library.define("hardswish(Tensor input) -> Tensor")
    _musa_activation_library.impl(
        "hardswish", torch.nn.functional.hardswish, "PrivateUse1"
    )
    _musa_activation_library.impl(
        "hardswish",
        lambda input: torch.empty_like(input, memory_format=torch.contiguous_format),
        "Meta",
    )
    _MUSA_HARDSWISH = torch.ops.executorch_musa.hardswish.default

try:
    _MUSA_HARDSIGMOID = torch.ops.executorch_musa.hardsigmoid.default
except AttributeError:
    _musa_activation_library.define("hardsigmoid(Tensor input) -> Tensor")
    _musa_activation_library.impl(
        "hardsigmoid", torch.nn.functional.hardsigmoid, "PrivateUse1"
    )
    _musa_activation_library.impl(
        "hardsigmoid",
        lambda input: torch.empty_like(input, memory_format=torch.contiguous_format),
        "Meta",
    )
    _MUSA_HARDSIGMOID = torch.ops.executorch_musa.hardsigmoid.default

_MUSA_ACTIVATION_HEADER = os.path.join(
    os.path.dirname(_MUSA_BATCH_NORM_HEADER), "musa_activation.h"
)
_MUSA_CONVOLUTION_HEADER = os.path.join(
    os.path.dirname(_MUSA_BATCH_NORM_HEADER), "musa_convolution.h"
)


def _decompose_musa_hardswish(input: torch.Tensor) -> torch.Tensor:
    return _MUSA_HARDSWISH(input)


def _decompose_musa_hardsigmoid(input: torch.Tensor) -> torch.Tensor:
    return _MUSA_HARDSIGMOID(input)


def _lower_musa_activation(input: Any, op: Any, name: str) -> Any:
    input.realize()
    layout = ir.FixedLayout(
        input.get_device(),
        input.get_dtype(),
        input.get_size(),
        ir.FlexibleLayout.contiguous_strides(input.get_size()),
    )
    return ir.TensorBox.create(
        ir.ExternKernelOut(
            layout=layout,
            inputs=[input],
            constant_args=[],
            python_kernel_name=f"aten.{name}.out",
            cpp_kernel_name=f"aoti_torch_musa_{name}_out",
            op_overload=op,
        )
    )


@lowering.register_lowering(_MUSA_HARDSWISH, type_promotion_kind=None)
def _lower_musa_hardswish(input: Any) -> Any:
    return _lower_musa_activation(input, torch.ops.aten.hardswish.out, "hardswish")


@lowering.register_lowering(_MUSA_HARDSIGMOID, type_promotion_kind=None)
def _lower_musa_hardsigmoid(input: Any) -> Any:
    return _lower_musa_activation(input, torch.ops.aten.hardsigmoid.out, "hardsigmoid")


def _materialize_musa_batch_norm_parameters(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    if running_mean is None or running_var is None:
        raise RuntimeError(
            "MUSA batch norm requires running statistics for inference"
        )
    if weight is None:
        weight = torch.ones_like(running_mean)
    if bias is None:
        bias = torch.zeros_like(running_mean)
    return _MUSA_BATCH_NORM(input, weight, bias, running_mean, running_var, eps)


def _empty_musa_batch_norm_stats(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    empty_mean = torch.empty((0,), device=input.device, dtype=torch.float32)
    empty_invstd = torch.empty((0,), device=input.device, dtype=torch.float32)
    return empty_mean, empty_invstd


def _decompose_musa_batch_norm(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    momentum: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = _materialize_musa_batch_norm_parameters(
        input, weight, bias, running_mean, running_var, eps
    )
    empty_mean, empty_invstd = _empty_musa_batch_norm_stats(input)
    return output, empty_mean, empty_invstd


def _decompose_aten_batch_norm(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    training: bool,
    momentum: float,
    eps: float,
    cudnn_enabled: bool,
) -> torch.Tensor:
    if training:
        raise RuntimeError("MUSA AOTI batch norm supports inference only")
    return _materialize_musa_batch_norm_parameters(
        input, weight, bias, running_mean, running_var, eps
    )


def _decompose_native_batch_norm(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if training:
        raise RuntimeError("MUSA AOTI batch norm supports inference only")
    output = _materialize_musa_batch_norm_parameters(
        input, weight, bias, running_mean, running_var, eps
    )
    empty_mean, empty_invstd = _empty_musa_batch_norm_stats(input)
    return output, empty_mean, empty_invstd


def _decompose_native_batch_norm_functional(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output, empty_mean, empty_invstd = _decompose_native_batch_norm(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )
    return output, empty_mean, empty_invstd, running_mean, running_var


def _decompose_batch_norm_no_update(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    momentum: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output = _materialize_musa_batch_norm_parameters(
        input, weight, bias, running_mean, running_var, eps
    )
    empty_mean, empty_invstd = _empty_musa_batch_norm_stats(input)
    empty_reserve = torch.empty((0,), device=input.device, dtype=torch.uint8)
    return output, empty_mean, empty_invstd, empty_reserve


def _lower_musa_batch_norm(
    input: Any,
    weight: Any,
    bias: Any,
    running_mean: Any,
    running_var: Any,
    eps: float,
) -> Any:
    for tensor in (input, weight, bias, running_mean, running_var):
        tensor.realize()
    layout = ir.FixedLayout(
        input.get_device(),
        input.get_dtype(),
        input.get_size(),
        ir.FlexibleLayout.contiguous_strides(input.get_size()),
    )
    return ir.TensorBox.create(
        ir.ExternKernelOut(
            layout=layout,
            inputs=[input, weight, bias, running_mean, running_var],
            constant_args=[eps],
            python_kernel_name="executorch_musa.batch_norm.out",
            cpp_kernel_name="aoti_torch_musa_batch_norm_out",
            op_overload=_MUSA_BATCH_NORM_OUT,
        )
    )


lowering.register_lowering(_MUSA_BATCH_NORM, type_promotion_kind=None)(
    _lower_musa_batch_norm
)


@contextlib.contextmanager
def _musa_aoti_compile_context():
    import sympy
    from torch._inductor.codegen import cpp_wrapper_cpu, memory_planning
    from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
    from torch._inductor.cpp_builder import CppTorchDeviceOptions
    from torch._inductor.kernel import conv
    from torch._inductor.utils import cache_on_self
    from torch._inductor.virtualized import V as virtualized
    from torch.utils._sympy.functions import Max

    original_add_device_include = CppWrapperCpu.add_device_include
    original_alloc_from_pool = CppWrapperCpu.codegen_alloc_from_pool
    original_cexpr = cpp_wrapper_cpu.cexpr
    original_symbolic_size = memory_planning.TemporalSplit.get_symbolic_size
    original_get_libraries = CppTorchDeviceOptions.get_libraries
    convolution_op = torch.ops.aten.convolution.default
    original_convolution = lowering.lowerings[convolution_op]

    def convolution_with_preallocated_output(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    ):
        if input.get_device().type == "musa":
            if len(input.get_size()) == len(weight.get_size()) - 1:
                return lowering.squeeze(
                    convolution_with_preallocated_output(
                        lowering.unsqueeze(input, 0),
                        weight,
                        bias,
                        stride,
                        padding,
                        dilation,
                        transposed,
                        output_padding,
                        groups,
                    ),
                    0,
                )
            input = ir.ExternKernel.require_contiguous(input)
            weight = ir.ExternKernel.require_contiguous(weight)
            inputs = [input, weight]
            if bias is not None:
                bias = ir.ExternKernel.require_contiguous(bias)
                inputs.append(bias)
            for tensor in inputs:
                tensor.realize()
            spatial_dimensions = len(weight.get_size()) - 2
            parameters = {
                "stride": conv.pad_listlike(tuple(stride), spatial_dimensions),
                "padding": conv.pad_listlike(tuple(padding), spatial_dimensions),
                "dilation": conv.pad_listlike(tuple(dilation), spatial_dimensions),
                "transposed": transposed,
                "output_padding": conv.pad_listlike(
                    tuple(output_padding), spatial_dimensions
                ),
                "groups": groups,
            }
            inferred_layout = conv.conv_layout(input, weight, bias, **parameters)
            layout = ir.FixedLayout(
                input.get_device(),
                input.get_dtype(),
                inferred_layout.size,
                ir.FlexibleLayout.contiguous_strides(inferred_layout.size),
            )
            constant_args = list(parameters.values())
            if bias is None:
                constant_args.insert(0, None)
            return ir.TensorBox.create(
                ir.ExternKernelOut(
                    layout=layout,
                    inputs=inputs,
                    constant_args=constant_args,
                    python_kernel_name="aten.convolution.out",
                    cpp_kernel_name="aoti_torch_musa_convolution_out",
                    op_overload=torch.ops.aten.convolution.out,
                )
            )
        return original_convolution(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )

    def add_device_include_with_musa_shims(self: Any, device: str) -> None:
        original_add_device_include(self, device)
        if device == "musa":
            self.header.splice(f'#include "{_MUSA_BATCH_NORM_HEADER}"')
            self.header.splice(f'#include "{_MUSA_ACTIVATION_HEADER}"')
            self.header.splice(f'#include "{_MUSA_CONVOLUTION_HEADER}"')

    def alloc_from_pool_without_expansion(self, name, offset, dtype, shape, stride):
        if "musa" not in self.get_codegened_graph().device_types:
            return original_alloc_from_pool(self, name, offset, dtype, shape, stride)

        def print_pool_offset(expression, **kwargs):
            if expression is offset and isinstance(expression, sympy.Expr):
                expression = expression.xreplace(
                    virtualized.graph.sizevars.replacements
                )
                kwargs["simplify"] = False
            return original_cexpr(expression, **kwargs)

        with mock.patch.object(cpp_wrapper_cpu, "cexpr", print_pool_offset):
            return original_alloc_from_pool(self, name, offset, dtype, shape, stride)

    @cache_on_self
    def get_symbolic_size(self):
        if "musa" not in virtualized.graph.device_types:
            return original_symbolic_size(self)
        if not self.allocations:
            return 0
        return Max(*(block.get_symbolic_size() for block in self.allocations))

    def get_runtime_libraries(self: Any) -> List[str]:
        libraries = original_get_libraries(self)
        if (
            not self.get_aot_mode()
            or self.get_compile_only()
            or "USE_MUSA" not in {value.strip() for value in self.get_definitions()}
        ):
            return libraries
        excluded = {"torch", "torch_cpu", "torch_python", "c10", "musa_python"}
        return list(
            dict.fromkeys(
                [
                    *(library for library in libraries if library not in excluded),
                    "musart",
                ]
            )
        )

    CppWrapperCpu.add_device_include = add_device_include_with_musa_shims
    try:
        with mock.patch.dict(
            lowering.lowerings, {convolution_op: convolution_with_preallocated_output}
        ), mock.patch.object(
            CppTorchDeviceOptions, "get_libraries", get_runtime_libraries
        ), mock.patch.object(
            CppWrapperCpu, "codegen_alloc_from_pool", alloc_from_pool_without_expansion
        ), mock.patch.object(
            memory_planning.TemporalSplit, "get_symbolic_size", get_symbolic_size
        ):
            yield
    finally:
        CppWrapperCpu.add_device_include = original_add_device_include


def _decompose_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    if dropout_p != 0.0:
        raise ValueError("MUSA AOTI SDPA does not support dropout")
    if is_causal and attn_mask is not None:
        raise ValueError("MUSA AOTI SDPA cannot combine causal and custom masks")
    if enable_gqa:
        if query.size(-3) % key.size(-3) or query.size(-3) % value.size(-3):
            raise ValueError("MUSA AOTI GQA requires divisible heads")
        key = key.repeat_interleave(query.size(-3) // key.size(-3), dim=-3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), dim=-3)

    output_dtype = query.dtype
    if output_dtype in (torch.float16, torch.bfloat16):
        query, key, value = query.float(), key.float(), value.float()
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores * (scale if scale is not None else 1.0 / math.sqrt(query.size(-1)))
    if is_causal:
        causal_mask = torch.ones(
            query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
        ).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    return torch.matmul(torch.softmax(scores, dim=-1), value).to(output_dtype)


def _decompose_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    contiguous_input = input.contiguous()
    if bias is None:
        return torch.matmul(contiguous_input, weight.t())
    if contiguous_input.dim() == 2:
        return torch.addmm(bias, contiguous_input, weight.t())
    return torch.matmul(contiguous_input, weight.t()) + bias


def export_with_musa_pooling_shapes(
    module: torch.nn.Module, args: tuple[Any, ...], **kwargs: Any
) -> torch.export.ExportedProgram:
    """Export pooling without the meta kernel's redundant remainder guards.

    Python integer division already rounds toward negative infinity. The
    installed pooling meta helper additionally tests the remainder, specializing
    symbolic spatial dimensions even though its correction is never needed.
    Exports must run sequentially while this scoped compatibility patch is active.
    """
    from torch import _meta_registrations

    with mock.patch.object(_meta_registrations, "div_rtn", operator.floordiv):
        return torch.export.export(module, args, **kwargs)


def export_with_musa_sdpa_decomposition(
    module: torch.nn.Module, args: tuple[Any, ...], **kwargs: Any
) -> torch.export.ExportedProgram:
    """Export functional SDPA without invoking TorchMUSA's shape-specializing path.

    Export calls must run sequentially because the functional replacement is global.
    """
    with mock.patch.object(
        torch.nn.functional,
        "scaled_dot_product_attention",
        _decompose_scaled_dot_product_attention,
    ):
        return torch.export.export(module, args, **kwargs)


@experimental(
    "MusaBackend is an experimental AOTInductor integration for ExecuTorch."
)
class MusaBackend(AotiBackend, BackendDetails):
    """AOTInductor backend for the MUSA PrivateUse1 device.

    The first implementation uses the existing CUDA AOTI weight metadata wire
    format. The runtime can therefore validate the export without introducing a
    new serialized device enum; native MUSA metadata is a follow-up task.
    """

    @classmethod
    def get_device_name(cls) -> str:
        return "musa"

    @classmethod
    def get_supported_fallback_kernels(cls) -> Dict[str, Any]:
        return {
            "aoti_torch_musa_addmm_out": None,
            "aoti_torch_musa_mm_out": None,
            "aoti_torch_musa_bmm_out": None,
            "aoti_torch_musa_convolution": None,
            "aoti_torch_musa_convolution_out": None,
            "aoti_torch_musa_batch_norm": None,
            "aoti_torch_musa_batch_norm_out": None,
            "aoti_torch_musa_hardswish": None,
            "aoti_torch_musa_hardsigmoid": None,
            "aoti_torch_musa_hardswish_out": None,
            "aoti_torch_musa_hardsigmoid_out": None,
        }

    @classmethod
    def get_decomposition_table(cls) -> Dict[Any, Any]:
        return {
            torch.ops.aten.scaled_dot_product_attention.default: (
                _decompose_scaled_dot_product_attention
            ),
            torch.ops.aten.linear.default: _decompose_linear,
            torch.ops.aten.hardswish.default: _decompose_musa_hardswish,
            torch.ops.aten.hardsigmoid.default: _decompose_musa_hardsigmoid,
            torch.ops.aten.batch_norm.default: _decompose_aten_batch_norm,
            torch.ops.aten._native_batch_norm_legit_no_training.default: (
                _decompose_musa_batch_norm
            ),
            torch.ops.aten.native_batch_norm.default: _decompose_native_batch_norm,
            torch.ops.aten._native_batch_norm_legit.default: (
                _decompose_native_batch_norm
            ),
            torch.ops.aten._native_batch_norm_legit_functional.default: (
                _decompose_native_batch_norm_functional
            ),
            torch.ops.aten._batch_norm_no_update.default: (
                _decompose_batch_norm_no_update
            ),
        }

    @classmethod
    def get_custom_passes(cls, compile_specs: List[CompileSpec]) -> List[typing.Any]:
        return [
            _normalize_musa_assert_devices,
            _make_musa_addmm_inputs_contiguous,
        ]

    @classmethod
    def save_data_externally(cls) -> bool:
        return True

    @classmethod
    def get_aoti_compile_options(
        cls, compile_specs: List[CompileSpec]
    ) -> Dict[str, typing.Any]:
        options: Dict[str, typing.Any] = {
            "layout_optimization": False,
            "loop_ordering_after_fusion": False,
            "emulate_precision_casts": True,
            "aot_inductor.embed_kernel_binary": True,
            "aot_inductor.package": True,
            "aot_inductor.package_constants_in_so": False,
            "aot_inductor.package_constants_on_disk": True,
            "max_autotune": False,
        }
        config = getattr(torch._inductor.config, "_config", {})
        return {key: value for key, value in options.items() if key in config}

    @classmethod
    def get_extra_aoti_compile_context_manager(
        cls, compile_specs: List[CompileSpec] | None = None
    ):
        return _musa_aoti_compile_context()

    @classmethod
    def materialize_weights_blob(
        cls, paths: Any, compile_specs: List[CompileSpec]
    ) -> Any:
        if not isinstance(paths, list):
            return paths

        from torch.export.pt2_archive._package_weights import Weights

        weights = [path for path in paths if isinstance(path, Weights)]
        if not weights:
            return paths
        if len(weights) != 1:
            raise RuntimeError(f"Expected one MUSA AOTI weights output, got {len(weights)}")

        so_path = next(
            (
                path
                for path in paths
                if isinstance(path, str) and path.endswith(".wrapper.so")
            ),
            None,
        )
        if so_path is None:
            raise RuntimeError(f"Expected a MUSA AOTI .wrapper.so output, got {paths}")

        blob_path = os.path.splitext(so_path)[0] + "_weights.blob"
        capture = CudaWeightCollector.current_capture()
        capture.artifact = capture.collector.materialize(
            weights[0], os.path.dirname(blob_path), capture.device_type_for_weight
        )
        with open(blob_path, "wb"):
            pass

        materialized = [path for path in paths if not isinstance(path, Weights)]
        materialized.append(blob_path)
        return materialized

    @classmethod
    def preprocess(
        cls,
        edge_program: Any,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        collector = CudaWeightCollector()
        has_sdpa = contains_any_call_fn_target_op(
            edge_program.graph_module,
            {torch.ops.aten.scaled_dot_product_attention.default},
        )
        has_attention_math = contains_any_call_fn_target_op(
            edge_program.graph_module,
            {torch.ops.aten.softmax.int, torch.ops.aten._softmax.default},
        ) and contains_any_call_fn_target_op(
            edge_program.graph_module,
            {torch.ops.aten.matmul.default, torch.ops.aten.bmm.default},
        )
        compiler_options = {}
        if has_sdpa or has_attention_math:
            compiler_options["pattern_matcher"] = False
        if edge_program.graph_signature.buffers_to_mutate:
            compiler_options["always_keep_tensor_constants"] = True
        with collector.capture(
            _aoti_device_type_for_weight
        ) as capture, torch._inductor.config.patch(compiler_options):
            result = AotiBackend.preprocess.__func__(cls, edge_program, compile_specs)
        if capture.artifact is None:
            raise RuntimeError("MUSA AOTI did not return a structured Weights output")
        if not capture.artifact.entries:
            collector.finish()
            return result
        collector.add_preprocess_result(
            result,
            capture.artifact,
            cls.get_device_name(),
        )
        collector.finish()
        return result
