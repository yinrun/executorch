# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import hashlib
import math
import os
import typing
from unittest import mock
from typing import Any, Dict, List

import torch
from torch._inductor import ir, lowering
from executorch.backends.aoti.aoti_backend import AotiBackend
from executorch.backends.cuda.cuda_weight_collector import (
    AOTI_DEVICE_TYPE_CPU,
    AOTI_DEVICE_TYPE_CUDA,
    CudaWeightCollector,
)
from executorch.exir._warnings import experimental
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.graph_module import contains_any_call_fn_target_op


def _aoti_device_type_for_weight(tensor: torch.Tensor) -> int:
    if tensor.device.type in ("cuda", "musa"):
        return AOTI_DEVICE_TYPE_CUDA
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


def _decompose_musa_batch_norm(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    momentum: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        _materialize_musa_batch_norm_parameters(
            input, weight, bias, running_mean, running_var, eps
        ),
        running_mean,
        running_var,
    )


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
        ir.ExternKernelAlloc(
            layout=layout,
            inputs=[input, weight, bias, running_mean, running_var],
            constant_args=[eps],
            python_kernel_name="executorch_musa.batch_norm",
            cpp_kernel_name="aoti_torch_musa_batch_norm",
            op_overload=_MUSA_BATCH_NORM,
        )
    )


lowering.register_lowering(_MUSA_BATCH_NORM, type_promotion_kind=None)(
    _lower_musa_batch_norm
)


@contextlib.contextmanager
def _musa_aoti_compile_context():
    from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu

    original_add_device_include = CppWrapperCpu.add_device_include

    def add_device_include_with_musa_shims(self: Any, device: str) -> None:
        original_add_device_include(self, device)
        if device == "musa":
            self.header.splice(
                f'#include "{_MUSA_BATCH_NORM_HEADER}"'
            )

    CppWrapperCpu.add_device_include = add_device_include_with_musa_shims
    try:
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
            "aoti_torch_musa_batch_norm": None,
        }

    @classmethod
    def get_decomposition_table(cls) -> Dict[Any, Any]:
        return {
            torch.ops.aten.scaled_dot_product_attention.default: (
                _decompose_scaled_dot_product_attention
            ),
            torch.ops.aten.linear.default: _decompose_linear,
            torch.ops.aten.batch_norm.default: _decompose_aten_batch_norm,
            torch.ops.aten._native_batch_norm_legit_no_training.default: (
                _decompose_musa_batch_norm
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
        sdpa_context = (
            torch._inductor.config.patch({"pattern_matcher": False})
            if has_sdpa or has_attention_math
            else contextlib.nullcontext()
        )
        with collector.capture(_aoti_device_type_for_weight) as capture, sdpa_context:
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
