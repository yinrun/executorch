"""Scoped native FP32 reductions and BF16 Qwen query-normalization sums."""

import os

import torch
from torch._inductor import ir, lowering

from executorch.exir.dialects.edge._ops import EdgeOpOverload


HEADER = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "cuda", "runtime", "shims", "musa_reduce.h"
    )
)
_library = torch.library.Library("executorch_musa", "FRAGMENT")


def supported_reduction(value, axis, mean):
    if not isinstance(value, torch.Tensor) or not 1 <= value.ndim <= 8:
        return False
    if type(axis) is not int or not -value.ndim <= axis < value.ndim:
        return False
    if type(mean) is not bool or any(
        type(size) is not int or size <= 0 for size in value.shape
    ):
        return False
    axis %= value.ndim
    if value.dtype == torch.bfloat16:
        return tuple(value.shape) == (1, 1, 16, 128) and axis == 3 and not mean
    if value.dtype != torch.float32:
        return False
    return (axis == value.ndim - 1 and value.shape[-1] in (128, 256, 1024)) or (
        not mean and axis == 2 and tuple(value.shape) == (1, 16, 128, 128)
    )


def reduction_shape(value, axis, keepdim, mean):
    torch._check(value.device.type in ("musa", "meta"))
    torch._check(type(keepdim) is bool and supported_reduction(value, axis, mean))
    shape = list(value.shape)
    axis %= value.ndim
    if keepdim:
        shape[axis] = 1
    else:
        shape.pop(axis)
    return shape


def reduction_meta(value, axis, keepdim, mean):
    return value.new_empty(reduction_shape(value, axis, keepdim, mean))


def reduction_impl(value, axis, keepdim, mean):
    reduction_shape(value, axis, keepdim, mean)
    return (
        value.mean(axis, keepdim=keepdim) if mean else value.sum(axis, keepdim=keepdim)
    )


def reduction_out_meta(value, axis, keepdim, mean, *, out):
    shape = reduction_shape(value, axis, keepdim, mean)
    torch._check(tuple(out.shape) == tuple(shape))
    torch._check(out.dtype == value.dtype and out.device == value.device)
    torch._check(out.is_contiguous())
    return out


def reduction_out_impl(value, axis, keepdim, mean, *, out):
    reduction_out_meta(value, axis, keepdim, mean, out=out)
    begin = value.data_ptr()
    span = 1 + sum(
        (size - 1) * stride for size, stride in zip(value.shape, value.stride())
    )
    if (
        out.data_ptr() < begin + span * value.element_size()
        and begin < out.data_ptr() + out.numel() * out.element_size()
    ):
        raise ValueError("Reduction output must not overlap its input")
    return out.copy_(reduction_impl(value, axis, keepdim, mean))


try:
    REDUCE = torch.ops.executorch_musa.reduce.default
except AttributeError:
    _library.define("reduce(Tensor value, int axis, bool keepdim, bool mean) -> Tensor")
    _library.impl("reduce", reduction_impl, "PrivateUse1")
    _library.impl("reduce", reduction_meta, "Meta")
    REDUCE = torch.ops.executorch_musa.reduce.default

try:
    REDUCE_OUT = torch.ops.executorch_musa.reduce.out
except AttributeError:
    _library.define(
        "reduce.out(Tensor value, int axis, bool keepdim, bool mean, *, Tensor(a!) out) -> Tensor(a!)"
    )
    _library.impl("reduce.out", reduction_out_impl, "PrivateUse1")
    _library.impl("reduce.out", reduction_out_meta, "Meta")
    REDUCE_OUT = torch.ops.executorch_musa.reduce.out


@lowering.register_lowering(REDUCE, type_promotion_kind=None)
def lower_reduction(value, axis, keepdim, mean):
    value.realize()
    shape = list(value.get_size())
    axis %= len(shape)
    if keepdim:
        shape[axis] = 1
    else:
        shape.pop(axis)
    layout = ir.FixedLayout(
        value.get_device(),
        value.get_dtype(),
        shape,
        ir.FlexibleLayout.contiguous_strides(shape),
    )
    symbol = (
        "aoti_torch_musa_bf16_sum_out"
        if value.get_dtype() == torch.bfloat16
        else "aoti_torch_musa_reduce_out"
    )
    return ir.TensorBox.create(
        ir.ExternKernelOut(
            layout=layout,
            inputs=[value],
            constant_args=[axis, keepdim, mean],
            python_kernel_name="executorch_musa.reduce.out",
            cpp_kernel_name=symbol,
            op_overload=REDUCE_OUT,
        )
    )


def replace_reductions(graph_module):
    count = 0
    for node in graph_module.graph.nodes:
        target = (
            node.target._op if isinstance(node.target, EdgeOpOverload) else node.target
        )
        if node.op != "call_function" or target not in (
            torch.ops.aten.mean.dim,
            torch.ops.aten.sum.dim_IntList,
        ):
            continue
        if not node.args or not isinstance(node.args[0], torch.fx.Node):
            continue
        value = node.args[0].meta.get("val")
        if not isinstance(value, torch.Tensor) or value.device.type != "musa":
            continue
        dimensions = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        dimensions = [dimensions] if type(dimensions) is int else dimensions
        if not isinstance(dimensions, (list, tuple)) or len(dimensions) != 1:
            continue
        axis = dimensions[0]
        mean = target == torch.ops.aten.mean.dim
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        if type(keepdim) is not bool or not supported_reduction(value, axis, mean):
            continue
        if node.kwargs.get("dtype") not in (None, value.dtype):
            continue
        node.target = REDUCE
        node.args = (node.args[0], axis % value.ndim, keepdim, mean)
        node.kwargs = {}
        count += 1
    graph_module.graph.lint()
    graph_module.recompile()
    return count
