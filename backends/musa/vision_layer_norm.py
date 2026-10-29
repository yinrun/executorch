"""Opt-in FP32 muDNN Welford LayerNorm for the two Qwen vision grids."""

import math
from pathlib import Path

import torch
from torch._inductor import ir, lowering


HEADER = str(
    Path(__file__).resolve().parents[1] / "cuda/runtime/shims/musa_vision_layer_norm.h"
)
_library = torch.library.Library("executorch_musa", "FRAGMENT")


def validate(value, weight, bias, epsilon):
    torch._check(value.device.type in ("musa", "meta"))
    torch._check(value.ndim == 2)
    torch._check(value.shape[0] in (800, 1200) and value.shape[1] == 1024)
    torch._check(math.isfinite(epsilon) and epsilon > 0)
    for tensor in (value, weight, bias):
        torch._check(tensor.device == value.device)
        torch._check(tensor.dtype == torch.float32 and tensor.is_contiguous())
    torch._check(weight.shape == (1024,) and bias.shape == (1024,))


def layer_norm_meta(value, weight, bias, epsilon):
    validate(value, weight, bias, epsilon)
    return torch.empty_like(value)


def layer_norm_eager(value, weight, bias, epsilon):
    validate(value, weight, bias, epsilon)
    return torch.nn.functional.layer_norm(value, (1024,), weight, bias, epsilon)


def layer_norm_out_meta(value, weight, bias, epsilon, *, out):
    validate(value, weight, bias, epsilon)
    torch._check(out.shape == value.shape and out.dtype == value.dtype)
    torch._check(out.device == value.device and out.is_contiguous())
    return out


def layer_norm_out_eager(value, weight, bias, epsilon, *, out):
    layer_norm_out_meta(value, weight, bias, epsilon, out=out)
    begin = out.data_ptr()
    end = begin + out.numel() * out.element_size()
    for source in (value, weight, bias):
        if (
            begin < source.data_ptr() + source.numel() * source.element_size()
            and source.data_ptr() < end
        ):
            raise ValueError("LayerNorm output must not overlap its inputs")
    return out.copy_(layer_norm_eager(value, weight, bias, epsilon))


_library.define(
    "vision_layer_norm(Tensor value, Tensor weight, Tensor bias, float epsilon) -> Tensor"
)
_library.impl("vision_layer_norm", layer_norm_eager, "PrivateUse1")
_library.impl("vision_layer_norm", layer_norm_meta, "Meta")
_library.define(
    "vision_layer_norm.out(Tensor value, Tensor weight, Tensor bias, float epsilon, *, Tensor(a!) out) -> Tensor(a!)"
)
_library.impl("vision_layer_norm.out", layer_norm_out_eager, "PrivateUse1")
_library.impl("vision_layer_norm.out", layer_norm_out_meta, "Meta")
VISION_LAYER_NORM = torch.ops.executorch_musa.vision_layer_norm.default
VISION_LAYER_NORM_OUT = torch.ops.executorch_musa.vision_layer_norm.out


@lowering.register_lowering(VISION_LAYER_NORM, type_promotion_kind=None)
def lower_layer_norm(value, weight, bias, epsilon):
    for tensor in (value, weight, bias):
        tensor.realize()
    shape = list(value.get_size())
    layout = ir.FixedLayout(
        value.get_device(),
        value.get_dtype(),
        shape,
        ir.FlexibleLayout.contiguous_strides(shape),
    )
    return ir.TensorBox.create(
        ir.ExternKernelOut(
            layout=layout,
            inputs=[value, weight, bias],
            constant_args=[epsilon],
            python_kernel_name="executorch_musa.vision_layer_norm.out",
            cpp_kernel_name="aoti_torch_musa_vision_layer_norm_out",
            op_overload=VISION_LAYER_NORM_OUT,
        )
    )


class VisionLayerNorm(torch.nn.LayerNorm):
    def forward(self, value):
        if self.normalized_shape != (1024,) or self.weight is None or self.bias is None:
            raise ValueError("Vision LayerNorm requires affine width 1024")
        return VISION_LAYER_NORM(value.contiguous(), self.weight, self.bias, self.eps)
