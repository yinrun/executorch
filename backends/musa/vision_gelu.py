"""Opt-in FP32 muDNN gelu for the original Qwen vision MLP and merger."""

from pathlib import Path

import torch
from torch._inductor import ir, lowering


HEADER = str(
    Path(__file__).resolve().parents[1] / "cuda/runtime/shims/musa_vision_gelu.h"
)
_library = torch.library.Library("executorch_musa", "FRAGMENT")


def validate(value, mode):
    torch._check(value.device.type in ("musa", "meta"))
    torch._check(value.dtype == torch.float32 and value.is_contiguous())
    torch._check(value.ndim == 2)
    torch._check(value.shape[0] in (200, 300, 800, 1200))
    torch._check(value.shape[1] == 4096)
    torch._check(type(mode) is int and mode in (0, 1))


def gelu_meta(value, mode):
    validate(value, mode)
    return torch.empty_like(value)


def gelu_eager(value, mode):
    validate(value, mode)
    return torch.nn.functional.gelu(value, approximate="tanh" if mode else "none")


def gelu_out_meta(value, mode, *, out):
    validate(value, mode)
    torch._check(out.shape == value.shape and out.dtype == value.dtype)
    torch._check(out.device == value.device and out.is_contiguous())
    return out


def gelu_out_eager(value, mode, *, out):
    gelu_out_meta(value, mode, out=out)
    size = value.numel() * value.element_size()
    if (
        value.data_ptr() < out.data_ptr() + size
        and out.data_ptr() < value.data_ptr() + size
    ):
        raise ValueError("Vision gelu output must not overlap its input")
    return out.copy_(gelu_eager(value, mode))


_library.define("vision_gelu(Tensor value, int mode) -> Tensor")
_library.impl("vision_gelu", gelu_eager, "PrivateUse1")
_library.impl("vision_gelu", gelu_meta, "Meta")
_library.define(
    "vision_gelu.out(Tensor value, int mode, *, Tensor(a!) out) -> Tensor(a!)"
)
_library.impl("vision_gelu.out", gelu_out_eager, "PrivateUse1")
_library.impl("vision_gelu.out", gelu_out_meta, "Meta")
VISION_GELU = torch.ops.executorch_musa.vision_gelu.default
VISION_GELU_OUT = torch.ops.executorch_musa.vision_gelu.out


@lowering.register_lowering(VISION_GELU, type_promotion_kind=None)
def lower_gelu(value, mode):
    value = ir.ExternKernel.require_contiguous(value)
    value.realize()
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
            inputs=[value],
            constant_args=[mode],
            python_kernel_name="executorch_musa.vision_gelu.out",
            cpp_kernel_name="aoti_torch_musa_vision_gelu_out",
            op_overload=VISION_GELU_OUT,
        )
    )
