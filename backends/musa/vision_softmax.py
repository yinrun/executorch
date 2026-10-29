"""Opt-in FP32 muDNN softmax for the original Qwen vision attention grids."""

from pathlib import Path

import torch
from torch._inductor import ir, lowering


HEADER = str(
    Path(__file__).resolve().parents[1] / "cuda/runtime/shims/musa_vision_softmax.h"
)
_library = torch.library.Library("executorch_musa", "FRAGMENT")


def validate(value):
    torch._check(value.device.type in ("musa", "meta"))
    torch._check(value.dtype == torch.float32 and value.is_contiguous())
    torch._check(value.ndim == 3)
    torch._check(value.shape[0] == 16)
    torch._check(value.shape[1] in (800, 1200))
    torch._check(value.shape[2] == value.shape[1])


def softmax_meta(value):
    validate(value)
    return torch.empty_like(value)


def softmax_eager(value):
    validate(value)
    return torch.nn.functional.softmax(value, dim=-1, dtype=torch.float32)


def softmax_out_meta(value, *, out):
    validate(value)
    torch._check(out.shape == value.shape and out.dtype == value.dtype)
    torch._check(out.device == value.device and out.is_contiguous())
    return out


def softmax_out_eager(value, *, out):
    softmax_out_meta(value, out=out)
    size = value.numel() * value.element_size()
    if (
        value.data_ptr() < out.data_ptr() + size
        and out.data_ptr() < value.data_ptr() + size
    ):
        raise ValueError("Vision softmax output must not overlap its input")
    return out.copy_(softmax_eager(value))


_library.define("vision_softmax(Tensor value) -> Tensor")
_library.impl("vision_softmax", softmax_eager, "PrivateUse1")
_library.impl("vision_softmax", softmax_meta, "Meta")
_library.define("vision_softmax.out(Tensor value, *, Tensor(a!) out) -> Tensor(a!)")
_library.impl("vision_softmax.out", softmax_out_eager, "PrivateUse1")
_library.impl("vision_softmax.out", softmax_out_meta, "Meta")
VISION_SOFTMAX = torch.ops.executorch_musa.vision_softmax.default
VISION_SOFTMAX_OUT = torch.ops.executorch_musa.vision_softmax.out


@lowering.register_lowering(VISION_SOFTMAX, type_promotion_kind=None)
def lower_softmax(value):
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
            constant_args=[],
            python_kernel_name="executorch_musa.vision_softmax.out",
            cpp_kernel_name="aoti_torch_musa_vision_softmax_out",
            op_overload=VISION_SOFTMAX_OUT,
        )
    )
