"""Opt-in single-token FP32 GDN recurrence for the native MUSA runtime."""

from pathlib import Path

import torch
from torch._inductor import ir, lowering


HEADER = str(Path(__file__).resolve().parents[1] / "cuda/runtime/shims/musa_gdn.h")
_library = torch.library.Library("executorch_musa", "FRAGMENT")


def recurrence_shape(query, key, value, decay, beta, state):
    torch._check(query.ndim == 4)
    heads = query.shape[1]
    torch._check(heads in (16, 32))
    shapes = ((1, heads, 1, 128),) * 3 + ((1, heads, 1),) * 2 + ((1, heads, 128, 128),)
    for tensor, shape in zip((query, key, value, decay, beta, state), shapes):
        torch._check(tensor.device == query.device)
        torch._check(tensor.device.type in ("musa", "meta"))
        torch._check(tensor.dtype == torch.float32)
        torch._check(tuple(tensor.shape) == shape and tensor.is_contiguous())
    return (heads * (128 * 128 + 128),)


def recurrence_meta(query, key, value, decay, beta, state):
    return query.new_empty(recurrence_shape(query, key, value, decay, beta, state))


def recurrence_eager(query, key, value, decay, beta, state):
    recurrence_shape(query, key, value, decay, beta, state)
    scaled = state * decay.unsqueeze(-1)
    memory = (scaled * key[:, :, 0].unsqueeze(-1)).sum(-2)
    delta = (value[:, :, 0] - memory) * beta
    updated = scaled + key[:, :, 0].unsqueeze(-1) * delta.unsqueeze(-2)
    output = (updated * query[:, :, 0].unsqueeze(-1)).sum(-2)
    return torch.cat((updated.reshape(-1), output.reshape(-1)))


def recurrence_out_meta(query, key, value, decay, beta, state, *, out):
    torch._check(
        tuple(out.shape) == recurrence_shape(query, key, value, decay, beta, state)
    )
    torch._check(out.device == query.device and out.dtype == query.dtype)
    torch._check(out.is_contiguous())
    return out


def recurrence_out_eager(query, key, value, decay, beta, state, *, out):
    recurrence_out_meta(query, key, value, decay, beta, state, out=out)
    for tensor in (query, key, value, decay, beta, state):
        if (
            out.data_ptr() < tensor.data_ptr() + tensor.numel() * tensor.element_size()
            and tensor.data_ptr() < out.data_ptr() + out.numel() * out.element_size()
        ):
            raise ValueError("GDN recurrence output must not overlap inputs")
    return out.copy_(recurrence_eager(query, key, value, decay, beta, state))


_arguments = (
    "Tensor query, Tensor key, Tensor value, Tensor decay, Tensor beta, Tensor state"
)
_library.define(f"gdn_recurrence({_arguments}) -> Tensor")
_library.impl("gdn_recurrence", recurrence_eager, "PrivateUse1")
_library.impl("gdn_recurrence", recurrence_meta, "Meta")
_library.define(f"gdn_recurrence.out({_arguments}, *, Tensor(a!) out) -> Tensor(a!)")
_library.impl("gdn_recurrence.out", recurrence_out_eager, "PrivateUse1")
_library.impl("gdn_recurrence.out", recurrence_out_meta, "Meta")
GDN_RECURRENCE = torch.ops.executorch_musa.gdn_recurrence.default


@lowering.register_lowering(GDN_RECURRENCE, type_promotion_kind=None)
def lower_recurrence(query, key, value, decay, beta, state):
    inputs = [query, key, value, decay, beta, state]
    for tensor in inputs:
        tensor.realize()
    shape = [query.get_size()[1] * (128 * 128 + 128)]
    layout = ir.FixedLayout(query.get_device(), torch.float32, shape, [1])
    return ir.TensorBox.create(
        ir.ExternKernelOut(
            layout=layout,
            inputs=inputs,
            constant_args=[],
            python_kernel_name="executorch_musa.gdn_recurrence.out",
            cpp_kernel_name="aoti_torch_musa_gdn_recurrence_out",
            op_overload=torch.ops.executorch_musa.gdn_recurrence.out,
        )
    )
