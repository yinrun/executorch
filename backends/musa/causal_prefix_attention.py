"""Explicit host prefix snapshots shared by valid-prefix GPU attention calls."""

import torch
from torch._inductor import ir, lowering

_library = torch.library.Library("executorch_musa", "FRAGMENT")


def validate_prefix_inputs(position, mask):
    torch._check(position.dtype == torch.int64 and tuple(position.shape) == (1,))
    torch._check(mask.dtype == torch.bool and mask.ndim == 2)
    torch._check(mask.shape[0] == 1 and 0 < mask.shape[1] <= 4096)
    torch._check(position.device == mask.device)
    torch._check(position.device.type in ("musa", "meta"))
    torch._check(position.is_contiguous() and mask.is_contiguous())


def causal_prefix_impl(position, mask):
    validate_prefix_inputs(position, mask)
    index = position.cpu().item()
    capacity = mask.shape[1]
    if not 0 <= index < capacity:
        raise ValueError("Position is outside the KV cache")
    if not torch.equal(mask.cpu()[0], torch.arange(capacity) <= index):
        raise ValueError("Attention mask must describe the valid causal prefix")
    return torch.tensor([index + 1, capacity], dtype=torch.int64, device="cpu")


def causal_prefix_meta(position, mask):
    validate_prefix_inputs(position, mask)
    return torch.empty((2,), dtype=torch.int64, device="cpu")


def validate_host_prefix(control):
    torch._check(control.device.type == "cpu" and control.dtype == torch.int64)
    torch._check(tuple(control.shape) == (2,) and control.is_contiguous())


def causal_prefix_out_meta(position, mask, *, out):
    validate_prefix_inputs(position, mask)
    validate_host_prefix(out)
    return out


def causal_prefix_out_impl(position, mask, *, out):
    causal_prefix_out_meta(position, mask, out=out)
    return out.copy_(causal_prefix_impl(position, mask))


def validate_attention_inputs(query, key, value, control):
    validate_host_prefix(control)
    torch._check(query.ndim == key.ndim == value.ndim == 4)
    grouped = query.dtype == torch.float32
    torch._check(tuple(query.shape) == (1, 16 if grouped else 8, 1, 256))
    torch._check(key.shape[0] == 1 and key.shape[1] == (4 if grouped else 8))
    torch._check(key.shape[3] == 256 and value.shape == key.shape)
    torch._check(
        0 < key.shape[2] <= 4096 if grouped else key.shape[2] in (64, 128, 256)
    )
    torch._check(query.dtype in (torch.float32, torch.bfloat16))
    torch._check(query.device.type in ("musa", "meta"))
    for tensor in (key, value):
        torch._check(tensor.device == query.device and tensor.dtype == query.dtype)


def prefix_attention_impl(query, key, value, control):
    validate_attention_inputs(query, key, value, control)
    length, capacity = control.tolist()
    if capacity != key.shape[2] or not 0 < length <= capacity:
        raise ValueError("Causal prefix length and capacity must match the KV cache")
    shape = query.shape
    key = key[..., :length, :].contiguous()
    value = value[..., :length, :].contiguous()
    if query.dtype == torch.float32:
        query = query.reshape(4, 4, 256)
        key = key.reshape(4, length, 256)
        value = value.reshape(4, length, 256)
    scores = torch.matmul(query.contiguous(), key.transpose(-2, -1)) * 0.0625
    probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    return torch.matmul(probabilities, value).reshape(shape).contiguous()


def prefix_attention_meta(query, key, value, control):
    validate_attention_inputs(query, key, value, control)
    return torch.empty_like(query, memory_format=torch.contiguous_format)


def prefix_attention_out_meta(query, key, value, control, *, out):
    validate_attention_inputs(query, key, value, control)
    torch._check(out.shape == query.shape and out.dtype == query.dtype)
    torch._check(out.device == query.device and out.is_contiguous())
    return out


def prefix_attention_out_impl(query, key, value, control, *, out):
    prefix_attention_out_meta(query, key, value, control, out=out)
    for tensor in (query, key, value):
        span = 1 + sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
        if (
            out.data_ptr() < tensor.data_ptr() + span * tensor.element_size()
            and tensor.data_ptr() < out.data_ptr() + out.nbytes
        ):
            raise ValueError("Attention output must not overlap inputs")
    return out.copy_(prefix_attention_impl(query, key, value, control))


try:
    CAUSAL_PREFIX = torch.ops.executorch_musa.causal_prefix.default
except AttributeError:
    _library.define("causal_prefix(Tensor position, Tensor mask) -> Tensor")
    _library.impl("causal_prefix", causal_prefix_impl, "PrivateUse1")
    torch.library.register_fake("executorch_musa::causal_prefix")(causal_prefix_meta)
    CAUSAL_PREFIX = torch.ops.executorch_musa.causal_prefix.default

try:
    CAUSAL_PREFIX_OUT = torch.ops.executorch_musa.causal_prefix.out
except AttributeError:
    _library.define(
        "causal_prefix.out(Tensor position, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)"
    )
    _library.impl("causal_prefix.out", causal_prefix_out_impl, "PrivateUse1")
    torch.library.register_fake("executorch_musa::causal_prefix.out")(
        causal_prefix_out_meta
    )
    CAUSAL_PREFIX_OUT = torch.ops.executorch_musa.causal_prefix.out

try:
    PREFIX_ATTENTION = torch.ops.executorch_musa.prefix_attention.default
except AttributeError:
    _library.define(
        "prefix_attention(Tensor query, Tensor key, Tensor value, Tensor control) -> Tensor"
    )
    _library.impl("prefix_attention", prefix_attention_impl, "PrivateUse1")
    torch.library.register_fake("executorch_musa::prefix_attention")(
        prefix_attention_meta
    )
    PREFIX_ATTENTION = torch.ops.executorch_musa.prefix_attention.default

try:
    PREFIX_ATTENTION_OUT = torch.ops.executorch_musa.prefix_attention.out
except AttributeError:
    _library.define(
        "prefix_attention.out(Tensor query, Tensor key, Tensor value, Tensor control, *, Tensor(a!) out) -> Tensor(a!)"
    )
    _library.impl("prefix_attention.out", prefix_attention_out_impl, "PrivateUse1")
    torch.library.register_fake("executorch_musa::prefix_attention.out")(
        prefix_attention_out_meta
    )
    PREFIX_ATTENTION_OUT = torch.ops.executorch_musa.prefix_attention.out


@lowering.register_lowering(CAUSAL_PREFIX, type_promotion_kind=None)
def lower_causal_prefix(position, mask):
    inputs = [ir.ExternKernel.require_contiguous(tensor) for tensor in (position, mask)]
    for tensor in inputs:
        tensor.realize()
    return ir.TensorBox.create(
        ir.ExternKernelOut(
            layout=ir.FixedLayout(torch.device("cpu"), torch.int64, [2], [1]),
            inputs=inputs,
            constant_args=[],
            python_kernel_name="executorch_musa.causal_prefix.out",
            cpp_kernel_name="aoti_torch_musa_causal_prefix_out",
            op_overload=CAUSAL_PREFIX_OUT,
        )
    )


@lowering.register_lowering(PREFIX_ATTENTION, type_promotion_kind=None)
def lower_prefix_attention(query, key, value, control):
    inputs = [
        ir.ExternKernel.require_contiguous(tensor)
        for tensor in (query, key, value, control)
    ]
    for tensor in inputs:
        tensor.realize()
    return ir.TensorBox.create(
        ir.ExternKernelOut(
            layout=ir.FixedLayout(
                query.get_device(),
                query.get_dtype(),
                query.get_size(),
                ir.FlexibleLayout.contiguous_strides(query.get_size()),
            ),
            inputs=inputs,
            constant_args=[],
            python_kernel_name="executorch_musa.prefix_attention.out",
            cpp_kernel_name="aoti_torch_musa_prefix_attention_out",
            op_overload=PREFIX_ATTENTION_OUT,
        )
    )
