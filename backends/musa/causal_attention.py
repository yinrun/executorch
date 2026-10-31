"""Fixed-storage, valid-prefix Qwen attention for single-token decoding."""

import os

import torch
from torch._inductor import ir, lowering


HEADER = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "cuda", "runtime", "shims",
        "musa_causal_attention.h",
    )
)
_library = torch.library.Library("executorch_musa", "FRAGMENT")


def validate_metadata(query, key, value, position, mask):
    torch._check(query.ndim == key.ndim == value.ndim == 4)
    grouped_fp32 = query.dtype == torch.float32
    torch._check(tuple(query.shape) == (1, 16 if grouped_fp32 else 8, 1, 256))
    torch._check(
        key.shape[0] == 1 and key.shape[1] == (4 if grouped_fp32 else 8)
        and key.shape[3] == 256
    )
    torch._check(
        0 < key.shape[2] <= 4096 if grouped_fp32 else key.shape[2] in (64, 128, 256)
    )
    torch._check(value.shape == key.shape)
    torch._check(tuple(position.shape) == (1,) and position.dtype == torch.int64)
    torch._check(tuple(mask.shape) == (1, key.shape[2]) and mask.dtype == torch.bool)
    for tensor in (query, key, value):
        torch._check(tensor.dtype == query.dtype)
    torch._check(query.dtype in (torch.bfloat16, torch.float32))
    for tensor in (query, key, value, position, mask):
        torch._check(tensor.device == query.device)
    torch._check(query.device.type in ("musa", "meta"))


def causal_attention_impl(query, key, value, position, mask):
    validate_metadata(query, key, value, position, mask)
    index = position.item()
    if not 0 <= index < key.shape[2]:
        raise ValueError("Position is outside the KV cache")
    expected_mask = torch.arange(key.shape[2], device=key.device) <= index
    if not torch.equal(mask[0], expected_mask):
        raise ValueError("Attention mask must describe the valid causal prefix")
    length = index + 1
    key = key[..., :length, :].contiguous()
    value = value[..., :length, :].contiguous()
    query_shape = query.shape
    if query.dtype == torch.float32:
        query = query.reshape(4, 4, 256)
        key = key.reshape(4, length, 256)
        value = value.reshape(4, length, 256)
    scores = torch.matmul(query.contiguous(), key.transpose(-2, -1))
    scaled = scores * 0.0625
    probabilities = torch.softmax(scaled.float(), dim=-1).to(query.dtype)
    return torch.matmul(probabilities, value).reshape(query_shape).contiguous()


def causal_attention_meta(query, key, value, position, mask):
    validate_metadata(query, key, value, position, mask)
    return torch.empty_like(query, memory_format=torch.contiguous_format)


def causal_attention_out_meta(query, key, value, position, mask, *, out):
    validate_metadata(query, key, value, position, mask)
    torch._check(out.shape == query.shape)
    torch._check(out.dtype == query.dtype and out.device == query.device)
    torch._check(out.is_contiguous())
    return out


def causal_attention_out_impl(query, key, value, position, mask, *, out):
    causal_attention_out_meta(query, key, value, position, mask, out=out)
    output_begin = out.data_ptr()
    output_end = output_begin + out.numel() * out.element_size()
    for tensor in (query, key, value, position, mask):
        begin = tensor.data_ptr()
        span = 1 + sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
        if output_begin < begin + span * tensor.element_size() and begin < output_end:
            raise ValueError("Attention output must not overlap its inputs")
    return out.copy_(causal_attention_impl(query, key, value, position, mask))


try:
    CAUSAL_ATTENTION = torch.ops.executorch_musa.causal_attention.default
except AttributeError:
    _library.define(
        "causal_attention(Tensor query, Tensor key, Tensor value, "
        "Tensor position, Tensor mask) -> Tensor"
    )
    _library.impl("causal_attention", causal_attention_impl, "PrivateUse1")
    _library.impl("causal_attention", causal_attention_meta, "Meta")
    CAUSAL_ATTENTION = torch.ops.executorch_musa.causal_attention.default

try:
    CAUSAL_ATTENTION_OUT = torch.ops.executorch_musa.causal_attention.out
except AttributeError:
    _library.define(
        "causal_attention.out(Tensor query, Tensor key, Tensor value, "
        "Tensor position, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)"
    )
    _library.impl("causal_attention.out", causal_attention_out_impl, "PrivateUse1")
    _library.impl("causal_attention.out", causal_attention_out_meta, "Meta")
    CAUSAL_ATTENTION_OUT = torch.ops.executorch_musa.causal_attention.out


@lowering.register_lowering(CAUSAL_ATTENTION, type_promotion_kind=None)
def lower_causal_attention(query, key, value, position, mask):
    inputs = [
        ir.ExternKernel.require_contiguous(tensor)
        for tensor in (query, key, value, position, mask)
    ]
    for tensor in inputs:
        tensor.realize()
    layout = ir.FixedLayout(
        query.get_device(), query.get_dtype(), query.get_size(),
        ir.FlexibleLayout.contiguous_strides(query.get_size()),
    )
    return ir.TensorBox.create(
        ir.ExternKernelOut(
            layout=layout,
            inputs=inputs,
            constant_args=[],
            python_kernel_name="executorch_musa.causal_attention.out",
            cpp_kernel_name="aoti_torch_musa_causal_attention_out",
            op_overload=CAUSAL_ATTENTION_OUT,
        )
    )
