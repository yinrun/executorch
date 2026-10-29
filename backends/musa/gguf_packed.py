"""Opaque packed GGUF linear/embedding operations for the native MUSA runtime."""

from pathlib import Path

import torch
from torch._inductor import ir, lowering

from executorch.backends.musa.gguf import dequantize_gguf, gguf_shape


HEADER = str(Path(__file__).resolve().parents[1] / "cuda/runtime/shims/musa_gguf.h")
QUANT_IDS = {"q4_k": 12, "q5_k": 13, "q6_k": 14}
QUANT_NAMES = {value: name for name, value in QUANT_IDS.items()}
_library = torch.library.Library("executorch_musa", "FRAGMENT")


def packed_shape(value, packed, quant_type, embedding):
    if quant_type not in QUANT_NAMES:
        raise ValueError(f"Unsupported GGUF quantization ID: {quant_type}")
    rows, columns = gguf_shape(packed, QUANT_NAMES[quant_type], torch.float32)
    torch._check(value.device == packed.device)
    torch._check(value.dtype == (torch.int64 if embedding else torch.float32))
    torch._check(1 <= value.ndim <= 4 and value.is_contiguous())
    for size in value.shape:
        torch._check(size > 0)
    if embedding:
        shape = (*value.shape, columns)
    else:
        torch._check(value.shape[-1] == columns)
        shape = (*value.shape[:-1], rows)
    elements = 1
    for size in shape:
        elements *= size
    torch._check(elements <= 2**31 - 256)
    return shape


def linear_meta(value, packed, quant_type):
    return value.new_empty(packed_shape(value, packed, quant_type, False))


def embedding_meta(value, packed, quant_type):
    return packed.new_empty(
        packed_shape(value, packed, quant_type, True), dtype=torch.float32
    )


def linear_eager(value, packed, quant_type):
    packed_shape(value, packed, quant_type, False)
    return torch.nn.functional.linear(
        value, dequantize_gguf(packed, QUANT_NAMES[quant_type])
    )


def embedding_eager(value, packed, quant_type):
    shape = packed_shape(value, packed, quant_type, True)
    if ((value < 0) | (value >= packed.shape[0])).any().item():
        raise IndexError("GGUF embedding index is outside the vocabulary")
    selected = torch.index_select(packed, 0, value.reshape(-1))
    return dequantize_gguf(selected, QUANT_NAMES[quant_type]).reshape(shape)


def check_out(value, packed, quant_type, out, embedding):
    torch._check(
        tuple(out.shape) == tuple(packed_shape(value, packed, quant_type, embedding))
    )
    torch._check(out.device == packed.device and out.dtype == torch.float32)
    torch._check(out.is_contiguous())
    return out


def linear_out_meta(value, packed, quant_type, *, out):
    return check_out(value, packed, quant_type, out, False)


def embedding_out_meta(value, packed, quant_type, *, out):
    return check_out(value, packed, quant_type, out, True)


def packed_out(value, packed, quant_type, out, embedding):
    check_out(value, packed, quant_type, out, embedding)
    begin = out.data_ptr()
    end = begin + out.numel() * out.element_size()
    for source in (value, packed):
        if (
            begin < source.data_ptr() + source.numel() * source.element_size()
            and source.data_ptr() < end
        ):
            raise ValueError("GGUF output must not overlap its inputs")
    result = (
        embedding_eager(value, packed, quant_type)
        if embedding
        else linear_eager(value, packed, quant_type)
    )
    return out.copy_(result)


def linear_out_eager(value, packed, quant_type, *, out):
    return packed_out(value, packed, quant_type, out, False)


def embedding_out_eager(value, packed, quant_type, *, out):
    return packed_out(value, packed, quant_type, out, True)


for name, eager, meta, out_eager, out_meta in (
    ("gguf_linear", linear_eager, linear_meta, linear_out_eager, linear_out_meta),
    (
        "gguf_embedding",
        embedding_eager,
        embedding_meta,
        embedding_out_eager,
        embedding_out_meta,
    ),
):
    _library.define(f"{name}(Tensor value, Tensor packed, int quant_type) -> Tensor")
    _library.impl(name, eager, "PrivateUse1")
    _library.impl(name, meta, "Meta")
    _library.define(
        f"{name}.out(Tensor value, Tensor packed, int quant_type, *, Tensor(a!) out) -> Tensor(a!)"
    )
    _library.impl(f"{name}.out", out_eager, "PrivateUse1")
    _library.impl(f"{name}.out", out_meta, "Meta")

GGUF_LINEAR = torch.ops.executorch_musa.gguf_linear.default
GGUF_EMBEDDING = torch.ops.executorch_musa.gguf_embedding.default


def lower_packed(value, packed, quant_type, embedding):
    value.realize()
    packed.realize()
    shape = list(value.get_size())
    if embedding:
        block_bytes = {12: 144, 13: 176, 14: 210}[quant_type]
        shape.append(packed.get_size()[1] // block_bytes * 256)
    else:
        shape[-1] = packed.get_size()[0]
    layout = ir.FixedLayout(
        value.get_device(),
        torch.float32,
        shape,
        ir.FlexibleLayout.contiguous_strides(shape),
    )
    name = "gguf_embedding" if embedding else "gguf_linear"
    return ir.TensorBox.create(
        ir.ExternKernelOut(
            layout=layout,
            inputs=[value, packed],
            constant_args=[quant_type],
            python_kernel_name=f"executorch_musa.{name}.out",
            cpp_kernel_name=f"aoti_torch_musa_{name}_out",
            op_overload=getattr(torch.ops.executorch_musa, name).out,
        )
    )


@lowering.register_lowering(GGUF_LINEAR, type_promotion_kind=None)
def lower_linear(value, packed, quant_type):
    return lower_packed(value, packed, quant_type, False)


@lowering.register_lowering(GGUF_EMBEDDING, type_promotion_kind=None)
def lower_embedding(value, packed, quant_type):
    return lower_packed(value, packed, quant_type, True)


class GgufLinear(torch.nn.Module):
    def __init__(self, packed, quant_type, bias=None):
        super().__init__()
        if quant_type not in QUANT_IDS:
            raise ValueError(f"Unsupported GGUF quantization: {quant_type}")
        self.register_buffer("packed_weight", packed)
        self.register_buffer("bias", bias)
        self.quant_type = QUANT_IDS[quant_type]

    def forward(self, value):
        result = GGUF_LINEAR(value.contiguous(), self.packed_weight, self.quant_type)
        return result if self.bias is None else result + self.bias


class GgufEmbedding(torch.nn.Module):
    def __init__(self, packed, quant_type):
        super().__init__()
        if quant_type not in QUANT_IDS:
            raise ValueError(f"Unsupported GGUF quantization: {quant_type}")
        self.register_buffer("packed_weight", packed)
        self.quant_type = QUANT_IDS[quant_type]

    def forward(self, indices):
        return GGUF_EMBEDDING(indices.contiguous(), self.packed_weight, self.quant_type)
