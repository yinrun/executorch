"""GPU-only decoding of unchanged GGUF K-quant blocks for MUSA validation."""

import torch


BLOCK_BYTES = {"q4_k": 144, "q5_k": 176, "q6_k": 210}


def decode_fp16_scale(blocks, offset):
    low = blocks[:, offset : offset + 1].to(torch.int32)
    high = blocks[:, offset + 1 : offset + 2].to(torch.int32)
    bits = low + high * 256
    mantissa = torch.bitwise_and(bits, 1023).float()
    exponent = torch.bitwise_and(torch.bitwise_right_shift(bits, 10), 31)
    magnitude = torch.where(
        exponent == 0,
        mantissa * (2.0**-24),
        torch.ldexp(mantissa + 1024, exponent - 25),
    )
    magnitude = torch.where(
        exponent == 31,
        torch.where(mantissa == 0, float("inf"), float("nan")),
        magnitude,
    )
    return torch.where(torch.bitwise_and(bits, 32768) != 0, -magnitude, magnitude)


def gguf_shape(raw, quant_type, output_dtype):
    if quant_type not in BLOCK_BYTES:
        raise ValueError(f"Unsupported GGUF quantization: {quant_type}")
    if output_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("GGUF output must be FP32, FP16, or BF16")
    torch._check(raw.device.type in ("musa", "meta"))
    torch._check(raw.dtype == torch.uint8 and raw.ndim == 2)
    torch._check(raw.is_contiguous())
    torch._check(raw.shape[0] > 0 and raw.shape[1] > 0)
    block_bytes = BLOCK_BYTES[quant_type]
    torch._check(raw.shape[1] % block_bytes == 0)
    rows = raw.shape[0]
    columns = raw.shape[1] // block_bytes * 256
    return rows, columns


def dequantize_gguf_impl(raw, quant_type, output_dtype=torch.float32):
    rows, columns = gguf_shape(raw, quant_type, output_dtype)
    block_bytes = BLOCK_BYTES[quant_type]
    blocks = raw.reshape(-1, block_bytes)
    if quant_type == "q6_k":
        low = blocks[:, :128].reshape(-1, 2, 1, 64)
        low = torch.cat(
            (torch.bitwise_and(low, 15), torch.bitwise_right_shift(low, 4)), dim=2
        ).reshape(-1, 256)
        high = blocks[:, 128:192].reshape(-1, 2, 1, 32)
        high = torch.cat(
            tuple(
                torch.bitwise_and(torch.bitwise_right_shift(high, shift), 3)
                for shift in (0, 2, 4, 6)
            ),
            dim=2,
        )
        values = (
            torch.bitwise_or(
                low, torch.bitwise_left_shift(high.reshape(-1, 256), 4)
            ).to(torch.int16)
            - 32
        )
        scales = blocks[:, 192:208].to(torch.int16)
        scales = torch.where(scales >= 128, scales - 256, scales).float()
        base = decode_fp16_scale(blocks, 208)
        decoded = (base * scales).unsqueeze(-1) * values.reshape(-1, 16, 16).float()
    else:
        base = decode_fp16_scale(blocks, 0)
        minimum = decode_fp16_scale(blocks, 2)
        packed_scales = blocks[:, 4:16]
        scales = torch.cat(
            (
                torch.bitwise_and(packed_scales[:, :4], 63),
                torch.bitwise_or(
                    torch.bitwise_and(packed_scales[:, 8:12], 15),
                    torch.bitwise_left_shift(
                        torch.bitwise_right_shift(packed_scales[:, :4], 6), 4
                    ),
                ),
            ),
            dim=1,
        ).float()
        minima = torch.cat(
            (
                torch.bitwise_and(packed_scales[:, 4:8], 63),
                torch.bitwise_or(
                    torch.bitwise_right_shift(packed_scales[:, 8:12], 4),
                    torch.bitwise_left_shift(
                        torch.bitwise_right_shift(packed_scales[:, 4:8], 6), 4
                    ),
                ),
            ),
            dim=1,
        ).float()
        offset = 16 if quant_type == "q4_k" else 48
        packed_values = blocks[:, offset:].reshape(-1, 4, 1, 32)
        values = torch.cat(
            (
                torch.bitwise_and(packed_values, 15),
                torch.bitwise_right_shift(packed_values, 4),
            ),
            dim=2,
        )
        values = values.reshape(-1, 8, 32)
        if quant_type == "q5_k":
            high = blocks[:, 16:48].unsqueeze(1)
            high = torch.cat(
                tuple(
                    torch.bitwise_and(torch.bitwise_right_shift(high, shift), 1)
                    for shift in range(8)
                ),
                dim=1,
            )
            values = torch.bitwise_or(values, torch.bitwise_left_shift(high, 4))
        decoded = (base * scales).unsqueeze(-1) * values.float()
        decoded = decoded - (minimum * minima).unsqueeze(-1)
    return decoded.reshape(rows, columns).to(output_dtype)


def dequantize_gguf_meta(raw, quant_type, output_dtype):
    return raw.new_empty(gguf_shape(raw, quant_type, output_dtype), dtype=output_dtype)


_library = torch.library.Library("executorch_musa", "FRAGMENT")
_library.define(
    "gguf_decode(Tensor raw, str quant_type, ScalarType output_dtype) -> Tensor"
)
_library.impl("gguf_decode", dequantize_gguf_impl, "PrivateUse1")
_library.impl("gguf_decode", dequantize_gguf_meta, "Meta")
GGUF_DECODE = torch.ops.executorch_musa.gguf_decode.default


def dequantize_gguf(raw, quant_type, output_dtype=torch.float32):
    gguf_shape(raw, quant_type, output_dtype)
    return GGUF_DECODE(raw, quant_type, output_dtype)
