"""Opt-in MUSA Q6 layout v1: 256 signed levels and 16 FP32 scales per block.

This 320-byte layout is not GGUF. It preserves Q6_K dequantized FP32 values,
uses no dense floating-point weight storage, and is prepared before export.
"""

import torch

from executorch.backends.musa.gguf import decode_fp16_scale, gguf_shape


Q6_PREDECODED = "musa_q6_predecoded_v1"
Q6_PREDECODED_ID = 114
Q6_PREDECODED_BYTES = 320


def predecode_q6(raw):
    rows, columns = gguf_shape(raw, "q6_k", torch.float32)
    if raw.device.type != "musa":
        raise ValueError("Q6 preprocessing requires a real MUSA tensor")
    result = torch.empty(
        (rows, columns // 256 * Q6_PREDECODED_BYTES),
        dtype=torch.uint8,
        device=raw.device,
    )
    for start in range(0, rows, 4096):
        source = raw[start : start + 4096].reshape(-1, 210)
        low = source[:, :128].reshape(-1, 2, 1, 64)
        low = torch.cat((low & 15, low >> 4), dim=2).reshape(-1, 256)
        high = source[:, 128:192].reshape(-1, 2, 1, 32)
        high = torch.cat(
            tuple((high >> shift) & 3 for shift in (0, 2, 4, 6)), dim=2
        ).reshape(-1, 256)
        levels = ((low | (high << 4)).to(torch.int16) - 32).to(torch.int8)
        scales = source[:, 192:208].contiguous().view(torch.int8).float()
        scales = (decode_fp16_scale(source, 208) * scales).contiguous()
        target = result[start : start + 4096].view(-1, Q6_PREDECODED_BYTES)
        target[:, :256].copy_(levels.view(torch.uint8))
        target[:, 256:].copy_(scales.view(torch.uint8))
    return result


def predecoded_shape(raw):
    torch._check(raw.device.type in ("musa", "meta"))
    torch._check(raw.dtype == torch.uint8 and raw.ndim == 2)
    torch._check(raw.is_contiguous())
    torch._check(raw.shape[0] > 0 and raw.shape[1] > 0)
    torch._check(raw.shape[1] % Q6_PREDECODED_BYTES == 0)
    return raw.shape[0], raw.shape[1] // Q6_PREDECODED_BYTES * 256


def dequantize_predecoded_q6(raw):
    rows, columns = predecoded_shape(raw)
    blocks = raw.reshape(-1, Q6_PREDECODED_BYTES)
    levels = blocks[:, :256].contiguous().view(torch.int8).float()
    scales = blocks[:, 256:].contiguous().view(torch.float32)
    return (scales.unsqueeze(-1) * levels.reshape(-1, 16, 16)).reshape(rows, columns)
