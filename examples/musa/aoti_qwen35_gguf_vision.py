"""Static-grid Qwen3.5 GGUF vision with original BF16/F32 storage on MUSA."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn


@dataclass(frozen=True)
class VisionConfig:
    width: int = 1024
    intermediate: int = 4096
    depth: int = 24
    heads: int = 16
    patch_size: int = 16
    merge_size: int = 2
    position_side: int = 48
    output_width: int = 2560
    epsilon: float = 1e-6


class VisionLinear(nn.Module):
    """Keep BF16 matrix bytes and FP32 biases; compute with FP32 activations."""

    def __init__(self, input_width, output_width):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(output_width, input_width, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(output_width, dtype=torch.float32), requires_grad=False
        )

    def forward(self, hidden_states):
        return functional.linear(hidden_states, self.weight.float(), self.bias)


class SplitPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        shape = (config.width, 3, config.patch_size, config.patch_size)
        self.weight_first = nn.Parameter(torch.empty(shape), requires_grad=False)
        self.weight_second = nn.Parameter(torch.empty(shape), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(config.width), requires_grad=False)

    def forward(self, patches):
        patches = patches.reshape(-1, 3, 2, self.patch_size, self.patch_size)
        weight = torch.stack((self.weight_first, self.weight_second), dim=2)
        return functional.conv3d(
            patches, weight, self.bias, stride=(2, self.patch_size, self.patch_size)
        ).reshape(patches.shape[0], -1)


class VisionAttention(nn.Module):
    def __init__(self, config, lengths):
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.lengths = lengths
        self.qkv = VisionLinear(config.width, config.width * 3)
        self.proj = VisionLinear(config.width, config.width)

    def forward(self, hidden_states, cosine, sine):
        length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(length, 3, self.heads, self.head_width)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query_first, query_second = query.chunk(2, dim=-1)
        key_first, key_second = key.chunk(2, dim=-1)
        query = query * cosine + torch.cat((-query_second, query_first), -1) * sine
        key = key * cosine + torch.cat((-key_second, key_first), -1) * sine
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        results = []
        offset = 0
        for chunk_length in self.lengths:
            stop = offset + chunk_length
            scores = torch.matmul(
                query[:, offset:stop], key[:, offset:stop].transpose(-1, -2)
            ) * (self.head_width**-0.5)
            probability = functional.softmax(scores, dim=-1, dtype=torch.float32)
            result = torch.matmul(probability, value[:, offset:stop])
            results.append(result.transpose(0, 1).reshape(chunk_length, -1))
            offset = stop
        return self.proj(torch.cat(results, dim=0))


class VisionBlock(nn.Module):
    def __init__(self, config, lengths):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width, eps=config.epsilon)
        self.norm2 = nn.LayerNorm(config.width, eps=config.epsilon)
        self.attn = VisionAttention(config, lengths)
        self.fc1 = VisionLinear(config.width, config.intermediate)
        self.fc2 = VisionLinear(config.intermediate, config.width)

    def forward(self, hidden_states, cosine, sine):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cosine, sine
        )
        return hidden_states + self.fc2(
            functional.gelu(self.fc1(self.norm2(hidden_states)), approximate="tanh")
        )


def vision_geometry(config, grids):
    if (
        config.width % config.heads
        or config.width // config.heads % 4
        or config.merge_size < 1
        or config.position_side < 2
    ):
        raise ValueError("Invalid vision head or spatial dimensions")
    if not grids or any(
        len(grid) != 3
        or any(type(size) is not int or size <= 0 for size in grid)
        or grid[0] != 1
        or grid[1] % config.merge_size
        or grid[2] % config.merge_size
        for grid in grids
    ):
        raise ValueError(
            "Vision export requires static still-image grids divisible by merge size"
        )
    indices, weights, coordinates = [], [], []
    for _, height, width in grids:
        row = torch.linspace(0, config.position_side - 1, height, device="cpu")
        column = torch.linspace(0, config.position_side - 1, width, device="cpu")
        row_floor, column_floor = row.int(), column.int()
        row_ceil = (row_floor + 1).clamp(max=config.position_side - 1)
        column_ceil = (column_floor + 1).clamp(max=config.position_side - 1)
        row_fraction, column_fraction = row - row_floor, column - column_floor
        local_indices = torch.stack(
            [
                (row_floor[:, None] * config.position_side + column_floor).flatten(),
                (row_floor[:, None] * config.position_side + column_ceil).flatten(),
                (row_ceil[:, None] * config.position_side + column_floor).flatten(),
                (row_ceil[:, None] * config.position_side + column_ceil).flatten(),
            ]
        ).long()
        local_weights = torch.stack(
            [
                ((1 - row_fraction[:, None]) * (1 - column_fraction)).flatten(),
                ((1 - row_fraction[:, None]) * column_fraction).flatten(),
                (row_fraction[:, None] * (1 - column_fraction)).flatten(),
                (row_fraction[:, None] * column_fraction).flatten(),
            ]
        )
        order = (
            torch.arange(height * width, device="cpu")
            .reshape(
                height // config.merge_size,
                config.merge_size,
                width // config.merge_size,
                config.merge_size,
            )
            .permute(0, 2, 1, 3)
            .flatten()
        )
        indices.append(local_indices[:, order])
        weights.append(local_weights[:, order])
        coordinates.append(torch.stack((order // width, order % width), dim=-1))
    return torch.cat(indices, dim=1), torch.cat(weights, dim=1), torch.cat(coordinates)


class GgufVision(nn.Module):
    def __init__(self, config, grids):
        super().__init__()
        self.config = config
        self.grids = tuple(tuple(grid) for grid in grids)
        indices, weights, coordinates = vision_geometry(config, self.grids)
        self.lengths = tuple(height * width for _, height, width in self.grids)
        self.patch_embed = SplitPatchEmbed(config)
        self.position_weight = nn.Parameter(
            torch.empty(config.position_side**2, config.width), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            VisionBlock(config, self.lengths) for _ in range(config.depth)
        )
        self.merger_norm = nn.LayerNorm(config.width, eps=config.epsilon)
        merged_width = config.width * config.merge_size**2
        self.merger_fc1 = VisionLinear(merged_width, merged_width)
        self.merger_fc2 = VisionLinear(merged_width, config.output_width)
        self.register_buffer("position_indices", indices)
        self.register_buffer("position_weights", weights)
        self.register_buffer("coordinates", coordinates)
        half_head = config.width // config.heads // 2
        self.register_buffer(
            "inverse_frequency",
            1.0
            / (
                10000
                ** (torch.arange(0, half_head, 2, device="cpu").float() / half_head)
            ),
        )

    def forward(self, patches):
        if patches.dtype != torch.float32 or patches.shape != (
            sum(self.lengths),
            6 * self.config.patch_size**2,
        ):
            raise ValueError("Expected FP32 patches matching the fixed image grids")
        hidden_states = self.patch_embed(patches)
        positions = functional.embedding(self.position_indices, self.position_weight)
        positions = positions * self.position_weights[:, :, None]
        hidden_states = hidden_states + (
            positions[0] + positions[1] + positions[2] + positions[3]
        )
        frequencies = (
            self.coordinates.float()[:, :, None] * self.inverse_frequency
        ).flatten(1)
        frequencies = torch.cat((frequencies, frequencies), dim=-1)
        cosine, sine = frequencies.cos()[:, None, :], frequencies.sin()[:, None, :]
        for block in self.blocks:
            hidden_states = block(hidden_states, cosine, sine)
        merged = self.merger_norm(hidden_states).reshape(
            -1, self.config.width * self.config.merge_size**2
        )
        merged = self.merger_fc2(functional.gelu(self.merger_fc1(merged)))
        return merged


def vision_tensor_mapping(depth=24):
    mapping = {
        "v.patch_embd.weight": "patch_embed.weight_first",
        "v.patch_embd.weight.1": "patch_embed.weight_second",
        "v.patch_embd.bias": "patch_embed.bias",
        "v.position_embd.weight": "position_weight",
        "v.post_ln.weight": "merger_norm.weight",
        "v.post_ln.bias": "merger_norm.bias",
        "mm.0.weight": "merger_fc1.weight",
        "mm.0.bias": "merger_fc1.bias",
        "mm.2.weight": "merger_fc2.weight",
        "mm.2.bias": "merger_fc2.bias",
    }
    names = {
        "attn_qkv": "attn.qkv",
        "attn_out": "attn.proj",
        "ffn_up": "fc1",
        "ffn_down": "fc2",
        "ln1": "norm1",
        "ln2": "norm2",
    }
    for index in range(depth):
        for source, target in names.items():
            for suffix in ("weight", "bias"):
                mapping[f"v.blk.{index}.{source}.{suffix}"] = (
                    f"blocks.{index}.{target}.{suffix}"
                )
    return mapping


def load_gguf_vision(path, grids):
    import gguf

    path = Path(path).resolve()
    before = path.stat()
    reader = gguf.GGUFReader(path)
    metadata = {name: field.contents() for name, field in reader.fields.items()}
    expected = {
        "general.architecture": "clip",
        "general.type": "mmproj",
        "clip.projector_type": "qwen3vl_merger",
        "clip.has_vision_encoder": True,
        "clip.vision.embedding_length": 1024,
        "clip.vision.feed_forward_length": 4096,
        "clip.vision.block_count": 24,
        "clip.vision.attention.head_count": 16,
        "clip.vision.patch_size": 16,
        "clip.vision.image_size": 768,
        "clip.vision.spatial_merge_size": 2,
        "clip.vision.projection_dim": 2560,
        "clip.vision.is_deepstack_layers": [False] * 24,
        "clip.use_gelu": True,
        "clip.vision.image_mean": [0.5] * 3,
        "clip.vision.image_std": [0.5] * 3,
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError("Vision GGUF metadata does not match Qwen3.5-4B")
    epsilon = metadata.get("clip.vision.attention.layer_norm_epsilon", 0)
    if not math.isfinite(epsilon) or abs(epsilon - 1e-6) > 1e-12:
        raise ValueError("Unsupported vision LayerNorm epsilon")
    with torch.device("meta"):
        model = GgufVision(VisionConfig(epsilon=epsilon), grids)
    mapping = vision_tensor_mapping()
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(reader.tensors) or set(tensors) != set(mapping):
        raise ValueError("Vision GGUF must contain exactly the original 298 tensors")
    state = dict(model.named_buffers())
    parameters = dict(model.named_parameters())
    records = []
    for name, target in mapping.items():
        tensor, parameter = tensors[name], parameters[target]
        shape = tuple(reversed(tensor.shape.tolist()))
        encoding = "BF16" if parameter.dtype == torch.bfloat16 else "F32"
        if shape != tuple(parameter.shape) or tensor.tensor_type.name != encoding:
            raise ValueError(f"Vision shape or precision mismatch: {name}")
        data = np.array(tensor.data, copy=True)
        if data.nbytes != parameter.numel() * parameter.element_size():
            raise ValueError(f"Malformed vision tensor bytes: {name}")
        value = torch.from_numpy(data)
        if encoding == "BF16":
            value = value.view(torch.bfloat16)
        state[target] = value.reshape(shape).to("musa")
        records.append(
            dict(
                name=name,
                target=target,
                encoding=encoding,
                shape=list(shape),
                nbytes=data.nbytes,
                sha256=hashlib.sha256(data.tobytes()).hexdigest(),
            )
        )
    model.load_state_dict(state, strict=True, assign=True)
    model = model.eval().requires_grad_(False).to("musa")
    if any(
        value.is_meta or value.device.type != "musa"
        for value in (*model.parameters(), *model.buffers())
    ):
        raise ValueError("Vision model has uninitialized or non-MUSA tensors")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("GGUF changed during vision loading")
    return model, dict(
        source=str(path),
        tensors=records,
        grids=model.grids,
        tensor_counts=dict(Counter(record["encoding"] for record in records)),
        activation_dtype="float32",
        weights_preserved=True,
        temporal_frames=1,
        video_supported=False,
    )
