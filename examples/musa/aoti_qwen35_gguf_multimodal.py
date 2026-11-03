"""Image-token transport and interleaved MRoPE for the packed text model."""

import torch
from torch import nn

from executorch.backends.musa.causal_prefix_attention import CAUSAL_PREFIX


def image_token_positions(token_ids, grids, image_token_id=248056):
    """Prepare host-side still-image position metadata, not model computations."""
    if not token_ids or any(
        type(token) is not int or not 0 <= token < 248320 for token in token_ids
    ):
        raise ValueError("Expected nonempty Qwen3.5 token IDs")
    if len(token_ids) > 4096:
        raise ValueError("Image prompt exceeds the 4096-position context")
    if any(
        len(grid) != 3
        or any(type(size) is not int or size <= 0 for size in grid)
        or grid[0] != 1
        or grid[1] % 2
        or grid[2] % 2
        for grid in grids
    ):
        raise ValueError("Expected still-image grids divisible by spatial merge two")
    positions = [[], [], []]
    image_rows = []
    current_position, token_offset, image_index, row_offset = 0, 0, 0, 0
    while token_offset < len(token_ids):
        if token_ids[token_offset] != image_token_id:
            for axis in positions:
                axis.append(current_position)
            image_rows.append(-1)
            current_position += 1
            token_offset += 1
            continue
        if image_index >= len(grids):
            raise ValueError("Image tokens lack a corresponding grid")
        _, height, width = grids[image_index]
        merged_height, merged_width = height // 2, width // 2
        count = merged_height * merged_width
        segment = token_ids[token_offset : token_offset + count]
        if len(segment) != count or any(token != image_token_id for token in segment):
            raise ValueError("Image placeholder count does not match the grid")
        if (
            token_offset + count < len(token_ids)
            and token_ids[token_offset + count] == image_token_id
        ):
            raise ValueError("Image groups must have separate text delimiters")
        for row in range(merged_height):
            for column in range(merged_width):
                positions[0].append(current_position)
                positions[1].append(current_position + row)
                positions[2].append(current_position + column)
                image_rows.append(row_offset + row * merged_width + column)
        token_offset += count
        row_offset += count
        current_position += max(merged_height, merged_width)
        image_index += 1
    if image_index != len(grids):
        raise ValueError("Image grids were not all consumed by placeholders")
    return dict(
        position_ids=torch.tensor(positions, dtype=torch.int64, device="cpu"),
        image_rows=image_rows,
        next_position=current_position,
        rope_delta=current_position - len(token_ids),
    )


class InterleavedMRope(nn.Module):
    def __init__(self, device):
        super().__init__()
        inverse = 1.0 / (
            10000000.0 ** (torch.arange(0, 64, 2, device=device).float() / 64)
        )
        axes = [
            1 if index % 3 == 1 else 2 if index % 3 == 2 and index < 30 else 0
            for index in range(32)
        ]
        self.register_buffer("inverse_frequency", inverse)
        self.register_buffer(
            "axes", torch.tensor(axes, dtype=torch.int64, device=device)
        )

    def forward(self, positions):
        if (
            positions.dtype != torch.int64
            or positions.ndim != 2
            or positions.shape[0] != 3
        ):
            raise ValueError("MRoPE requires int64 positions with shape [3, sequence]")
        selected = positions.index_select(0, self.axes).transpose(0, 1).float()
        frequencies = selected * self.inverse_frequency
        frequencies = torch.cat((frequencies, frequencies), dim=-1)
        return frequencies.cos(), frequencies.sin()


class GgufMultimodalStep(nn.Module):
    """Single-token text/image step with separate cache and three-axis positions."""

    def __init__(self, model, share_attention_prefix=False):
        super().__init__()
        params = model.params
        if (
            not model.apply_embedding
            or not model.apply_output
            or not params.use_hf_rope
            or params.rope_freq_base != 10000000.0
            or int(params.head_dim * params.partial_rotary_factor) != 64
        ):
            raise ValueError("Multimodal step requires Qwen3.5 partial HF RoPE")
        self.model = model
        self.share_attention_prefix = share_attention_prefix
        if share_attention_prefix:
            from examples.musa.aoti_qwen35_gguf import GgufSharedPrefixAttention

            consumers = [
                layer.attention
                for layer in model.layers
                if isinstance(layer.attention, GgufSharedPrefixAttention)
            ]
            if len(consumers) != 8:
                raise ValueError(
                    "Shared-prefix wrapper requires eight shared-prefix attention layers"
                )
            self.register_buffer(
                "prefix_columns",
                torch.arange(
                    params.max_context_len, device=next(model.parameters()).device
                ).reshape(1, -1),
            )
        self.mrope = InterleavedMRope(next(model.parameters()).device)

    def forward(self, tokens, input_pos, positions, image_embedding, use_image):
        if (
            tokens.shape != (1, 1)
            or tokens.dtype != torch.int64
            or input_pos.shape != (1,)
            or input_pos.dtype != torch.int64
            or positions.shape != (3, 1)
            or positions.dtype != torch.int64
            or image_embedding.shape != (1, 1, self.model.params.dim)
            or image_embedding.dtype != torch.float32
            or use_image.shape != (1,)
            or use_image.dtype != torch.bool
        ):
            raise ValueError("Invalid single-token multimodal input shape or dtype")
        options = {"input_pos": input_pos}
        if self.share_attention_prefix:
            options["causal_prefix"] = CAUSAL_PREFIX(
                input_pos, self.prefix_columns <= input_pos.reshape(1, 1)
            )
        hidden_states = torch.where(
            use_image.reshape(1, 1, 1),
            image_embedding,
            self.model.tok_embeddings(tokens),
        )
        cosine, sine = self.mrope(positions)
        for layer in self.model.layers:
            hidden_states, update = layer(hidden_states, cosine, sine, options)
            if update is not None:
                raise ValueError("Unexpected attention option update in Qwen3.5")
        logits = self.model.output(self.model.norm(hidden_states[:, -1, :]))
        return logits, logits.reshape(-1).argmax().reshape(1)
