"""Transformers-derived same-GGUF GPU reference, not original BF16 weights."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from executorch.backends.musa.gguf import BLOCK_BYTES, dequantize_gguf
from executorch.backends.musa.gguf_packed import GgufEmbedding, GgufLinear, QUANT_NAMES


class ReferenceLinear(GgufLinear):
    """Optional decoded GPU cache for references only, never exported deployment."""

    def __init__(self, packed, quant_type):
        super().__init__(packed, quant_type)
        self.register_buffer("decoded_weight", None)

    def decoded_nbytes(self):
        columns = (
            self.packed_weight.shape[1]
            // BLOCK_BYTES[QUANT_NAMES[self.quant_type]]
            * 256
        )
        return self.packed_weight.shape[0] * columns * 4

    @torch.no_grad()
    def cache_decoded_weight(self):
        if self.decoded_weight is not None:
            raise ValueError("Reference weight is already cached")
        rows = self.packed_weight.shape[0]
        columns = self.decoded_nbytes() // (rows * 4)
        decoded = torch.empty(
            (rows, columns), device=self.packed_weight.device, dtype=torch.float32
        )
        for start in range(0, rows, 4096):
            decoded[start : start + 4096].copy_(
                dequantize_gguf(
                    self.packed_weight[start : start + 4096],
                    QUANT_NAMES[self.quant_type],
                )
            )
        self.decoded_weight = decoded

    def forward(self, hidden_states):
        if self.decoded_weight is None:
            return super().forward(hidden_states)
        outputs = [
            functional.linear(hidden_states, self.decoded_weight[start : start + 4096])
            for start in range(0, self.decoded_weight.shape[0], 4096)
        ]
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)


def cache_reference_linears(model, budget_bytes):
    if not isinstance(budget_bytes, int) or budget_bytes < 0:
        raise ValueError("Reference cache budget must be a nonnegative integer")
    layers = [layer for layer in model.modules() if isinstance(layer, ReferenceLinear)]
    if any(layer.decoded_weight is not None for layer in layers):
        raise ValueError("Reference cache must be configured once on a fresh model")
    layers.sort(key=lambda layer: layer.decoded_nbytes(), reverse=True)
    used, count = 0, 0
    for layer in layers:
        required = layer.decoded_nbytes()
        if required <= budget_bytes - used:
            layer.cache_decoded_weight()
            used += required
            count += 1
    return dict(
        budget_bytes=budget_bytes,
        decoded_bytes=used,
        cached_linears=count,
        total_linears=len(layers),
        reference_only=True,
        native_export_unchanged=True,
    )


class DirectGgufRMSNorm(hf.Qwen3_5RMSNorm):
    def forward(self, hidden_states):
        output = self._norm(hidden_states.float())
        return (output * self.weight.float()).type_as(hidden_states)


class PermutedProjection(torch.nn.Module):
    def __init__(self, projection, permutation, *, input_order=False):
        super().__init__()
        self.projection = projection
        self.register_buffer("permutation", permutation)
        self.input_order = input_order

    def forward(self, hidden_states):
        if self.input_order:
            return self.projection(hidden_states.index_select(-1, self.permutation))
        return self.projection(hidden_states).index_select(-1, self.permutation)


class DirectGgufDeltaNet(hf.Qwen3_5GatedDeltaNet):
    """Keep HF cache/conv/chunk/recurrence logic, consuming stored negative A."""

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        del self.A_log
        self.register_buffer("negative_a", torch.empty(self.num_v_heads))
        self.causal_conv1d_fn = None
        self.causal_conv1d_update = hf.torch_causal_conv1d_update
        self.chunk_gated_delta_rule = hf.torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = hf.torch_recurrent_gated_delta_rule

    def forward(
        self, hidden_states, cache_params=None, cache_position=None, attention_mask=None
    ):
        hidden_states = hf.apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, sequence_length, _ = hidden_states.shape
        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and sequence_length == 1
            and cache_position is not None
        )
        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        gate = self.in_proj_z(hidden_states).reshape(
            batch_size, sequence_length, -1, self.head_v_dim
        )
        beta_input = self.in_proj_b(hidden_states)
        decay_input = self.in_proj_a(hidden_states)
        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                cache_params.conv_states[self.layer_idx] = functional.pad(
                    mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0)
                )
            mixed_qkv = functional.silu(self.conv1d(mixed_qkv)[:, :, :sequence_length])
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, sequence_length, -1, self.head_k_dim)
        key = key.reshape(batch_size, sequence_length, -1, self.head_k_dim)
        value = value.reshape(batch_size, sequence_length, -1, self.head_v_dim)
        beta = beta_input.sigmoid()
        decay = self.negative_a * functional.softplus(
            decay_input.float() + self.dt_bias
        )
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        if use_precomputed_states:
            output, last_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=decay,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            output, last_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=decay,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_state
        output = self.norm(
            output.reshape(-1, self.head_v_dim), gate.reshape(-1, self.head_v_dim)
        )
        return self.out_proj(output.reshape(batch_size, sequence_length, -1))


def head_permutation(key_heads, value_heads, width, *, tiled_to_grouped):
    if value_heads % key_heads:
        raise ValueError("Value heads must be divisible by key heads")
    repeat = value_heads // key_heads
    axes = (
        (repeat, key_heads, width) if tiled_to_grouped else (key_heads, repeat, width)
    )
    return (
        torch.arange(value_heads * width, device="musa")
        .reshape(axes)
        .transpose(0, 1)
        .reshape(-1)
    )


def load_hf_gguf_reference(path, config_path):
    import gguf

    if hf.FusedRMSNormGated is not None:
        raise ValueError(
            "Reference requires the Torch RMSNorm implementation, not optional fused CUDA extensions"
        )
    path = Path(path).resolve()
    before = path.stat()
    reader = gguf.GGUFReader(path)
    metadata = {
        name: field.contents()
        for name, field in reader.fields.items()
        if not name.startswith(("tokenizer.", "GGUF."))
    }
    config = Qwen3_5TextConfig(
        **json.loads(Path(config_path).read_text())["text_config"]
    )
    comparisons = (
        (config.hidden_size, metadata.get("qwen35.embedding_length"), 2560),
        (config.num_hidden_layers, metadata.get("qwen35.block_count"), 32),
        (config.num_attention_heads, metadata.get("qwen35.attention.head_count"), 16),
        (config.num_key_value_heads, metadata.get("qwen35.attention.head_count_kv"), 4),
        (config.head_dim, metadata.get("qwen35.attention.key_length"), 256),
        (config.intermediate_size, metadata.get("qwen35.feed_forward_length"), 9216),
        (config.linear_num_key_heads, metadata.get("qwen35.ssm.group_count"), 16),
        (config.linear_num_value_heads, metadata.get("qwen35.ssm.time_step_rank"), 32),
    )
    if metadata.get("general.architecture") != "qwen35" or any(
        first != second or second != expected for first, second, expected in comparisons
    ):
        raise ValueError("Reference configuration is not the expected Qwen3.5-4B")
    if config.vocab_size != 248320 or not config.tie_word_embeddings:
        raise ValueError("Reference requires the 248320-entry shared vocabulary")
    if (
        config.layer_types
        != [
            "full_attention" if index % 4 == 3 else "linear_attention"
            for index in range(32)
        ]
        or config.linear_conv_kernel_dim != metadata.get("qwen35.ssm.conv_kernel")
        or config.linear_key_head_dim != metadata.get("qwen35.ssm.state_size")
        or config.linear_value_head_dim * config.linear_num_value_heads
        != metadata.get("qwen35.ssm.inner_size")
        or config.rope_parameters["rope_type"] != "default"
        or config.rope_parameters["rope_theta"] != metadata.get("qwen35.rope.freq_base")
        or config.rope_parameters["partial_rotary_factor"] * config.head_dim
        != metadata.get("qwen35.rope.dimension_count")
        or config.rope_parameters.get("mrope_section", [11, 11, 10])
        != metadata.get("qwen35.rope.dimension_sections", [])[:3]
    ):
        raise ValueError("Reference GDN or RoPE configuration does not match GGUF")
    config.dtype = torch.float32
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = hf.Qwen3_5ForCausalLM(config)
        for index, layer in enumerate(model.model.layers):
            if layer.layer_type == "linear_attention":
                layer.linear_attn = DirectGgufDeltaNet(config, index)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    used = set()
    records = []

    def read(name, shape, quantized=False):
        tensor = tensors[name]
        if name in used or tuple(reversed(tensor.shape.tolist())) != tuple(shape):
            raise ValueError(f"Duplicated or shape-mismatched reference tensor: {name}")
        data = np.array(tensor.data, copy=True)
        encoding = tensor.tensor_type.name.lower()
        if quantized:
            if encoding not in BLOCK_BYTES or data.dtype != np.uint8:
                raise ValueError(f"Unsupported packed reference weight: {name}")
            if (
                shape[1] % 256
                or data.nbytes != shape[0] * shape[1] // 256 * BLOCK_BYTES[encoding]
            ):
                raise ValueError(f"Malformed packed reference weight: {name}")
            data = data.reshape(shape[0], -1)
        elif encoding != "f32" or data.dtype != np.float32:
            raise ValueError(f"Expected direct FP32 reference parameter: {name}")
        used.add(name)
        records.append(
            dict(
                name=name,
                encoding=encoding,
                shape=list(shape),
                nbytes=data.nbytes,
                sha256=hashlib.sha256(data.tobytes()).hexdigest(),
            )
        )
        return torch.from_numpy(data).to("musa"), encoding

    def projection(name, original):
        value, encoding = read(name, original.weight.shape, True)
        return ReferenceLinear(value, encoding)

    def norm(name, original):
        replacement = DirectGgufRMSNorm(original.weight.numel(), eps=original.eps)
        value, _ = read(name, original.weight.shape)
        replacement.weight = torch.nn.Parameter(value, requires_grad=False)
        return replacement

    packed, encoding = read(
        "token_embd.weight", (config.vocab_size, config.hidden_size), True
    )
    model.model.embed_tokens = GgufEmbedding(packed, encoding)
    model.lm_head = ReferenceLinear(packed, encoding)
    model.model.norm = norm("output_norm.weight", model.model.norm)
    heads = head_permutation(16, 32, 1, tiled_to_grouped=True)
    value_rows = head_permutation(16, 32, 128, tiled_to_grouped=True)
    output_columns = head_permutation(16, 32, 128, tiled_to_grouped=False)
    qkv_rows = torch.cat((torch.arange(4096, device="musa"), 4096 + value_rows))
    for index, layer in enumerate(model.model.layers):
        prefix = f"blk.{index}."
        layer.input_layernorm = norm(prefix + "attn_norm.weight", layer.input_layernorm)
        layer.post_attention_layernorm = norm(
            prefix + "post_attention_norm.weight", layer.post_attention_layernorm
        )
        layer.mlp.gate_proj = projection(
            prefix + "ffn_gate.weight", layer.mlp.gate_proj
        )
        layer.mlp.up_proj = projection(prefix + "ffn_up.weight", layer.mlp.up_proj)
        layer.mlp.down_proj = projection(
            prefix + "ffn_down.weight", layer.mlp.down_proj
        )
        if layer.layer_type == "linear_attention":
            attention = layer.linear_attn
            attention.in_proj_qkv = PermutedProjection(
                projection(prefix + "attn_qkv.weight", attention.in_proj_qkv), qkv_rows
            )
            attention.in_proj_z = PermutedProjection(
                projection(prefix + "attn_gate.weight", attention.in_proj_z), value_rows
            )
            attention.in_proj_a = PermutedProjection(
                projection(prefix + "ssm_alpha.weight", attention.in_proj_a), heads
            )
            attention.in_proj_b = PermutedProjection(
                projection(prefix + "ssm_beta.weight", attention.in_proj_b), heads
            )
            attention.out_proj = PermutedProjection(
                projection(prefix + "ssm_out.weight", attention.out_proj),
                output_columns,
                input_order=True,
            )
            negative_a, _ = read(prefix + "ssm_a", (32,))
            attention.negative_a = negative_a[heads]
            bias, _ = read(prefix + "ssm_dt.bias", (32,))
            attention.dt_bias = torch.nn.Parameter(bias[heads], requires_grad=False)
            convolution, _ = read(prefix + "ssm_conv1d.weight", (8192, 4))
            attention.conv1d.weight = torch.nn.Parameter(
                convolution[qkv_rows].unsqueeze(1), requires_grad=False
            )
            value, _ = read(prefix + "ssm_norm.weight", (128,))
            attention.norm.weight = torch.nn.Parameter(value, requires_grad=False)
        else:
            attention = layer.self_attn
            attention.q_proj = projection(prefix + "attn_q.weight", attention.q_proj)
            attention.k_proj = projection(prefix + "attn_k.weight", attention.k_proj)
            attention.v_proj = projection(prefix + "attn_v.weight", attention.v_proj)
            attention.o_proj = projection(
                prefix + "attn_output.weight", attention.o_proj
            )
            attention.q_norm = norm(prefix + "attn_q_norm.weight", attention.q_norm)
            attention.k_norm = norm(prefix + "attn_k_norm.weight", attention.k_norm)
    model.model.rotary_emb = hf.Qwen3_5TextRotaryEmbedding(config, device="musa")
    if used != set(tensors) or len(used) != 426:
        raise ValueError(
            "Reference did not consume exactly the complete original GGUF text model"
        )
    if any(value.is_meta for value in (*model.parameters(), *model.buffers())):
        raise ValueError("Reference contains uninitialized tensors")
    model = model.eval().to("musa")
    if any(
        value.device.type != "musa" for value in (*model.parameters(), *model.buffers())
    ):
        raise ValueError("Reference must keep all model tensors on MUSA")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("GGUF file changed during reference loading")
    report = dict(
        source=str(path),
        reference="Transformers Qwen3.5 with explicit GGUF representation adapters",
        activation_dtype="float32",
        tensors=records,
        tensor_counts=dict(Counter(record["encoding"] for record in records)),
        grouped_head_order=True,
        packed_weights_unchanged=True,
        original_bf16_model=False,
        cpu_model_inference=False,
    )
    return model, report
