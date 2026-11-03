"""Load unchanged Qwen3.5-4B GGUF text weights for MUSA export."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from executorch.backends.musa.causal_attention import CAUSAL_ATTENTION
from executorch.backends.musa.causal_prefix_attention import (
    CAUSAL_PREFIX,
    PREFIX_ATTENTION,
)
from executorch.backends.musa.gguf import BLOCK_BYTES
from executorch.backends.musa.gguf_packed import GgufEmbedding, GgufLinear, QUANT_NAMES
from executorch.backends.musa.q6_predecoded import Q6_PREDECODED, predecode_q6
from executorch.examples.models.llama.attention import (
    AttentionGatedDeltaNet,
    AttentionMHA,
)
from executorch.examples.models.llama.feed_forward import FeedForward
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from examples.musa.aoti_qwen35_export import Qwen35MusaSDPA, STATE_SUFFIXES


class GgufMergedFeedForward(torch.nn.Module):
    def __init__(self, previous):
        super().__init__()
        if not isinstance(previous, FeedForward) or not all(
            isinstance(layer, GgufLinear) and layer.bias is None
            for layer in (previous.w1, previous.w2, previous.w3)
        ):
            raise ValueError("Merged GGUF FFN requires bias-free packed projections")
        gate, up = previous.w1, previous.w3
        if (
            gate.quant_type != up.quant_type
            or gate.packed_weight.shape != up.packed_weight.shape
            or gate.packed_weight.device != up.packed_weight.device
        ):
            raise ValueError("Merged GGUF gate/up must have matching packed geometry")
        self.gate_up = GgufLinear(
            torch.cat((gate.packed_weight, up.packed_weight), dim=0),
            QUANT_NAMES[gate.quant_type],
        )
        self.w2 = previous.w2
        self.act_fn = previous.act_fn

    def forward(self, value):
        gate, up = self.gate_up(value).chunk(2, dim=-1)
        return self.w2(self.act_fn(gate) * up)


class GgufMergedGdnProjection(torch.nn.Module):
    def __init__(self, gate, beta, decay):
        super().__init__()
        projections = (gate, beta, decay)
        if not all(
            isinstance(layer, GgufLinear) and layer.bias is None
            for layer in projections
        ):
            raise ValueError("Merged GDN requires bias-free packed projections")
        if (
            not all(
                layer.quant_type == gate.quant_type
                and layer.packed_weight.ndim == 2
                and layer.packed_weight.shape[1] == gate.packed_weight.shape[1]
                and layer.packed_weight.device == gate.packed_weight.device
                for layer in projections
            )
            or beta.packed_weight.shape != decay.packed_weight.shape
        ):
            raise ValueError("Merged GDN projections must have matching packed inputs")
        self.sizes = tuple(layer.packed_weight.shape[0] for layer in projections)
        self.projection = GgufLinear(
            torch.cat(tuple(layer.packed_weight for layer in projections), dim=0),
            QUANT_NAMES[gate.quant_type],
        )

    def forward(self, value):
        return self.projection(value).split(self.sizes, dim=-1)


class GgufValidLengthSDPA(Qwen35MusaSDPA):
    def forward(self, input_pos, query, key, value, batch_size, sequence_length, mask):
        if batch_size != 1 or sequence_length != 1:
            raise ValueError("GGUF valid-prefix attention requires one token")
        output = CAUSAL_ATTENTION(query, key, value, input_pos, mask)
        return output.transpose(1, 2).reshape(batch_size, sequence_length, self.dim)


class GgufSharedPrefixAttention(AttentionMHA):
    def forward(self, hidden, cosine, sine, **options):
        batch, sequence, _ = hidden.shape
        if (
            batch != 1
            or sequence != 1
            or not self.use_kv_cache
            or not self.use_q_gate
            or options.get("shared_kv") is not None
        ):
            raise ValueError(
                "Shared-prefix attention requires Qwen single-token cached geometry"
            )
        position = options["input_pos"]
        control = options.get("causal_prefix")
        if control is None:
            control = CAUSAL_PREFIX(position, self.mask[position])
        projected = self.wq(hidden).view(
            batch, sequence, self.n_local_heads, self.head_dim * 2
        )
        query, gate = torch.chunk(projected, 2, dim=-1)
        gate = gate.reshape(batch, sequence, -1)
        query, key, value = self._prepare_qkv(
            query, hidden, batch, sequence, cosine, sine
        )
        key, value = self.kv_cache.update(position, key, value)
        output = PREFIX_ATTENTION(query, key, value, control)
        output = output.transpose(1, 2).reshape(
            batch, sequence, self.n_local_heads * self.head_dim
        )
        output = self._apply_output_transforms(output, hidden, gate, batch, sequence)
        return self.wo(output), None


class GgufGatedDeltaNet(AttentionGatedDeltaNet):
    def __init__(self, args, layer_id, rope):
        super().__init__(args, layer_id, rope)
        del self.A_log
        self.register_buffer("negative_a", torch.empty(self.num_v_heads))
        self.in_proj_zba = None

    def merge_gate_projections(self):
        if self.in_proj_zba is not None:
            raise ValueError("GDN gate projections are already merged")
        merged = GgufMergedGdnProjection(self.in_proj_z, self.in_proj_b, self.in_proj_a)
        if merged.sizes != (self.value_dim, self.num_v_heads, self.num_v_heads):
            raise ValueError("Merged GDN outputs do not match the configured heads")
        self.in_proj_zba = merged
        del self.in_proj_z, self.in_proj_b, self.in_proj_a

    def forward(self, hidden, cosine, sine, **options):
        if self.in_proj_zba is None:
            return super().forward(hidden, cosine, sine, **options)
        batch, sequence, _ = hidden.shape
        if not 1 <= batch <= self.max_batch_size:
            raise ValueError("Merged GDN batch exceeds the configured state capacity")
        self._maybe_reset_state(options.get("input_pos"), batch)
        mixed_qkv = self.in_proj_qkv(hidden)
        gate, beta, decay = self.in_proj_zba(hidden)
        return self._forward_projected(
            mixed_qkv,
            gate.reshape(batch, sequence, -1, self.head_v_dim),
            beta,
            decay,
        )

    def _forward_projected(self, mixed_qkv, gate, beta_input, decay_input):
        batch_size, sequence_length, _ = mixed_qkv.shape
        mixed_qkv = self._apply_causal_conv(mixed_qkv)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(
            batch_size, sequence_length, self.num_k_heads, self.head_k_dim
        )
        key = key.reshape(
            batch_size, sequence_length, self.num_k_heads, self.head_k_dim
        )
        value = value.reshape(
            batch_size, sequence_length, self.num_v_heads, self.head_v_dim
        )
        query = query.repeat(1, 1, self.head_repeat, 1)
        key = key.repeat(1, 1, self.head_repeat, 1)
        beta = beta_input.sigmoid()
        decay = self.negative_a * functional.softplus(
            decay_input.float() + self.dt_bias
        )
        output = self._recurrent_gated_delta_rule(query, key, value, decay, beta)
        output = self.norm(
            output.reshape(-1, self.head_v_dim), gate.reshape(-1, self.head_v_dim)
        )
        return (
            self.out_proj(output.reshape(batch_size, sequence_length, self.value_dim)),
            None,
        )


def model_parameters(metadata, context_length):
    params = json.loads(
        (Path(__file__).parents[1] / "models/qwen3_5/config/4b_config.json").read_text()
    )
    expected = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 32,
        "qwen35.embedding_length": 2560,
        "qwen35.feed_forward_length": 9216,
        "qwen35.attention.head_count": 16,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "qwen35.rope.dimension_count": 64,
        "qwen35.rope.dimension_sections": [11, 11, 10, 0],
        "qwen35.rope.freq_base": 10000000.0,
        "qwen35.ssm.conv_kernel": 4,
        "qwen35.ssm.state_size": 128,
        "qwen35.ssm.group_count": 16,
        "qwen35.ssm.time_step_rank": 32,
        "qwen35.ssm.inner_size": 4096,
        "qwen35.full_attention_interval": 4,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(
                f"Unsupported Qwen3.5-4B metadata: {name}={metadata.get(name)!r}"
            )
    epsilon = metadata.get("qwen35.attention.layer_norm_rms_epsilon", 0)
    if (
        not isinstance(epsilon, (int, float))
        or not math.isfinite(epsilon)
        or abs(epsilon - 1e-6) > 1e-12
    ):
        raise ValueError("Unsupported Qwen3.5 RMSNorm epsilon")
    if not isinstance(context_length, int) or not 2 <= context_length <= min(
        4096, metadata.get("qwen35.context_length", 0)
    ):
        raise ValueError("GGUF bring-up requires context between 2 and 4096")
    params.update(norm_eps=epsilon, rms_norm_add_unit_offset=False)
    return ModelArgs(
        **params,
        max_seq_len=context_length,
        max_context_len=context_length,
        use_kv_cache=True,
    )


def tensor_mapping(args):
    mapping = {"output_norm.weight": "norm.weight"}
    for index in range(args.n_layers):
        prefix = f"blk.{index}."
        layer = f"layers.{index}."
        names = {
            "attn_norm.weight": "attention_norm.weight",
            "post_attention_norm.weight": "ffn_norm.weight",
            "ffn_gate.weight": "feed_forward.w1",
            "ffn_down.weight": "feed_forward.w2",
            "ffn_up.weight": "feed_forward.w3",
        }
        if args.layer_types[index] == "linear_attention":
            names.update(
                {
                    "attn_qkv.weight": "attention.in_proj_qkv",
                    "attn_gate.weight": "attention.in_proj_z",
                    "ssm_beta.weight": "attention.in_proj_b",
                    "ssm_alpha.weight": "attention.in_proj_a",
                    "ssm_out.weight": "attention.out_proj",
                    "ssm_a": "attention.negative_a",
                    "ssm_dt.bias": "attention.dt_bias",
                    "ssm_norm.weight": "attention.norm.weight",
                    "ssm_conv1d.weight": "attention.conv1d.weight",
                }
            )
        else:
            names.update(
                {
                    "attn_q.weight": "attention.wq",
                    "attn_k.weight": "attention.wk",
                    "attn_v.weight": "attention.wv",
                    "attn_output.weight": "attention.wo",
                    "attn_q_norm.weight": "attention.q_norm_fn.weight",
                    "attn_k_norm.weight": "attention.k_norm_fn.weight",
                }
            )
        mapping.update(
            {prefix + name: layer + target for name, target in names.items()}
        )
    return mapping


def load_gguf_text(
    path,
    context_length=4096,
    valid_length_attention=False,
    predecode_q6_weights=False,
    share_attention_prefix=False,
    merge_ffn_gate_up=False,
    merge_gdn_gate_projections=False,
):
    import gguf

    if share_attention_prefix and not valid_length_attention:
        raise ValueError("Shared attention prefix requires valid-length attention")
    path = Path(path).resolve()
    before = path.stat()
    reader = gguf.GGUFReader(path)
    metadata = {
        name: field.contents()
        for name, field in reader.fields.items()
        if not name.startswith(("tokenizer.", "GGUF."))
    }
    args = model_parameters(metadata, context_length)
    mapping = tensor_mapping(args)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(reader.tensors) or set(tensors) != set(mapping) | {
        "token_embd.weight"
    }:
        raise ValueError(
            "GGUF tensor names do not match the complete 32-layer text model"
        )
    with torch.device("meta"):
        model = construct_transformer(args)
        for index, layer in enumerate(model.layers):
            if isinstance(layer.attention, AttentionGatedDeltaNet):
                layer.attention = GgufGatedDeltaNet(args, index, model.rope)
            elif share_attention_prefix and isinstance(layer.attention, AttentionMHA):
                layer.attention = GgufSharedPrefixAttention(args, index, model.rope)
    records = []

    def read_tensor(name, shape, quantized):
        tensor = tensors[name]
        logical_shape = tuple(reversed(tuple(int(size) for size in tensor.shape)))
        if logical_shape != tuple(shape):
            raise ValueError(
                f"GGUF shape mismatch for {name}: {logical_shape} != {tuple(shape)}"
            )
        encoding = tensor.tensor_type.name.lower()
        data = np.array(tensor.data, copy=True)
        if quantized:
            if encoding not in BLOCK_BYTES or len(shape) != 2 or shape[1] % 256:
                raise ValueError(f"Expected supported packed K-quant matrix: {name}")
            expected_bytes = shape[0] * (shape[1] // 256) * BLOCK_BYTES[encoding]
            if data.dtype != np.uint8 or data.size != expected_bytes:
                raise ValueError(f"Invalid packed storage for {name}")
            data = data.reshape(shape[0], -1)
        elif encoding != "f32" or data.dtype != np.float32:
            raise ValueError(f"Expected unchanged FP32 auxiliary tensor: {name}")
        if tensor.n_bytes != data.nbytes:
            raise ValueError(f"Inconsistent tensor byte size: {name}")
        records.append(
            dict(
                name=name,
                shape=list(shape),
                encoding=encoding,
                nbytes=data.nbytes,
                sha256=hashlib.sha256(data.tobytes()).hexdigest(),
            )
        )
        value = torch.from_numpy(data).to("musa")
        if predecode_q6_weights and encoding == "q6_k":
            value = predecode_q6(value)
            encoding = Q6_PREDECODED
        records[-1].update(runtime_encoding=encoding, runtime_nbytes=value.nbytes)
        return value, encoding

    embedding, encoding = read_tensor(
        "token_embd.weight", (args.vocab_size, args.dim), True
    )
    model.tok_embeddings = GgufEmbedding(embedding, encoding)
    model.output = GgufLinear(embedding, encoding)
    for source, destination in mapping.items():
        owner_name, field = destination.rsplit(".", 1)
        owner = model.get_submodule(owner_name)
        if destination.endswith(".weight") or field in ("dt_bias", "negative_a"):
            target = (
                owner.get_buffer(field)
                if field == "negative_a"
                else owner.get_parameter(field)
            )
            shape = tuple(target.shape)
            if destination.endswith("conv1d.weight"):
                value, _ = read_tensor(source, (shape[0], shape[-1]), False)
                value = value.unsqueeze(1)
            else:
                value, _ = read_tensor(source, shape, False)
            if field == "negative_a":
                if (
                    not torch.isfinite(value).all().item()
                    or not (value < 0).all().item()
                ):
                    raise ValueError(
                        "GGUF ssm_a must contain finite negative decay coefficients"
                    )
                owner.register_buffer(field, value)
            else:
                owner.register_parameter(
                    field, torch.nn.Parameter(value, requires_grad=False)
                )
        else:
            layer = model.get_submodule(destination)
            if not isinstance(layer, torch.nn.Linear) or layer.bias is not None:
                raise ValueError(f"Expected bias-free projection: {destination}")
            value, encoding = read_tensor(source, layer.weight.shape, True)
            model.set_submodule(destination, GgufLinear(value, encoding))
    if merge_ffn_gate_up:
        for layer in model.layers:
            layer.feed_forward = GgufMergedFeedForward(layer.feed_forward)
    if merge_gdn_gate_projections:
        for layer in model.layers:
            if isinstance(layer.attention, GgufGatedDeltaNet):
                layer.attention.merge_gate_projections()
    for name, value in list(model.named_buffers()):
        if value.is_meta:
            if not name.endswith(STATE_SUFFIXES):
                raise ValueError(f"Uninitialized GGUF model buffer: {name}")
            owner_name, field = name.rsplit(".", 1)
            model.get_submodule(owner_name).register_buffer(
                field, torch.zeros(value.shape, dtype=value.dtype, device="musa")
            )
    if any(value.is_meta for value in model.parameters()):
        raise ValueError("GGUF loading left uninitialized model parameters")
    model = model.eval().to(device="musa")
    for layer in model.layers:
        if isinstance(layer.attention, AttentionMHA):
            previous = layer.attention.SDPA
            attention_type = (
                GgufValidLengthSDPA if valid_length_attention else Qwen35MusaSDPA
            )
            layer.attention.SDPA = attention_type(
                previous.dim,
                previous.head_dim,
                previous.n_rep,
                previous.max_context_len,
            ).eval()
    if any(
        value.device.type != "musa" for value in (*model.parameters(), *model.buffers())
    ):
        raise RuntimeError("GGUF model tensors must reside on MUSA")
    if (
        model.output.packed_weight.data_ptr()
        != model.tok_embeddings.packed_weight.data_ptr()
    ):
        raise RuntimeError("Shared GGUF embedding/output weight was duplicated")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("GGUF file changed while loading")
    report = dict(
        source=str(path),
        context_length=context_length,
        tensors=records,
        tensor_counts=dict(Counter(record["encoding"] for record in records)),
        packed_bytes=sum(
            record["nbytes"] for record in records if record["encoding"] in BLOCK_BYTES
        ),
        runtime_packed_bytes=sum(
            record["runtime_nbytes"]
            for record in records
            if record["encoding"] in BLOCK_BYTES
        ),
        q6_predecoded=predecode_q6_weights,
        shared_attention_prefix=share_attention_prefix,
        merged_ffn_gate_up=merge_ffn_gate_up,
        merged_gdn_gate_projections=merge_gdn_gate_projections,
        text_layers=args.n_layers,
        gdn_head_order="tiled",
        negative_a_preserved=True,
        norm_offset_already_in_weights=True,
        activation_dtype="float32",
        shared_output_head=True,
        multimodal_enabled=False,
        full_model_accuracy_validated=False,
    )
    return model, report
