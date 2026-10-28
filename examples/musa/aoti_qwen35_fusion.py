"""Opt-in single-token MUSA BF16 projection fusion for the Qwen3.5 exporter."""

import torch
from torch import nn
from torch.nn import functional as functional

from executorch.examples.models.llama.attention import (
    AttentionGatedDeltaNet,
    AttentionMHA,
)
from executorch.examples.models.llama.feed_forward import FeedForward


FUSION_NAME = "ffn_gate_up_and_gdn_qkv_z_b_a"


def _validate_linear(module, output_features, input_features=1024):
    if (
        type(module) is not nn.Linear
        or module.training
        or module.bias is not None
        or module.in_features != input_features
        or module.out_features != output_features
        or tuple(module.weight.shape) != (output_features, input_features)
        or module.weight.dtype != torch.bfloat16
        or module.weight.device.type != "musa"
        or not module.weight.is_contiguous()
    ):
        raise ValueError("Fusion requires unmodified eval MUSA BF16 Qwen linears")


def _fused_linear(first, second):
    if first.weight.device != second.weight.device:
        raise ValueError("Fused projection weights must share a device")
    result = nn.Linear(
        first.in_features,
        first.out_features + second.out_features,
        bias=False,
        device="meta",
        dtype=torch.bfloat16,
    )
    result.weight = nn.Parameter(
        torch.cat((first.weight, second.weight), dim=0), requires_grad=False
    )
    return result.eval()


def _validate_input(hidden, weight, training):
    if (
        training
        or tuple(hidden.shape) != (1, 1, 1024)
        or hidden.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or hidden.device.type != "musa"
        or hidden.device != weight.device
        or not hidden.is_contiguous()
    ):
        raise ValueError("Fusion requires one contiguous MUSA BF16 token in eval mode")


class Qwen35MusaFusedFeedForward(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.validate(original)
        self.gate_up = _fused_linear(original.w1, original.w3)
        self.down = original.w2
        self.act_fn = original.act_fn
        self.eval()

    @staticmethod
    def validate(original):
        if type(original) is not FeedForward or original.act_fn is not functional.silu:
            raise ValueError("Fusion requires the Qwen SiLU feed-forward module")
        if original.training:
            raise ValueError("Projection fusion is inference-only")
        _validate_linear(original.w1, 3584)
        _validate_linear(original.w3, 3584)
        _validate_linear(original.w2, 1024, 3584)

    def forward(self, hidden):
        _validate_input(hidden, self.gate_up.weight, self.training)
        gate, up = self.gate_up(hidden).split(3584, dim=-1)
        return self.down(self.act_fn(gate) * up)


class Qwen35MusaFusedDeltaAttention(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.validate(original)
        self.qkv_z = _fused_linear(original.in_proj_qkv, original.in_proj_z)
        self.b_a = _fused_linear(original.in_proj_b, original.in_proj_a)
        self.attention = original
        self.eval()

    @staticmethod
    def validate(original):
        if type(original) is not AttentionGatedDeltaNet or original.training:
            raise ValueError("Fusion requires the eval Qwen GatedDeltaNet module")
        if (
            original.max_batch_size != 1
            or original.hidden_size != 1024
            or original.num_k_heads != 16
            or original.num_v_heads != 16
            or original.head_k_dim != 128
            or original.head_v_dim != 128
            or original.conv_kernel_size != 4
        ):
            raise ValueError("Unsupported GatedDeltaNet fusion geometry")
        _validate_linear(original.in_proj_qkv, 6144)
        _validate_linear(original.in_proj_z, 2048)
        _validate_linear(original.in_proj_b, 16)
        _validate_linear(original.in_proj_a, 16)
        _validate_linear(original.out_proj, 1024, 2048)
        if (
            tuple(original.conv_state.shape) != (1, 6144, 4)
            or original.conv_state.dtype != torch.bfloat16
            or tuple(original.recurrent_state.shape) != (1, 16, 128, 128)
            or original.recurrent_state.dtype != torch.float32
        ):
            raise ValueError(
                "Fusion requires BF16 convolution and FP32 recurrent state"
            )

    def forward(self, hidden, freqs_cos, freqs_sin, **kwargs):
        _validate_input(hidden, self.qkv_z.weight, self.training)
        if self.attention.recurrent_state.dtype != torch.float32:
            raise ValueError("Fusion requires FP32 recurrent state")
        self.attention._maybe_reset_state(kwargs.get("input_pos"), 1)
        mixed_qkv, gate = self.qkv_z(hidden).split((6144, 2048), dim=-1)
        beta_input, decay_input = self.b_a(hidden).split(16, dim=-1)
        return self.attention._forward_projected(
            mixed_qkv, gate, beta_input, decay_input
        )


@torch.no_grad()
def fuse_qwen35_projections(model):
    if any(module.training for module in model.modules()):
        raise ValueError("Projection fusion is inference-only")
    if model.max_context_len not in (64, 128, 256) or model.params.max_batch_size != 1:
        raise ValueError("Fusion supports batch one and contexts 64, 128 or 256")
    if model.output.weight is not model.tok_embeddings.weight:
        raise ValueError("Qwen output and embedding weights must remain tied")
    device = model.output.weight.device
    if device.type != "musa" or any(
        value.device != device or value.dtype != torch.bfloat16
        for value in model.parameters()
    ):
        raise ValueError("Fusion requires BF16 model parameters on one MUSA device")
    if any(value.device != device for value in model.buffers()):
        raise ValueError("Model buffers must share the projection device")
    for layer in model.layers:
        Qwen35MusaFusedFeedForward.validate(layer.feed_forward)
        if type(layer.attention) is AttentionGatedDeltaNet:
            Qwen35MusaFusedDeltaAttention.validate(layer.attention)
        elif type(layer.attention) is not AttentionMHA:
            raise ValueError("Unsupported attention module for Qwen projection fusion")
    parameters = sum(value.numel() for value in model.parameters())
    replacements = [
        (
            layer,
            Qwen35MusaFusedFeedForward(layer.feed_forward),
            (
                Qwen35MusaFusedDeltaAttention(layer.attention)
                if type(layer.attention) is AttentionGatedDeltaNet
                else None
            ),
        )
        for layer in model.layers
    ]
    for layer, feed_forward, attention in replacements:
        if attention is not None:
            del attention.attention.in_proj_qkv
            del attention.attention.in_proj_z
            del attention.attention.in_proj_b
            del attention.attention.in_proj_a
            layer.attention = attention
        layer.feed_forward = feed_forward
    if parameters != sum(value.numel() for value in model.parameters()):
        raise RuntimeError("Projection fusion changed the model parameter count")
    return model
