"""Explicit native math conversion for the two original Qwen3.5 vision grids."""

import torch

from executorch.backends.musa.vision_gelu import VISION_GELU, validate as validate_gelu
from executorch.backends.musa.vision_layer_norm import (
    VISION_LAYER_NORM,
    validate as validate_layer_norm,
)
from executorch.backends.musa.vision_softmax import (
    VISION_SOFTMAX,
    validate as validate_softmax,
)


def use_native_vision_math(exported):
    """Validate every replacement before mutating an undecomposed export."""
    replacements = []
    counts = dict(layer_norm=0, softmax=0, gelu_tanh=0, gelu_erf=0)
    grid_lengths = set()
    for node in exported.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target == torch.ops.aten.layer_norm.default:
            if len(node.args) != 5 or node.kwargs or tuple(node.args[1]) != (1024,):
                raise ValueError("Expected explicit affine vision LayerNorm arguments")
            value, _, weight, bias, epsilon = node.args
            validate_layer_norm(
                value.meta["val"], weight.meta["val"], bias.meta["val"], epsilon
            )
            grid_lengths.add(value.meta["val"].shape[0])
            replacements.append(
                (node, VISION_LAYER_NORM, (value, weight, bias, epsilon))
            )
            counts["layer_norm"] += 1
        elif node.target == torch.ops.aten.softmax.int:
            if node.args[1:] != (-1, torch.float32) or node.kwargs:
                raise ValueError("Expected FP32 last-axis vision softmax")
            value = node.args[0]
            validate_softmax(value.meta["val"])
            grid_lengths.add(value.meta["val"].shape[1])
            replacements.append((node, VISION_SOFTMAX, (value,)))
            counts["softmax"] += 1
        elif node.target == torch.ops.aten.gelu.default:
            if len(node.args) != 1 or set(node.kwargs) - {"approximate"}:
                raise ValueError("Expected explicit vision GELU arguments")
            approximation = node.kwargs.get("approximate", "none")
            if approximation not in ("none", "tanh"):
                raise ValueError("Unsupported vision GELU approximation")
            mode = int(approximation == "tanh")
            value = node.args[0]
            validate_gelu(value.meta["val"], mode)
            rows = value.meta["val"].shape[0]
            grid_lengths.add(rows if mode else rows * 4)
            replacements.append((node, VISION_GELU, (value, mode)))
            counts["gelu_tanh" if mode else "gelu_erf"] += 1
    if counts != dict(layer_norm=49, softmax=24, gelu_tanh=24, gelu_erf=1):
        raise ValueError(f"Expected complete 24-block vision math, found {counts}")
    if grid_lengths not in ({800}, {1200}):
        raise ValueError("Expected a consistent original single-image vision grid")
    for node, target, arguments in replacements:
        node.target = target
        node.args = arguments
        node.kwargs = {}
    exported.graph.lint()
    exported.graph_module.recompile()
    exported.validate()
    return counts
