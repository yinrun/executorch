# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export and execute representative models through the MUSA AOTI backend.

The script checks fixed or bounded dynamic inputs through the complete export
and runner path. It is a correctness smoke test, not a benchmark. The
``narrow_resnet18`` model keeps the ResNet-18 block topology but uses narrow
channels; ``resnet18`` is the standard torchvision model with optional weights.
``mobilenet_v2`` exercises depthwise/grouped convolutions and ReLU6;
``deeplabv3_mobilenet_v3_large`` covers a pretrained segmentation network.
``efficientnet_b0`` covers squeeze-and-excitation and SiLU blocks.
``densenet121`` covers dense concatenation blocks and transition pooling.
``convnext_tiny`` covers depthwise convolution, LayerNorm, GELU, and residuals.
``vit_b_16`` covers full-depth multihead attention and LayerNorm.
``transposed_bmm`` covers a non-contiguous left-hand batch matrix.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import sysconfig
from functools import partial
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F
from torch.export import Dim, export

from executorch.backends.aoti.host_tensor import snapshot_to_cpu as _snapshot_tensor
from executorch.backends.musa import (
    MusaBackend,
    MusaPartitioner,
    export_with_musa_pooling_shapes,
    export_with_musa_sdpa_decomposition,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program

from examples.musa.acceptance_artifacts import record_artifact, verify_artifacts


class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SimpleCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class SimpleAudioCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 4, 3, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Conv1d(4, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(8),
        )
        self.classifier = nn.Linear(8, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class SimpleVideoCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 4, 3, padding=1),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class SimpleDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(3, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SimpleAudioDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(
                2, 4, 3, stride=2, padding=1, output_padding=1, groups=2
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(4, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SimpleVideoDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose3d(1, 2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(2, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DilatedDecoder(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        conv_type = {
            1: nn.ConvTranspose1d,
            2: nn.ConvTranspose2d,
            3: nn.ConvTranspose3d,
        }[dimensions]
        self.conv = conv_type(
            2, 3, 3, stride=1, padding=2, dilation=2, output_padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.GroupNorm(4, 8), nn.SiLU()
        )
        self.down = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
        )
        self.up = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.head = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.GroupNorm(4, 8),
            nn.SiLU(),
            nn.Conv2d(8, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.stem(x)
        encoded = self.down(skip)
        return self.head(torch.cat((skip, self.up(encoded)), dim=1))


class BilinearUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.GroupNorm(4, 8), nn.SiLU()
        )
        self.down = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
        )
        self.up = nn.Conv2d(16, 8, 1)
        self.head = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.stem(x)
        encoded = self.down(skip)
        resized = F.interpolate(
            encoded, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.head(torch.cat((skip, self.up(resized)), dim=1))


class DeepLabV3MobileNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

        self.model = deeplabv3_mobilenet_v3_large(
            weights=None, weights_backbone=None, aux_loss=True
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)["out"]


class TransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=4,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class TransposedBmm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x.transpose(1, 2), x)


class BroadcastBmm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(16, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.bmm(x, weight)


class CausalAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(16, 16, bias=False)
        self.key_value = nn.Linear(16, 16, bias=False)
        self.output = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        query = self.query(x).view(batch, sequence, 4, 4).transpose(1, 2)
        key, value = (
            tensor.view(batch, sequence, 2, 4).transpose(1, 2)
            for tensor in self.key_value(x).chunk(2, dim=-1)
        )
        attended = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True, enable_gqa=True
        )
        return self.output(attended.transpose(1, 2).reshape(batch, sequence, 16))


TOKEN_VOCAB_SIZE = 64


class EmbeddingClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(TOKEN_VOCAB_SIZE, 16)
        self.classifier = nn.Linear(16, 8)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedding(token_ids).mean(dim=1))


class TokenEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(TOKEN_VOCAB_SIZE, 16)
        self.encoder = TransformerBlock()
        self.classifier = nn.Linear(16, 8)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        features = self.encoder(self.embedding(token_ids))
        return self.classifier(features.mean(dim=1))


class TokenDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(TOKEN_VOCAB_SIZE, 16)
        self.attention = CausalAttention()
        self.norm = nn.LayerNorm(16)
        self.output = nn.Linear(16, TOKEN_VOCAB_SIZE, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        features = self.embedding(token_ids)
        features = self.norm(features + self.attention(features))
        return self.output(features)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        if stride != 1 or input_channels != output_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + identity)


class Resnet18(nn.Module):
    """A narrow-channel ResNet-18 topology for MUSA acceptance testing."""

    def __init__(self) -> None:
        super().__init__()
        self.input_channels = 8
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(8, 2, 1)
        self.layer2 = self._make_layer(16, 2, 2)
        self.layer3 = self._make_layer(32, 2, 2)
        self.layer4 = self._make_layer(64, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 10)

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(self.input_channels, channels, stride)]
        self.input_channels = channels
        layers.extend(ResidualBlock(channels, channels, 1) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(self.pool(x).flatten(1))


def _torchvision_resnet18() -> nn.Module:
    from torchvision.models import resnet18

    return resnet18(weights=None)


def _torchvision_mobilenet_v2() -> nn.Module:
    from torchvision.models import mobilenet_v2

    return mobilenet_v2(weights=None)


def _torchvision_efficientnet_b0() -> nn.Module:
    from torchvision.models import efficientnet_b0

    return efficientnet_b0(weights=None)


def _torchvision_densenet121() -> nn.Module:
    from torchvision.models import densenet121

    return densenet121(weights=None)


def _torchvision_convnext_tiny() -> nn.Module:
    from torchvision.models import convnext_tiny

    return convnext_tiny(weights=None)


def _torchvision_vit_b_16() -> nn.Module:
    from torchvision.models import vit_b_16

    return vit_b_16(weights=None)


class VitLayerTokens(nn.Module):
    def __init__(self, model: nn.Module, attention_stages: bool = False) -> None:
        super().__init__()
        self.model = model
        self.attention_stages = attention_stages

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patches = self.model._process_input(image)
        tokens = torch.cat(
            [self.model.class_token.expand(patches.shape[0], -1, -1), patches],
            dim=1,
        )
        hidden = self.model.encoder.dropout(
            tokens + self.model.encoder.pos_embedding
        )
        observed = [hidden[:, 0]]
        for layer in self.model.encoder.layers:
            if self.attention_stages:
                normalized = layer.ln_1(hidden)
                attention, _ = layer.self_attention(
                    normalized, normalized, normalized, need_weights=False
                )
                residual = layer.dropout(attention) + hidden
                observed.append(residual[:, 0])
                hidden = residual + layer.mlp(layer.ln_2(residual))
            else:
                hidden = layer(hidden)
            observed.append(hidden[:, 0])
        return torch.stack(observed, dim=1)


MODEL_FACTORIES = {
    "mlp": (Mlp, (1, 16)),
    "simple_cnn": (SimpleCnn, (1, 3, 32, 32)),
    "simple_audio_cnn": (SimpleAudioCnn, (1, 2, 16)),
    "simple_video_cnn": (SimpleVideoCnn, (1, 1, 4, 8, 8)),
    "simple_decoder": (SimpleDecoder, (1, 3, 8, 8)),
    "simple_audio_decoder": (SimpleAudioDecoder, (1, 2, 5)),
    "simple_video_decoder": (SimpleVideoDecoder, (1, 1, 2, 3, 4)),
    "dilated_audio_decoder": (partial(DilatedDecoder, 1), (1, 2, 5)),
    "dilated_image_decoder": (partial(DilatedDecoder, 2), (1, 2, 5, 4)),
    "dilated_video_decoder": (partial(DilatedDecoder, 3), (1, 2, 5, 4, 3)),
    "tiny_unet": (TinyUNet, (1, 3, 32, 32)),
    "bilinear_unet": (BilinearUNet, (1, 3, 32, 40)),
    "deeplabv3_mobilenet_v3_large": (DeepLabV3MobileNet, (1, 3, 64, 64)),
    "transformer_block": (TransformerBlock, (1, 4, 16)),
    "transposed_bmm": (TransposedBmm, (1, 4, 16)),
    "broadcast_bmm": (BroadcastBmm, (1, 4, 16)),
    "causal_attention": (CausalAttention, (1, 4, 16)),
    "embedding_classifier": (EmbeddingClassifier, (1, 4)),
    "token_encoder": (TokenEncoder, (1, 4)),
    "token_decoder": (TokenDecoder, (1, 4)),
    "narrow_resnet18": (Resnet18, (1, 3, 32, 32)),
    "resnet18": (_torchvision_resnet18, (1, 3, 64, 64)),
    "mobilenet_v2": (_torchvision_mobilenet_v2, (1, 3, 64, 64)),
    "efficientnet_b0": (_torchvision_efficientnet_b0, (1, 3, 64, 64)),
    "densenet121": (_torchvision_densenet121, (1, 3, 64, 64)),
    "convnext_tiny": (_torchvision_convnext_tiny, (1, 3, 64, 64)),
    "vit_b_16": (_torchvision_vit_b_16, (1, 3, 224, 224)),
}
TOKEN_MODEL_NAMES = {"embedding_classifier", "token_encoder", "token_decoder"}
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
TOLERANCES = {"fp32": 1e-4, "fp16": 1e-2, "bf16": 4e-2}


@contextlib.contextmanager
def _serialization_schema() -> Iterator[None]:
    serialize_dir = Path(__file__).parents[2] / "src/executorch/exir/_serialize"
    schema_dir = Path(__file__).parents[2] / "schema"
    saved_files = {}
    try:
        for name in ("program.fbs", "scalar_type.fbs"):
            destination = serialize_dir / name
            saved_files[destination] = (
                destination.read_bytes() if destination.exists() else None
            )
            shutil.copyfile(schema_dir / name, destination)
        yield
    finally:
        for destination, content in saved_files.items():
            if content is None:
                destination.unlink(missing_ok=True)
            else:
                destination.write_bytes(content)


def _write_tensor(tensor: torch.Tensor, path: Path) -> torch.Tensor:
    snapshot = _snapshot_tensor(tensor)
    payload = snapshot.reshape(-1).view(torch.uint8).numpy().tobytes()
    path.write_bytes(payload)
    if path.read_bytes() != payload:
        raise RuntimeError(f"Tensor artifact write/read mismatch: {path}")
    return snapshot


def _record_tensor_artifact(tensor: torch.Tensor, path: Path) -> dict:
    if tensor.device.type != "cpu":
        raise ValueError("Artifact records require a frozen CPU tensor snapshot")
    payload = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return record_artifact(path, payload)


def _make_input(
    model_name: str, input_shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    if model_name in TOKEN_MODEL_NAMES:
        result = torch.randint(
            TOKEN_VOCAB_SIZE, input_shape, device="musa", dtype=torch.int64
        )
    else:
        result = torch.randn(input_shape, device="musa", dtype=dtype)
    torch.musa.synchronize()
    return result


def _write_input(
    tensor: torch.Tensor, path: Path, max_shape: tuple[int, ...] | None
) -> None:
    if max_shape is not None and (
        len(max_shape) != tensor.ndim
        or any(size > bound for size, bound in zip(tensor.shape, max_shape))
    ):
        raise ValueError("Input shape exceeds the dynamic shape bound")
    _write_tensor(tensor, path)
    if max_shape is not None:
        padding = (math.prod(max_shape) - tensor.numel()) * tensor.element_size()
        with path.open("ab") as input_file:
            input_file.write(bytes(padding))


def _runner_environment() -> dict[str, str]:
    import torch_musa

    torch_root = Path(torch.__file__).resolve().parent
    torch_musa_root = Path(torch_musa.__file__).resolve().parent
    library_dirs = [
        torch_musa_root / "lib",
        torch_root / "lib",
        Path("/usr/local/musa/lib"),
    ]
    existing_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    library_path = ":".join(str(path) for path in library_dirs if path.is_dir())
    if existing_library_path:
        library_path = f"{library_path}:{existing_library_path}"

    preload = []
    python_library = sysconfig.get_config_var("LDLIBRARY")
    python_library_dir = sysconfig.get_config_var("LIBDIR")
    if python_library and python_library_dir:
        preload.append(str(Path(python_library_dir) / python_library))
    torch_python_library = torch_root / "lib/libtorch_python.so"
    if torch_python_library.exists():
        preload.append(str(torch_python_library))

    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = library_path
    if preload:
        environment["LD_PRELOAD"] = ":".join(preload)
    return environment


def _run_runner(
    runner: Path,
    pte_path: Path,
    ptd_path: Path,
    input_path: Path,
    output_prefix: Path,
    input_shape: tuple[int, ...] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        str(runner),
        f"--model_path={pte_path}",
        f"--data_path={ptd_path}",
        f"--inputs={input_path}",
        f"--output_file={output_prefix}",
        "--print_output=summary",
    ]
    if input_shape is not None:
        command.append(f"--input_shapes={'x'.join(map(str, input_shape))}")
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=_runner_environment(),
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "executor_runner failed\n"
            f"stdout:\n{error.stdout}\n"
            f"stderr:\n{error.stderr}"
        ) from error


def export_model(
    model_name: str,
    output_dir: Path,
    dtype: torch.dtype,
    input_size: int | None,
    weights_path: Path | None,
    batch_size: int = 1,
    dynamic_batch_max: int | None = None,
    dynamic_spatial_max: int | None = None,
    vit_blocks: int | None = None,
    vit_layer_tokens: bool = False,
    vit_attention_stages: bool = False,
) -> tuple[Path, Path, Path, torch.Tensor, nn.Module, tuple[int, ...]]:
    if not torch.musa.is_available():
        raise RuntimeError("MUSA is not available")

    model_type, input_shape = MODEL_FACTORIES[model_name]
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if dynamic_batch_max is not None and (
        batch_size < 2 or dynamic_batch_max < batch_size
    ):
        raise ValueError("--dynamic-batch-max requires 2 <= batch-size <= max")
    input_shape = (batch_size, *input_shape[1:])
    if input_size is not None:
        if model_name not in (
            "resnet18",
            "narrow_resnet18",
            "mobilenet_v2",
            "deeplabv3_mobilenet_v3_large",
            "efficientnet_b0",
            "densenet121",
            "convnext_tiny",
        ):
            raise ValueError("--input-size is only supported for CNN models")
        if input_size < 32:
            raise ValueError("--input-size must be at least 32")
        input_shape = (batch_size, 3, input_size, input_size)
    spatial_minimum = {
        "bilinear_unet": 24,
        "resnet18": 48,
        "mobilenet_v2": 48,
        "deeplabv3_mobilenet_v3_large": 48,
        "efficientnet_b0": 48,
        "densenet121": 32,
    }.get(model_name)
    if dynamic_spatial_max is not None:
        if (
            spatial_minimum is None
            or min(input_shape[2:]) < spatial_minimum
            or dynamic_spatial_max < max(input_shape[2:])
        ):
            raise ValueError(
                "--dynamic-spatial-max requires bilinear_unet (min=24) or "
                "resnet18/mobilenet_v2/deeplabv3_mobilenet_v3_large/"
                "efficientnet_b0 (min=48), "
                "or densenet121 (min=32), "
                "with max >= input size"
            )
        if model_name == "densenet121" and (
            input_shape[2] % 8 != 0
            or dynamic_spatial_max % 8 != 0
            or dynamic_spatial_max <= spatial_minimum
        ):
            raise ValueError(
                "densenet121 dynamic spatial sizes must be divisible by 8 "
                "and the maximum must exceed 32"
            )
    if weights_path is not None and model_name not in (
        "resnet18",
        "mobilenet_v2",
        "deeplabv3_mobilenet_v3_large",
        "efficientnet_b0",
        "densenet121",
        "convnext_tiny",
        "vit_b_16",
    ):
        raise ValueError("--weights is only supported for torchvision models")
    if vit_blocks is not None and (
        model_name != "vit_b_16" or not 0 <= vit_blocks <= 12
    ):
        raise ValueError("--vit-blocks requires vit_b_16 and a value from 0 to 12")
    if vit_layer_tokens and model_name != "vit_b_16":
        raise ValueError("--vit-layer-tokens requires vit_b_16")
    if vit_attention_stages and not vit_layer_tokens:
        raise ValueError("--vit-attention-stages requires --vit-layer-tokens")
    torch.manual_seed(0)
    model = model_type()
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        checkpoint_model = (
            model.model if isinstance(model, DeepLabV3MobileNet) else model
        )
        checkpoint_model.load_state_dict(state_dict)
    if vit_blocks is not None:
        model.encoder.layers = model.encoder.layers[:vit_blocks]
    if vit_layer_tokens:
        model = VitLayerTokens(model, vit_attention_stages)
    model = model.eval().to(device="musa", dtype=dtype)
    example = _make_input(model_name, input_shape, dtype)
    original_input = _snapshot_tensor(example)
    with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
        expected = model(example)

    dynamic_dimensions = {}
    max_shape = list(input_shape)
    if dynamic_batch_max is not None:
        dynamic_dimensions[0] = Dim("batch", min=1, max=dynamic_batch_max)
        max_shape[0] = dynamic_batch_max
    if dynamic_spatial_max is not None:
        if model_name == "densenet121":
            dynamic_dimensions[2] = 8 * Dim(
                "height_div8", min=spatial_minimum // 8, max=dynamic_spatial_max // 8
            )
            dynamic_dimensions[3] = 8 * Dim(
                "width_div8", min=spatial_minimum // 8, max=dynamic_spatial_max // 8
            )
        else:
            dynamic_dimensions[2] = Dim(
                "height", min=spatial_minimum, max=dynamic_spatial_max
            )
            dynamic_dimensions[3] = Dim(
                "width", min=spatial_minimum, max=dynamic_spatial_max
            )
        max_shape[2:] = [dynamic_spatial_max, dynamic_spatial_max]
    dynamic_shapes = (dynamic_dimensions,) if dynamic_dimensions else None
    export_model_graph = (
        export_with_musa_sdpa_decomposition
        if model_name in ("transformer_block", "causal_attention", "vit_b_16")
        else export
    )
    if model_name in ("resnet18", "mobilenet_v2") and dynamic_spatial_max is not None:
        export_model_graph = export_with_musa_pooling_shapes
    exported = export_model_graph(model, (example,), dynamic_shapes=dynamic_shapes)
    compile_spec = [MusaBackend.generate_method_name_compile_spec("forward")]
    with _serialization_schema():
        manager = to_edge_transform_and_lower(
            exported,
            partitioner=[MusaPartitioner(compile_spec)],
        ).to_executorch()
        output_dir.mkdir(parents=True, exist_ok=True)
        pte_path = Path(save_pte_program(manager, model_name, str(output_dir)))

    input_path = output_dir / "input-0.bin"
    expected_path = output_dir / "expected-0.bin"
    _write_input(
        original_input, input_path, tuple(max_shape) if dynamic_shapes else None
    )
    expected = _write_tensor(expected, expected_path)
    ptd_path = output_dir / "aoti_musa_blob.ptd"
    if not ptd_path.exists():
        raise RuntimeError(f"AOTI external data was not created: {ptd_path}")
    return pte_path, ptd_path, input_path, expected, model, input_shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_FACTORIES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--runner-repeats", type=int, default=1)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dynamic-batch-max", type=int)
    parser.add_argument("--dynamic-spatial-max", type=int)
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--vit-blocks", type=int, help="run only the first 0-12 ViT encoder blocks"
    )
    parser.add_argument("--vit-layer-tokens", action="store_true")
    parser.add_argument("--vit-attention-stages", action="store_true")
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    args = parser.parse_args()
    if args.runner_repeats < 1 or (args.runner_repeats > 1 and args.runner is None):
        parser.error("--runner-repeats requires a runner and a positive count")
    (args.output_dir / "result.json").unlink(missing_ok=True)

    pte_path, ptd_path, input_path, expected, model, input_shape = export_model(
        args.model,
        args.output_dir,
        DTYPES[args.dtype],
        args.input_size,
        args.weights,
        args.batch_size,
        args.dynamic_batch_max,
        args.dynamic_spatial_max,
        args.vit_blocks,
        args.vit_layer_tokens,
        args.vit_attention_stages,
    )
    atol = args.atol if args.atol is not None else TOLERANCES[args.dtype]
    rtol = args.rtol if args.rtol is not None else TOLERANCES[args.dtype]
    result = {
        "model": args.model,
        "dtype": args.dtype,
        "input_dtype": "int64" if args.model in TOKEN_MODEL_NAMES else args.dtype,
        "input_size": args.input_size,
        "batch_size": args.batch_size,
        "dynamic_batch_max": args.dynamic_batch_max,
        "dynamic_spatial_max": args.dynamic_spatial_max,
        "weights": str(args.weights) if args.weights is not None else None,
        "vit_blocks": args.vit_blocks,
        "vit_layer_tokens": args.vit_layer_tokens,
        "vit_attention_stages": args.vit_attention_stages,
        "atol": atol,
        "rtol": rtol,
        "pte": str(pte_path),
        "ptd": str(ptd_path),
        "input": str(input_path),
        "expected_shape": list(expected.shape),
        "runner": None,
        "runner_repeats": args.runner_repeats,
        "artifact_version": 1,
        "artifact_cases": [],
    }

    if args.runner is not None:
        result["runner"] = str(args.runner)
        if args.dynamic_spatial_max is not None:
            max_size = args.dynamic_spatial_max
            min_size = 24 if args.model == "bilinear_unet" else 48
            max_batch = args.dynamic_batch_max or args.batch_size
            if args.model == "densenet121":
                min_size = 32
                small_width = min_size + 8
                alternate_height = max(min_size, max_size - 8)
                alternate_width = max(min_size, max_size - 16)
            else:
                small_width = min_size + 6
                alternate_height = max_size - 1
                alternate_width = max_size - 9
            input_shapes = dict.fromkeys(
                (
                    input_shape,
                    (
                        1 if args.dynamic_batch_max else args.batch_size,
                        3,
                        min_size,
                        small_width,
                    ),
                    (max_batch, 3, max_size, max_size),
                    (args.batch_size, 3, alternate_height, alternate_width),
                )
            )
            max_shape = (max_batch, 3, max_size, max_size)
        elif args.dynamic_batch_max is not None:
            input_shapes = dict.fromkeys(
                (
                    input_shape,
                    (1, *input_shape[1:]),
                    (args.dynamic_batch_max, *input_shape[1:]),
                )
            )
            max_shape = (args.dynamic_batch_max, *input_shape[1:])
        else:
            input_shapes = (input_shape,)
            max_shape = None
        for current_shape in input_shapes:
            label = (
                (
                    f"{current_shape[0]}-{current_shape[-2]}x{current_shape[-1]}"
                    if args.dynamic_batch_max is not None
                    else f"{current_shape[-2]}x{current_shape[-1]}"
                )
                if args.dynamic_spatial_max is not None
                else str(current_shape[0])
            )
            if current_shape == input_shape:
                current_input_path = input_path
                current_expected = expected
                current_expected_path = args.output_dir / "expected-0.bin"
            else:
                example = _make_input(args.model, current_shape, DTYPES[args.dtype])
                original_input = _snapshot_tensor(example)
                with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
                    current_expected = model(example)
                current_input_path = args.output_dir / f"input-{label}.bin"
                _write_input(original_input, current_input_path, max_shape)
                current_expected_path = args.output_dir / f"expected-{label}.bin"
                current_expected = _write_tensor(
                    current_expected, current_expected_path
                )
            artifact_case = {
                "input_shape": list(current_shape),
                "expected_shape": list(current_expected.shape),
                "input": record_artifact(current_input_path),
                "expected": _record_tensor_artifact(
                    current_expected, current_expected_path
                ),
                "outputs": [],
            }
            result["artifact_cases"].append(artifact_case)
            output_prefix = args.output_dir / f"runner-output-{label}"
            completed = _run_runner(
                args.runner,
                pte_path,
                ptd_path,
                current_input_path,
                output_prefix,
                current_shape if max_shape is not None else None,
            )
            output_path = Path(f"{output_prefix}-0.bin")
            actual = torch.frombuffer(
                bytearray(output_path.read_bytes()), dtype=DTYPES[args.dtype]
            )
            expected_flat = current_expected.detach().cpu().flatten()
            artifact_case["outputs"].append(_record_tensor_artifact(actual, output_path))
            if args.vit_layer_tokens:
                nan_counts = torch.isnan(
                    actual.reshape(current_expected.shape)
                ).sum(-1)
                if nan_counts.any():
                    raise AssertionError(
                        f"ViT stage NaNs for shape {current_shape}: "
                        f"{nan_counts.tolist()} ({output_path})"
                    )
            torch.testing.assert_close(actual, expected_flat, atol=atol, rtol=rtol)
            max_error = float((actual.float() - expected_flat.float()).abs().max())
            if args.model == "token_decoder":
                expected_token_ids = current_expected.argmax(dim=-1).cpu()
                actual_token_ids = actual.reshape(current_expected.shape).argmax(dim=-1)
                if not torch.equal(actual_token_ids, expected_token_ids):
                    raise AssertionError(
                        f"predicted token IDs differ for batch {current_shape[0]}: "
                        f"expected {expected_token_ids.tolist()}, "
                        f"actual {actual_token_ids.tolist()}"
                    )
                if current_shape == input_shape:
                    result["predicted_token_ids"] = actual_token_ids.tolist()
            if (
                args.model
                in (
                    "efficientnet_b0",
                    "vit_b_16",
                    "densenet121",
                    "convnext_tiny",
                )
                and not args.vit_layer_tokens
            ):
                expected_classes = current_expected.argmax(dim=-1).cpu()
                actual_classes = actual.reshape(current_expected.shape).argmax(dim=-1)
                if not torch.equal(actual_classes, expected_classes):
                    raise AssertionError(
                        f"{args.model} predicted classes differ for batch {current_shape[0]}"
                    )
                if current_shape == input_shape:
                    result["predicted_classes"] = actual_classes.tolist()
            if args.model == "deeplabv3_mobilenet_v3_large":
                expected_labels = current_expected.argmax(dim=1).cpu()
                actual_labels = actual.reshape(current_expected.shape).argmax(dim=1)
                if not torch.equal(actual_labels, expected_labels):
                    mismatched = (actual_labels != expected_labels).sum().item()
                    raise AssertionError(
                        f"segmentation pixel labels differ at {mismatched} positions"
                    )
                if current_shape == input_shape:
                    result["segmentation_pixel_labels_match"] = True
            if (
                args.dynamic_batch_max is not None
                and args.dynamic_spatial_max is not None
            ):
                result.setdefault("dynamic_shape_results", []).append(
                    {"shape": list(current_shape), "max_error": max_error}
                )
            elif args.dynamic_batch_max is not None:
                result.setdefault("dynamic_batch_results", []).append(
                    {"batch_size": current_shape[0], "max_error": max_error}
                )
            elif args.dynamic_spatial_max is not None:
                result.setdefault("dynamic_spatial_results", []).append(
                    {"shape": list(current_shape), "max_error": max_error}
                )
            if current_shape == input_shape:
                result["output"] = str(output_path)
                result["max_error"] = max_error
                result["runner_stdout"] = completed.stdout
                result["runner_stderr"] = completed.stderr
            for repeat in range(1, args.runner_repeats):
                repeat_prefix = args.output_dir / f"runner-output-{label}-repeat-{repeat}"
                _run_runner(
                    args.runner,
                    pte_path,
                    ptd_path,
                    current_input_path,
                    repeat_prefix,
                    current_shape if max_shape is not None else None,
                )
                repeated = torch.frombuffer(
                    bytearray(Path(f"{repeat_prefix}-0.bin").read_bytes()),
                    dtype=DTYPES[args.dtype],
                )
                artifact_case["outputs"].append(
                    _record_tensor_artifact(repeated, Path(f"{repeat_prefix}-0.bin"))
                )
                if args.vit_layer_tokens:
                    nan_counts = torch.isnan(
                        repeated.reshape(current_expected.shape)
                    ).sum(-1)
                    if nan_counts.any():
                        raise AssertionError(
                            f"ViT stage NaNs for shape {current_shape} "
                            f"repeat {repeat}: {nan_counts.tolist()} "
                            f"({repeat_prefix}-0.bin)"
                        )
                torch.testing.assert_close(
                    repeated,
                    expected_flat,
                    atol=atol,
                    rtol=rtol,
                    msg=f"{args.model} shape {current_shape} repeat {repeat}",
                )
                result.setdefault("runner_repeat_results", []).append(
                    {
                        "shape": list(current_shape),
                        "repeat": repeat,
                        "max_error": float(
                            (repeated.float() - expected_flat.float()).abs().max()
                        ),
                    }
                )

    else:
        result["artifact_cases"].append(
            {
                "input_shape": list(input_shape),
                "expected_shape": list(expected.shape),
                "input": record_artifact(input_path),
                "expected": _record_tensor_artifact(
                    expected, args.output_dir / "expected-0.bin"
                ),
                "outputs": [],
            }
        )
    verify_artifacts(args.output_dir, result)
    result_json = json.dumps(result, indent=2, sort_keys=True)
    (args.output_dir / "result.json").write_text(result_json + "\n")
    print(result_json)


if __name__ == "__main__":
    main()
