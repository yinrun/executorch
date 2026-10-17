# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export and execute representative models through the MUSA AOTI backend.

The script uses fixed-shape inputs for each export. It is an acceptance
smoke test for the complete export path, not a benchmark. The
``narrow_resnet18`` model keeps the ResNet-18 block topology but uses narrow
channels; ``resnet18`` is the standard torchvision model with optional weights.
``mobilenet_v2`` exercises depthwise/grouped convolutions and ReLU6.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.export import Dim, export

from executorch.backends.musa import MusaBackend, MusaPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program


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
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class SimpleVideoCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 4, 3, padding=1),
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


MODEL_FACTORIES = {
    "mlp": (Mlp, (1, 16)),
    "simple_cnn": (SimpleCnn, (1, 3, 32, 32)),
    "simple_video_cnn": (SimpleVideoCnn, (1, 1, 4, 8, 8)),
    "simple_decoder": (SimpleDecoder, (1, 3, 8, 8)),
    "transformer_block": (TransformerBlock, (1, 4, 16)),
    "causal_attention": (CausalAttention, (1, 4, 16)),
    "embedding_classifier": (EmbeddingClassifier, (1, 4)),
    "token_encoder": (TokenEncoder, (1, 4)),
    "token_decoder": (TokenDecoder, (1, 4)),
    "narrow_resnet18": (Resnet18, (1, 3, 32, 32)),
    "resnet18": (_torchvision_resnet18, (1, 3, 64, 64)),
    "mobilenet_v2": (_torchvision_mobilenet_v2, (1, 3, 64, 64)),
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


def _write_tensor(tensor: torch.Tensor, path: Path) -> None:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    path.write_bytes(raw.tobytes())


def _make_input(
    model_name: str, input_shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    if model_name in TOKEN_MODEL_NAMES:
        return torch.randint(
            TOKEN_VOCAB_SIZE, input_shape, device="musa", dtype=torch.int64
        )
    return torch.randn(input_shape, device="musa", dtype=dtype)


def _write_input(tensor: torch.Tensor, path: Path, max_batch: int | None) -> None:
    _write_tensor(tensor, path)
    if max_batch is not None:
        padding = (
            (max_batch - tensor.shape[0]) * tensor[0].numel() * tensor.element_size()
        )
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
        if model_name not in ("resnet18", "narrow_resnet18", "mobilenet_v2"):
            raise ValueError("--input-size is only supported for CNN models")
        if input_size < 32:
            raise ValueError("--input-size must be at least 32")
        input_shape = (batch_size, 3, input_size, input_size)
    if weights_path is not None and model_name not in ("resnet18", "mobilenet_v2"):
        raise ValueError("--weights is only supported for torchvision models")
    torch.manual_seed(0)
    model = model_type()
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    model = model.eval().to(device="musa", dtype=dtype)
    example = _make_input(model_name, input_shape, dtype)
    with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
        expected = model(example)

    dynamic_shapes = (
        ({0: Dim("batch", min=1, max=dynamic_batch_max)},)
        if dynamic_batch_max is not None
        else None
    )
    exported = export(model, (example,), dynamic_shapes=dynamic_shapes)
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
    _write_input(example, input_path, dynamic_batch_max)
    _write_tensor(expected, expected_path)
    ptd_path = output_dir / "aoti_musa_blob.ptd"
    if not ptd_path.exists():
        raise RuntimeError(f"AOTI external data was not created: {ptd_path}")
    return pte_path, ptd_path, input_path, expected, model, input_shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_FACTORIES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dynamic-batch-max", type=int)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    args = parser.parse_args()

    pte_path, ptd_path, input_path, expected, model, input_shape = export_model(
        args.model,
        args.output_dir,
        DTYPES[args.dtype],
        args.input_size,
        args.weights,
        args.batch_size,
        args.dynamic_batch_max,
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
        "weights": str(args.weights) if args.weights is not None else None,
        "atol": atol,
        "rtol": rtol,
        "pte": str(pte_path),
        "ptd": str(ptd_path),
        "input": str(input_path),
        "expected_shape": list(expected.shape),
        "runner": None,
    }

    if args.runner is not None:
        result["runner"] = str(args.runner)
        batches = (
            dict.fromkeys((args.batch_size, 1, args.dynamic_batch_max))
            if args.dynamic_batch_max is not None
            else (args.batch_size,)
        )
        for batch_size in batches:
            if batch_size == args.batch_size:
                current_input_path = input_path
                current_expected = expected
            else:
                example = _make_input(
                    args.model, (batch_size, *input_shape[1:]), DTYPES[args.dtype]
                )
                with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
                    current_expected = model(example)
                current_input_path = args.output_dir / f"input-{batch_size}.bin"
                _write_input(example, current_input_path, args.dynamic_batch_max)
                _write_tensor(
                    current_expected, args.output_dir / f"expected-{batch_size}.bin"
                )
            current_shape = (batch_size, *input_shape[1:])
            output_prefix = args.output_dir / f"runner-output-{batch_size}"
            completed = _run_runner(
                args.runner,
                pte_path,
                ptd_path,
                current_input_path,
                output_prefix,
                current_shape if args.dynamic_batch_max is not None else None,
            )
            output_path = Path(f"{output_prefix}-0.bin")
            actual = torch.frombuffer(
                bytearray(output_path.read_bytes()), dtype=DTYPES[args.dtype]
            )
            expected_flat = current_expected.detach().cpu().flatten()
            torch.testing.assert_close(actual, expected_flat, atol=atol, rtol=rtol)
            max_error = float((actual.float() - expected_flat.float()).abs().max())
            if args.model == "token_decoder":
                expected_token_ids = current_expected.argmax(dim=-1).cpu()
                actual_token_ids = actual.reshape(current_expected.shape).argmax(dim=-1)
                if not torch.equal(actual_token_ids, expected_token_ids):
                    raise AssertionError(
                        f"predicted token IDs differ for batch {batch_size}: "
                        f"expected {expected_token_ids.tolist()}, "
                        f"actual {actual_token_ids.tolist()}"
                    )
                if batch_size == args.batch_size:
                    result["predicted_token_ids"] = actual_token_ids.tolist()
            if args.dynamic_batch_max is not None:
                result.setdefault("dynamic_batch_results", []).append(
                    {"batch_size": batch_size, "max_error": max_error}
                )
            if batch_size == args.batch_size:
                result["output"] = str(output_path)
                result["max_error"] = max_error
                result["runner_stdout"] = completed.stdout
                result["runner_stderr"] = completed.stderr

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
