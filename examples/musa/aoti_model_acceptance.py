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
from torch.export import export

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
    "narrow_resnet18": (Resnet18, (1, 3, 32, 32)),
    "resnet18": (_torchvision_resnet18, (1, 3, 64, 64)),
    "mobilenet_v2": (_torchvision_mobilenet_v2, (1, 3, 64, 64)),
}
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
) -> subprocess.CompletedProcess[str]:
    command = [
        str(runner),
        f"--model_path={pte_path}",
        f"--data_path={ptd_path}",
        f"--inputs={input_path}",
        f"--output_file={output_prefix}",
        "--print_output=summary",
    ]
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
) -> tuple[Path, Path, Path, torch.Tensor]:
    if not torch.musa.is_available():
        raise RuntimeError("MUSA is not available")

    model_type, input_shape = MODEL_FACTORIES[model_name]
    if input_size is not None:
        if model_name not in ("resnet18", "narrow_resnet18", "mobilenet_v2"):
            raise ValueError("--input-size is only supported for CNN models")
        if input_size < 32:
            raise ValueError("--input-size must be at least 32")
        input_shape = (1, 3, input_size, input_size)
    if weights_path is not None and model_name not in ("resnet18", "mobilenet_v2"):
        raise ValueError("--weights is only supported for torchvision models")
    torch.manual_seed(0)
    model = model_type()
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    model = model.eval().to(device="musa", dtype=dtype)
    example = torch.randn(input_shape, device="musa", dtype=dtype)
    with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
        expected = model(example)

    exported = export(model, (example,))
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
    _write_tensor(example, input_path)
    _write_tensor(expected, expected_path)
    ptd_path = output_dir / "aoti_musa_blob.ptd"
    if not ptd_path.exists():
        raise RuntimeError(f"AOTI external data was not created: {ptd_path}")
    return pte_path, ptd_path, input_path, expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_FACTORIES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    args = parser.parse_args()

    pte_path, ptd_path, input_path, expected = export_model(
        args.model,
        args.output_dir,
        DTYPES[args.dtype],
        args.input_size,
        args.weights,
    )
    atol = args.atol if args.atol is not None else TOLERANCES[args.dtype]
    rtol = args.rtol if args.rtol is not None else TOLERANCES[args.dtype]
    result = {
        "model": args.model,
        "dtype": args.dtype,
        "input_size": args.input_size,
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
        output_prefix = args.output_dir / "runner-output"
        completed = _run_runner(
            args.runner,
            pte_path,
            ptd_path,
            input_path,
            output_prefix,
        )
        output_path = Path(f"{output_prefix}-0.bin")
        actual = torch.frombuffer(
            bytearray(output_path.read_bytes()), dtype=DTYPES[args.dtype]
        )
        expected_flat = expected.detach().cpu().flatten()
        torch.testing.assert_close(
            actual,
            expected_flat,
            atol=atol,
            rtol=rtol,
        )
        result["runner"] = str(args.runner)
        result["output"] = str(output_path)
        result["max_error"] = float(
            (actual.float() - expected_flat.float()).abs().max()
        )
        result["runner_stdout"] = completed.stdout
        result["runner_stderr"] = completed.stderr

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
