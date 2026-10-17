# MUSA AOTI model acceptance

`aoti_model_acceptance.py` validates the complete MUSA path for five models:

- `mlp`: Linear + ReLU + Linear + ReLU;
- `simple_cnn`: Conv2d + ReLU + MaxPool2d + Conv2d + Flatten + Linear;
- `narrow_resnet18`: ResNet-18 block topology with narrow channels and a 32x32 input;
- `resnet18`: unmodified torchvision ResNet18 with random weights and a 64x64 input;
- `mobilenet_v2`: torchvision MobileNetV2 with depthwise convolutions and a 64x64 input.

The script writes the PTE, external AOTI data, raw input, expected output, and
runner output to the selected directory. It compares the `executor_runner`
output with the MUSA eager result; no CPU execution is used as a correctness
baseline.

Inputs and weights default to FP32. `--dtype fp16` or `--dtype bf16` exercises
the same export and runner path in lower precision. The default absolute and
relative comparison tolerances are 1e-4 (FP32), 1e-2 (FP16), and 4e-2 (BF16);
override them with `--atol` and `--rtol` if needed. `--input-size 224` exports
ResNet or MobileNet with a 224x224 input instead of its default size. Each shape and
dtype combination requires its own export; dynamic shapes are not covered.
For torchvision models, `--weights /path/to/state_dict.pth` loads a
local PyTorch state dictionary (for example, a cached pretrained checkpoint)
before exporting. Without it, the model uses random weights; the script never
downloads weights automatically.

The MUSA backend disables TorchInductor's layout optimization so Conv1d/2d
fallback outputs and generated kernels agree on contiguous NCHW layout. The
eager reference disables muDNN TF32 to match the fallback's FP32 FMA math.

## Prerequisites

- MUSA and `torch_musa` are available to the selected Python environment;
- the MUSA-compatible Triton package required by TorchInductor is installed;
- `torchvision` is available for the standard `resnet18` model;
- `flatc` is available on `PATH` for PTE/PTD serialization;
- an ExecuTorch `executor_runner` built with `EXECUTORCH_BUILD_MUSA=ON`.

For example, from the repository root:

```bash
export PYTHON=/usr/bin/python3.12
export PYTHONPATH=/tmp/executorch-pydeps:$PWD/src:$PWD
export PATH=/tmp/executorch-flatc-build:$PATH
export RUNNER=/tmp/executorch-musa-cmake-1792075449/executor_runner

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model mlp \
  --output-dir /tmp/executorch-musa-acceptance/mlp \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model simple_cnn \
  --output-dir /tmp/executorch-musa-acceptance/simple_cnn \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model resnet18 \
  --output-dir /tmp/executorch-musa-acceptance/resnet18 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model resnet18 --input-size 224 \
  --output-dir /tmp/executorch-musa-acceptance/resnet18_224 \
  --runner "$RUNNER"
```

To validate official pretrained ResNet18 weights, download
`https://download.pytorch.org/models/resnet18-f37072fd.pth` separately and pass
the local checkpoint with `--weights`. Its SHA-256 is
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
This checks inference against MUSA eager using a random input, not ImageNet
accuracy or performance.
MobileNetV2 IMAGENET1K_V2 weights are available at
`https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth` (SHA-256
`7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f`).

On M1000 with torch-musa 2.9.0.post1+musa5.1.2mp22, the following exports
passed `executor_runner` versus MUSA eager (maximum absolute output error):

| Model | Input | Dtype | Weights | Max error |
| --- | --- | --- | --- | ---: |
| MLP | 1x16 | FP16 / BF16 | Random | 4.88e-4 / 3.91e-3 |
| Simple CNN | 1x3x32x32 | FP16 / BF16 | Random | 2.44e-4 / 1.95e-3 |
| ResNet18 | 1x3x224x224 | FP32 / FP16 | Official pretrained | 6.05e-6 / 3.91e-3 |
| ResNet18 | 1x3x64x64 | BF16 | Official pretrained | 0 |
| MobileNetV2 | 1x3x224x224 | FP32 | Official pretrained | 2.03e-6 |
| MobileNetV2 | 1x3x64x64 | FP16 | Official pretrained | 1.37e-2 (`--atol 0.02`) |

These are single-input correctness checks, not shape-generalization or
throughput measurements. Pretrained MobileNetV2 BF16 at 64x64 does not yet
meet the default tolerance: the maximum difference is 0.140625 and 121/1000
logits differ beyond `--atol 0.02 --rtol 0.04`. Do not count this configuration
as verified until the precision difference is understood. In isolated prefix
exports, the first two feature blocks match bit-for-bit. The next block's
pointwise Conv2d also matches, but its following BatchNorm differs at 4 of
98,304 BF16 values (maximum 0.03125); the discrepancy grows through later
blocks. Investigate BF16 BatchNorm rounding before relaxing the tolerance.

Without `--runner`, the script only exports the PTE/PTD and writes the eager
reference files. The runner command automatically adds the MUSA, Torch, and
Python shared-library directories used by the current environment.
