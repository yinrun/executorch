# MUSA AOTI model acceptance

`aoti_model_acceptance.py` validates the complete MUSA path for three fixed
shape models:

- `mlp`: Linear + ReLU + Linear + ReLU;
- `simple_cnn`: Conv2d + ReLU + MaxPool2d + Conv2d + Flatten + Linear;
- `resnet18`: ResNet-18 block topology with narrow channels and a 32x32 input.

The script writes the PTE, external AOTI data, raw input, expected output, and
runner output to the selected directory. It compares the `executor_runner`
output with the MUSA eager result; no CPU execution is used as a correctness
baseline.

## Prerequisites

- MUSA and `torch_musa` are available to the selected Python environment;
- the MUSA-compatible Triton package required by TorchInductor is installed;
- an ExecuTorch `executor_runner` built with `EXECUTORCH_BUILD_MUSA=ON`.

For example, from the repository root:

```bash
export PYTHON=/home/qianfan/miniforge3/envs/apxinf-pi05/bin/python3
export PYTHONPATH=/tmp/executorch-pydeps:$PWD/src:$PWD
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
```

Without `--runner`, the script only exports the PTE/PTD and writes the eager
reference files. The runner command automatically adds the MUSA, Torch, and
Python shared-library directories used by the current environment.
