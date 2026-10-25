# MUSA regression entry point

Run from a checkout with the required submodules, a working MUSA GPU and SDK,
and a Python environment containing compatible PyTorch, TorchMUSA, Triton,
torchvision, and ExecuTorch export dependencies. CMake 3.26 or newer, a C++
compiler, and `flatc` must be available. The script does not install packages
or modify the selected Python environment.

```bash
python .ci/scripts/test_musa.py \
  --suite all \
  --build-dir /tmp/executorch-musa-ci \
  --output-dir /tmp/executorch-musa-ci-results
```

Use the desired Python executable to select the environment. `--cmake`,
`--ctest`, `--flatc`, and `--musa-home` accept explicit tool/SDK paths.
`--musa-home` defaults to `$MUSA_HOME` or `/usr/local/musa`.
The selected source checkout takes precedence on child processes' Python
import path. The build-tree shims, selected TorchMUSA libraries, and SDK are
placed on their runtime library path; no installed ExecuTorch C++ package is
required. Use a dedicated build directory.

## Suites

| Suite | Coverage |
| --- | --- |
| `native` | Build and CTest for addmm/BMM, convolution, BatchNorm, and activation shims |
| `export` | Complete `backends.musa.tests.test_musa_aoti` unittest suite |
| `runner` | FP16/BF16 fused-bias convolution and FP32/FP16/BF16 Hardsigmoid precision tests through PTE/PTD and `executor_runner` |
| `models` | Model export, PTE/PTD loading, runner execution, and MUSA eager output comparison |
| `all` | All of the above, in order |

The four native CTest executables contain 33 GoogleTest cases. Each executable
runs in three fresh processes by default (`--native-repeats`). The padded
no-bias addmm regression is enabled by default again after the earlier packing
fix and repeated successful validation. Native tests run serially to avoid
GPU contention. Model export and execution are also serial; do not launch
concurrent regression invocations against the same checkout because the
existing export schema helper temporarily updates source schema files.

The default model matrix is MLP, SimpleCNN, ResNet18, and MobileNetV2 in FP32,
FP16, and BF16. All use bounded dynamic batches 1/2/3, with two fresh runner
executions per shape. ResNet18 and MobileNetV2 additionally exercise bounded
dynamic spatial sizes. `--models` also accepts EfficientNet-B0, DenseNet121,
and ConvNeXt-Tiny. DenseNet's spatial dimensions use multiples of eight;
ConvNeXt and SimpleCNN keep fixed spatial sizes. These are random-weight
execution checks, not pretrained or dataset-level accuracy measurements.

For a smaller end-to-end matrix while retaining all native, compiler, and
precision tests:

```bash
python .ci/scripts/test_musa.py --suite all \
  --models mlp simple_cnn --dtypes fp32 bf16 \
  --build-dir /tmp/executorch-musa-ci \
  --output-dir /tmp/executorch-musa-ci-results
```

## Failure handling and artifacts

Missing GPU/tooling, command failure, timeout, zero selected CTest entries,
skipped native execution, and empty/skipped Python suites produce a nonzero
exit status. Inherited `GTEST_*` filters and repeat settings are removed so
they cannot silently reduce coverage. `--timeout` limits each subprocess
(default: 3,600 seconds); `--jobs` controls build parallelism only.

The output directory contains `summary.json` with source revision, dirty-tree
status, device and framework versions, commands, durations, and step status.
It is reset before preflight, so a failed rerun cannot retain an old overall
success status. Only a final `status: passed` means the requested suite
completed. Each step has a separate log; `native.xml` is CTest JUnit output,
and model directories retain PTE/PTD, inputs, outputs, and `result.json`.
JUnit counts the four native executables, not individual GoogleTest cases;
the verbose native log retains the case-level results and repeated runs.

The orchestration itself has CPU-only tests requiring no MUSA installation:

```bash
python .ci/scripts/tests/test_musa_ci.py -v
```

This entry point is ready to invoke from a provisioned MUSA CI worker. No
remote workflow or self-hosted runner registration is configured by it.
Do not run untrusted pull-request code on a privileged self-hosted worker.
