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
| `runner` | Fused-bias convolution, depthwise algorithm rounding, and Hardsigmoid precision tests through PTE/PTD and `executor_runner` |
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

Supply local torchvision state dictionaries to exercise pretrained weights;
repeat `--weights MODEL PATH` for multiple selected models. The selected
checkpoint is used for every requested dtype and recorded in `summary.json`.
The script does not download checkpoints or relax numerical tolerances.

```bash
python .ci/scripts/test_musa.py --suite models \
  --models mobilenet_v2 --dtypes fp16 \
  --weights mobilenet_v2 /checkpoints/mobilenet_v2-7ebf99e0.pth \
  --build-dir /tmp/executorch-musa-ci \
  --output-dir /tmp/executorch-musa-pretrained-results
```

Random-weight MobileNetV2 FP16 logits can underflow entirely to zero; passing
that smoke test alone is not evidence of pretrained numerical accuracy.
On M1000 (TorchMUSA 2.9.0.post1+musa5.1.2mp22), the original hard-coded
IMPLICIT_GEMM convolution path failed pretrained FP16 acceptance with maximum
logit error `0.01953125`. Stage diagnostics located the first differences in
the first inverted-residual depthwise convolution. Selecting the muDNN
recommended forward algorithm, as TorchMUSA eager does, reduces the maximum
error to `0.00048828125`: all four shapes and both runner repetitions pass
the unchanged `atol=rtol=0.01` criterion. Use the same input as an acceptance
run to investigate individual operators:

```bash
python examples/musa/aoti_backbone_diagnostics.py \
  --model mobilenet_v2 --dtype fp16 \
  --module-path features.1.conv.0 \
  --input /tmp/executorch-musa-pretrained-results/models/mobilenet_v2-fp16/input-0.bin \
  --weights /checkpoints/mobilenet_v2-7ebf99e0.pth \
  --runner /tmp/executorch-musa-ci/executor_runner \
  --output-dir /tmp/executorch-musa-depthwise-diagnostics
```

`--include-classifier` instead reports every backbone stage and the classifier.
Intermediate outputs can change fusion, so always rerun full-model acceptance
after a precision fix.

The forward shim queries the muDNN C++ recommendation API used by TorchMUSA
and explicitly maps its algorithm enum to the C API used for execution.
Workspace queries and both biased/unbiased forward calls use that selection.
FP32 remains in scalar mode (TF32 disabled); FP16/BF16 use tensor mode.
Transposed convolution retains its existing backward-data algorithm path.
The checkpoint-independent depthwise runner regression compares FP32, FP16,
and BF16 outputs exactly against MUSA eager at batches 1/2/3.

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

## Validation snapshot (2026-10-25)

Validated on M1000 with PyTorch/TorchMUSA
`2.9.0.post1+musa5.1.2mp22`. The fresh in-tree build includes the runner and
native CTest targets without relying on an installed ExecuTorch C++ package.

| Check | Observed result |
| --- | --- |
| Fresh `all` smoke with MLP/SimpleCNN, FP32/BF16 | Build, native tests, original 38-test AOTI suite, original 15 precision runner cases, and 24 model runs passed |
| New pooling-shape tests | Two targeted tests passed separately; not a single combined 40-test invocation |
| Missing-device preflight | Nonzero exit, failed report, no configure/build steps |
| CI orchestration unit tests | Eight passed |
| Native tests after convolution algorithm fix | 33 cases, each repeated three times, passed |
| Precision runner tests after fix | Three unittest methods, 24 runner cases, passed |
| Depthwise regression against old shim | Six exact-comparison failures across FP32/FP16; the fixed shim passes all nine dtype/batch cases |
| Pretrained MobileNetV2 FP16, fresh export | Four batch/spatial shapes, two runs each, passed; maximum logit error `0.00048828125`, top-1 labels match in every run |
| Random-weight MobileNetV2 FP32/FP16/BF16 artifact replay | 24 runs passed; FP16's zero logits remain only a smoke check |
| Random-weight ResNet18 FP32/FP16/BF16 artifact replay | 24 runs against freshly recomputed eager references passed; maximum errors `3.099442e-6`, `0.001953125`, `1.907349e-6`, respectively |
| Pretrained DeepLabV3-MobileNetV3 BF16 artifact replay | Eight runs against freshly recomputed eager references passed; maximum logit error `0.03125`, all pixel labels match |

The CNN and segmentation shapes are `(2,3,64,64)`, `(1,3,48,54)`,
`(3,3,80,80)`, and `(2,3,79,71)`. Artifact replay reuses existing PTE/PTD;
it is not a new export. These checks establish execution/numerical parity
on the tested inputs, not dataset accuracy or performance targets.

An initial saved-reference replay stopped at ResNet18 BF16's `(2,3,79,71)`
case: the historical `expected-2-79x71.bin` has 16 zero tail entries that
disagree with both its original runner output and fresh MUSA eager output.
Five eager recomputations agree exactly with the original runner output.
The old file and failed replay report were retained, not rewritten to pass.
All ResNet18 and DeepLab references were then recomputed from the same model
factory, seed/checkpoint, and saved inputs for the successful replay above.
The origin of the historical file discrepancy is not established; saved
reference integrity checks remain follow-up work.
