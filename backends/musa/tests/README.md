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
required. Use a dedicated build directory. The `all` and `deployment` suites
also require GNU `readelf` and `patchelf` (`--patchelf` accepts an explicit path).

## Suites

| Suite | Coverage |
| --- | --- |
| `native` | Build and CTest for addmm/BMM, convolution, BatchNorm, activation shims, and native/legacy device metadata |
| `export` | Complete `backends.musa.tests.test_musa_aoti` unittest suite |
| `runner` | Frozen tensor snapshots/artifact failures, native MUSA and legacy device tags, foreign-backend rejection, plus convolution and Hardsigmoid precision through PTE/PTD and `executor_runner` |
| `models` | Model export, PTE/PTD loading, runner execution, and MUSA eager output comparison |
| `deployment` | Package validation and relocated MLP FP32/SimpleCNN BF16 execution without Torch/Python runtime libraries |
| `all` | All of the above, in order |

The five native CTest executables contain 37 GoogleTest cases. Each executable
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

## Runtime-only deployment regression

```bash
python .ci/scripts/test_musa.py --suite deployment \
  --patchelf /path/to/patchelf \
  --build-dir /tmp/executorch-musa-ci \
  --output-dir /tmp/executorch-musa-deployment-results
```

This suite builds the runner, validates the packager, exports two models, and
moves each package to a directory containing spaces. It runs dynamic batches
1/2/3 with a clean subprocess environment, `LD_BIND_NOW=1`, and no preload.
It compares results against MUSA eager and inspects loader initialization
records to reject Torch/Python libraries and Python package directories.
The launcher, runner, and ExecuTorch libraries come from the moved package;
the MUSA SDK and GPU driver remain external prerequisites. This is a Linux,
same-architecture test of the current static-core/shared-shims configuration,
not a universal wheel or cross-platform SDK test.

See `examples/musa/package_aoti.py` and the example README for packaging
commands. Old artifacts that directly link Torch/Python must be re-exported;
the packager refuses them instead of deleting ELF dependencies after linking.

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
JUnit counts the five native executables, not individual GoogleTest cases;
the verbose native log retains the case-level results and repeated runs.

### Tensor artifact consistency

Model acceptance freezes each eager reference into an owned, contiguous CPU
snapshot. Writing `expected-*.bin`, numerical comparisons, and class/token
checks all use that snapshot rather than separate device-to-host transfers.
The shared tensor writer checks its immediate byte-for-byte readback.

New `result.json` files contain `artifact_version: 1` and `artifact_cases`.
Each case records input/output shapes and relative filenames, byte lengths,
and SHA-256 checksums for its input, expected output, and every runner output.
Reference/output records are checked against the exact tensor bytes used in
the comparison. Before publishing the result, acceptance checks all files
again, including input padding and output sizes. If a file changes while the
runner executes, acceptance fails without publishing a successful result.

The `models`/`all` CI suites also run eight standalone artifact tests and a
separate verification process after each model acceptance step. Verification
needs only the Python standard library and works after moving the report
directory together with its tensor files:

```bash
python -S examples/musa/acceptance_artifacts.py /path/to/model/result.json
python -S .ci/scripts/tests/test_musa_artifacts.py -v
```

Missing files, same-length corruption (including zeroed tails), truncation,
invalid padding, incomplete repetitions, or inconsistent recorded dynamic
shapes fail verification. Legacy reports without the manifest are rejected;
regenerate them instead of creating checksums for unverified old references.
This is tensor artifact integrity checking, not an independent accuracy test,
and does not hash PTE/PTD or authenticate the report itself.

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
The origin of the historical file discrepancy is not established. New runs
now use the frozen-reference and integrity checks described above; the old
file is retained as evidence, not silently repaired.

After the artifact checks were added, both fresh ResNet18 BF16 and pretrained
MobileNetV2 FP16 exports passed their four dynamic shapes with two runner
executions each, followed by the independent standard-library verifier.
Maximum errors remained `1.907349e-6` and `0.00048828125`, respectively.
The complete runner suite passed seven unittest methods (the original 24
precision runner cases plus four snapshot/artifact test methods), alongside
eight standalone artifact tests and eight CI orchestration tests. MLP FP32
export-only mode also passed artifact verification without claiming execution.

A copy of the fresh ResNet report and tensor files passed verification after
relocation. Zeroing the last 16 BF16 values of its reference without changing
file length then produced a checksum failure; the original files were not
modified. Separate tests inject reference, input, or earlier-output mutation
during runner execution and confirm that acceptance refuses to write
`result.json` even when numerical comparisons against the frozen reference pass.

## Native device metadata validation (2026-10-25)

After commit `884b38e481`, new exports use serialized MUSA device 2 and the
AOTI boundary uses PrivateUse1 device 20. The in-tree MUSA build was rebuilt
and the following checks ran on the same M1000/TorchMUSA environment above:

| Check | Observed result |
| --- | --- |
| Complete `all` suite, MLP/SimpleCNN FP32/BF16 | Passed, including 24 dynamic-batch model executions and artifact verification |
| Native CTest | Five executables, 36 cases, each repeated three times; no skips |
| Complete AOTI unittest suite | 41 tests passed in one invocation |
| Complete runner unittest suite | Nine methods passed: 28 successful real runner executions, one intentional foreign-backend rejection, and snapshot/artifact checks |
| Fresh native-tagged ResNet18 BF16 | Four combined batch/spatial shapes, two runs each; maximum error `1.9073486328125e-6` |
| Fresh native-tagged pretrained MobileNetV2 FP16 | Four combined batch/spatial shapes, two runs each; maximum error `0.00048828125` |
| Historical CUDA-tagged `MusaBackend` ResNet18/MobileNetV2 artifacts | 16 replay runs passed against frozen references; top-1 IDs match; original PTE/PTD hashes and tensor manifests remain unchanged |
| Device metadata inspection | Deserialized both new CNN PTEs and verified device 2 in arenas, tensors, and non-CPU weights; historical PTEs retain device 1 |
| Growing KV-cache BF16 | Two chained runner steps passed; token IDs `[28, 32]` match eager, key/value outputs are exact, maximum logit error `0.0078125` |
| Additional host-side checks | 12 portable-device cases, three non-MUSA-compiled header probes, eight CUDA/ROCm metadata codec tests, and eight CI orchestration tests passed |

The device test includes SlimTensor allocation/copy/item, distinct wire/AOTI
IDs, invalid device strings, native/legacy allocator registrations, and an
ATen device-bridge round trip. It does not establish a full ATen-mode build.
The header probes verify unchanged CUDA ABI IDs and rejection of MUSA storage
without the MUSA build define; they are not CUDA/ROCm hardware regressions.
The weight-mapping fixture was subsequently made independent of constructing
a registered `torch.device("musa")`, and its focused test passed again.
The KV-cache check feeds the first runner's cache outputs into the second
runner through explicit tensor I/O; it is not validation of the runtime's
internal mutable-state or paged-KV allocator.

Local evidence from this run:

- Full CI: `/tmp/et-musa-native-device-all-fixed-20261025/summary.json`.
- New CNN reports: `/tmp/et-musa-native-device-followup-20261025/resnet18-bf16/summary.json`
  and `/tmp/et-musa-native-device-followup-20261025/mobilenet_v2-fp16/summary.json`.
- Legacy replay records: `/tmp/et-musa-native-device-followup-20261025/summary.json`.
  This combined follow-up report is **failed**, not passed: after all 16 replay
  checks succeeded, the additional generic device-propagation suite failed.
- Independently completed KV check: `/tmp/et-musa-native-device-kv-growing-bf16-20261025/result.json`.

The supplemental `exir.tests.test_propagate_device_pass` invocation ran 24
tests with 19 errors: the installed Torch partitioner does not accept
`skip_horizontal_fusion`. Loading the unchanged partitioner helper from
pre-change commit `dfda07b86e` reproduces the same `TypeError`; this is not a
native-device regression. The failed log and baseline reproduction are kept
at `/tmp/et-musa-native-device-followup-20261025/device-propagation.log` and
`/tmp/et-musa-native-device-propagation-baseline-20261025.log`. No compatibility
monkeypatch was applied to make this suite pass. Resolve the generic
partitioner/Torch version mismatch before claiming full EXIR regression
coverage. Standalone deployment packaging, performance baselines, full
ATen-mode integration, and CUDA/ROCm hardware validation remain separate work.
