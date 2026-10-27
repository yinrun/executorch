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
| `native` | Build and CTest for addmm/BMM, convolution, BatchNorm, activation shims, native/legacy device metadata, and SlimTensor copy-stream selection |
| `export` | Complete `backends.musa.tests.test_musa_aoti` unittest suite |
| `runner` | Frozen tensor snapshots/artifact failures, native MUSA and legacy device tags, foreign-backend rejection, convolution/Hardsigmoid precision, and activation/decoder output reuse through PTE/PTD and `executor_runner` |
| `models` | Model export, PTE/PTD loading, runner execution, and MUSA eager output comparison |
| `deployment` | Package/benchmark validation and relocated MLP FP32/SimpleCNN BF16 execution without Torch/Python runtime libraries, including synchronized timing and fresh-process repeatability |
| `all` | All of the above, in order |

The six native CTest executables contain 63 GoogleTest cases. Each executable
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
ConvNeXt-Tiny, MobileNetV3-Small/Large, and DeepLabV3-MobileNetV3-Large.
Both MobileNetV3 classifiers and DeepLab use bounded dynamic spatial sizes.
DenseNet's spatial dimensions use multiples of eight;
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
Batch 1 also exercises the ordinary, uninstrumented runner; batches 2/3
collect three synchronized samples after two warmups. Each model then runs
two fresh benchmark processes through the standard-library-only driver
(`python -S`), checking the final outputs byte-for-byte and sample counts.
The MLP package additionally rejects zero measured executions, count overflow,
and server-mode benchmarking without writing a successful timing report.
Four benchmark unit tests check statistics, invalid samples, package hashes,
and path traversal; the deployment suite includes these tests automatically.
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

## Runtime-only validation snapshot (2026-10-25)

Commit `b530b491dc` removes the MUSA AOTI libraries' direct Torch/Python
dependencies at link time and supplies the previously missing MUSA guard
and PrivateUse1 symbols in the ExecuTorch runtime. A clean-process check
initially failed on `aoti_torch_create_musa_stream_guard`; providing the ABI,
rather than preloading TorchMUSA to supply it, resolved the failure.

On M1000 with the same SDK/TorchMUSA environment:

| Check | Observed result |
| --- | --- |
| Deployment CI | Seven packager unit tests and relocated MLP FP32/SimpleCNN BF16 batches 1/2/3 passed; six real clean-environment runner executions |
| Native CTest | 37 cases, each repeated three times, passed |
| Existing runner regression suite | All nine methods passed again, including native/legacy device tags, foreign-backend rejection, precision checks, and snapshots |
| Compiler-context isolation | MUSA AOTI library filtering, unchanged eager/CPU/CUDA/compile-only options, and exception restoration passed |
| CI orchestration | Nine tests passed |
| Freshly exported/packaged ResNet18 BF16 and pretrained MobileNetV2 FP16 | Two runs each at `(2,3,64,64)` passed |
| Full dynamic-spatial package replay | Four shapes per model, two runs each, all 16 passed; maximum errors `1.9073486328125e-6` and `0.00048828125`; top-1 IDs match |
| Historical Torch-linked model packaging | Rejected with a re-export instruction; no partial package published |
| Historical Torch-linked MLP execution | Still passes with the development loader environment; maximum error `1.4901161193847656e-8` |
| MobileNetV2 file-access trace | From `/tmp`, execution passed without accessing the checkout, build tree, Python package directories, or TorchInductor cache |

The full spatial replay uses the earlier frozen-reference reports. Before
execution, it verifies those tensor manifests and compares the complete
multiset of PTD buffer SHA-256 hashes against each new package, establishing
identical weights. The new export-only reports contain only the initial
input/reference, not four dynamic cases; they are not used to claim a
16-case fresh-reference run. All relocated processes use `LD_BIND_NOW=1`
and loader tracing, without Torch/Python preloads. The observed initialized
libraries are restricted to the package, MUSA SDK, driver, and system libraries.

Local evidence:

- `/tmp/et-musa-deployment-ci-20261025/summary.json`
- `/tmp/et-musa-deployment-native-20261025/summary.json`
- `/tmp/et-musa-deployment-runner-20261025/summary.json`
- `/tmp/et-musa-deployment-models-20261025/summary.json` (four initial-shape executions)
- `/tmp/et-musa-deployment-spatial-replay-20261025/summary.json` (16 dynamic executions)
- `/tmp/et-musa-deployment-files-20261025.trace`
- `/tmp/et-musa-deployment-package-checks-20261025.log`
- `/tmp/et-musa-deployment-legacy-runner-20261025.log`

A runnable MLP package is retained at `/tmp/et-musa-runtime-package-20261025`;
relocated CNN packages are under `/tmp/et-musa-deployment-models-20261025`.
These checks do not claim a different host/GPU ISA, absence of the external
SDK/driver requirement, dataset-level model accuracy, or performance gains.
The separate generic EXIR/Torch partitioner mismatch described above remains
open; no version-compatibility workaround is hidden in this deployment path.
The follow-up full 42-test AOTI compiler suite passes without skips at
`6a79a2c593` (528.949 seconds reported by unittest). Evidence is retained at
`/tmp/et-musa-runtime-only-full-export-20261025/summary.json` and `export.log`.

## Synchronized benchmark validation (2026-10-25)

Commit `f9409a2eff` adds opt-in synchronized runner samples, a standard-library
benchmark driver, and deployment regression coverage. Seven package tests,
four benchmark tests, and the relocated deployment test pass, as do all nine
CI orchestration tests. The deployment test covers ordinary batch-1 execution,
instrumented batches 2/3, two fresh driver processes per model, exact final
output repeatability, and invalid benchmark flags. Logs are under
`/tmp/et-musa-benchmark-deployment-20261025`.
The four benchmark unit tests also pass directly with `python -S` and only
the checkout on `PYTHONPATH`. After the model measurements, all nine existing
runner regression methods pass without skips (242.535 seconds), covering
convolution/Hardsigmoid precision, native and legacy device tags, and tensor
artifact failures. Evidence is at
`/tmp/et-musa-benchmark-runner-20261025/summary.json` and `runner.log`.

The performance baseline uses Release (`-O3 -DNDEBUG`) on aarch64/Cortex-A78,
M1000, driver `5.2.2-M1000`, SDK `/usr/local/musa-5.1.0`, and exports made
with TorchMUSA `2.9.0.post1+musa5.1.2mp22`. Each model uses five fresh
processes, ten warmups per process, and fifty measured executions per
process: 250 measured samples per model. Model/inputs/references/package
files are hashed, outputs are checked exactly against previously accepted
runner outputs, and those references are also rechecked against the frozen
MUSA eager tensors at the original tolerances before measurement.

These are host-synchronized `Method::execute()` measurements, not pure
kernel latency or process-startup latency. The inclusive metric adds input
preparation and its synchronization. No loader tracing or concurrent
ExecuTorch export/regression work runs during the measurements. Clocks and
exclusive GPU access are not locked; the retained device snapshots and
per-process distributions expose, but do not eliminate, environmental noise.
The numbers are a baseline for these exact inputs, not a speedup claim,
cross-model ranking, dataset accuracy result, or production throughput.

| Model | Dtype | Input shape | Execute min (ms) | Execute median (ms) | Execute p90 (ms) | Prepare + execute median (ms) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| MLP, random weights | FP32 | 2x16 | 3.582620 | 3.972702 | 5.089428 | 3.978342 |
| SimpleCNN, random weights | BF16 | 2x3x32x32 | 2.977056 | 3.202298 | 3.718140 | 3.214298 |
| ResNet18, random weights | BF16 | 2x3x64x64 | 17.686656 | 19.257625 | 20.734833 | 19.293725 |
| MobileNetV2, pretrained | FP16 | 2x3x64x64 | 20.986793 | 23.206886 | 25.605579 | 23.250927 |

All four benchmark reports pass: 1,000 measured samples total and exact
final-output agreement for all twenty fresh processes. This checks one
shape per model, not the full dynamic-spatial matrix. SimpleCNN was freshly
exported and accepted at batches 1/2/3, twice each, before packaging. The
ResNet18 and MobileNetV2 PTE/PTD hashes match their previously relocated and
accepted runtime-only packages; only the runner is updated for timing.
Maximum differences between
the frozen MUSA eager references and the accepted runner outputs are
`1.4901161193847656e-8`, `0`, `1.9073486328125e-6`, and `0.00048828125`,
respectively. Timing repeats do not compare against eager anew or validate
every intermediate iteration.

Aggregate evidence is at
`/tmp/et-musa-benchmark-baseline-20261025/summary.json`. That directory also
contains `run_baseline.py`, device snapshots, the fresh SimpleCNN export,
four runtime packages, and one `<model>-timings` directory per model.
Each timing directory retains the manifest snapshot, frozen input/reference,
per-process output hashes, commands, raw samples, stdout/stderr, and summaries.
The minimum/median are pooled execution samples, not subprocess runtimes or
an average of process medians. The benchmark utility and exact timing
boundaries are documented in `examples/musa/README.md`.

Next performance work should collect a trace that separates host dispatch,
allocations/copies, synchronization, and GEMM/convolution kernels before
changing those paths. The tiny-MLP result alone cannot identify the dominant
overhead. Stateful KV-cache workloads need their own reset and workload
protocol rather than reusing these stateless warmups. The generic EXIR/Torch
partitioner version mismatch remains a separate compatibility task.

## MLP profiling and rejected handle-cache experiment (2026-10-26)

MUSA System Profiler `1.9.0-0a3ca05e9-internal-release-1.9` captured the
accepted, runtime-only MLP FP32 package on the same M1000. Its output remains
byte-identical to the previously accepted reference. Three warmups precede
five profiled iterations. For this capture, the main-thread device-sync API
events delimit preparation and execution; the extracted execution boundaries
are checked against every corresponding runner timing sample within 0.05 ms.
Only those five measured execution intervals are used below, not the entire
process trace or its 573 ms profiler initialization event.

Each profiled execution contains four `musaMalloc` calls and two `musaFree`
calls. Their non-overlapping, main-thread API durations account for about
86.8–87.8% of the 5.56–6.20 ms instrumented execution time. Source inspection
finds that both Linear calls create and destroy a muBLAS handle on each
invocation. The trace also contains two SGEMM kernels and two fused pointwise
kernels per iteration. Host allocation durations and GPU kernel durations
are separate categories, not additive components of one total.

A bounded thread-local handle-cache experiment reuses a handle only for the
same device and stream, recreates it on a device/stream switch, and destroys
it at thread exit. It does not share a workspace across threads or rebind a
live handle across streams. In the experimental MLP trace, steady-state
allocation count drops from four to two and synchronous free count from two
to zero. The five profiled execution samples fall to 0.946–1.020 ms, with
identical final MLP outputs. These instrumented samples are diagnostic
evidence, **not an accepted performance improvement**.

The optimization fails its correctness gate and is withdrawn from the
runtime source. The first complete native regression reports FP32 classifier
tail zeros, and a subsequent alternating old/new-library fresh-process
comparison reproduces the failure: the original library passes all 20
processes; the cached library passes 19 and fails one. That failed process
reads zero at output positions 996–999 on the second classifier execution;
an immediate second read, without rerunning GEMM, sees the expected `2.5`,
and the bias readback is `1`. This does not yet identify whether asynchronous
GEMM completion, memory visibility, or readback is responsible. No retry is
used to turn a failed comparison into a pass, and tolerances are unchanged.

Other checks deliberately do not override that failure:

- Thirty same-process repetitions of the three projection tests pass with
  both old and experimental libraries.
- Thirty repetitions of all matrix tests except the large attention-score
  test pass with both libraries.
- A transfer-only check reads immutable FP32 tensors of 1,000, 1,001 and
  1,999 elements 5,000 times each, repeated ten times: all 150,000 readbacks
  pass, without GEMM. However, a later full native run after withdrawing the
  handle cache fails that same transfer-only check at 1,999 elements,
  repetition 4,940, index 1,992. This reproduces tail zeros without GEMM or
  the experimental cache; the initial passing stress run was insufficient.

The retained regressions add queued calls across two nonblocking streams,
three worker threads switching streams and exiting, and immutable tail
readback stress. Classifier failures now retain first/second readback and
bias-tail diagnostics while still failing on the original mismatch. Runtime
code at that checkpoint remained on the per-call-handle path; the separate SDK-only readback
investigation and staging fix below address the reproduced transfer problem.

Those first traces, experimental patch/binary, and comparison logs were
captured under `/tmp/et-musa-profile-20261026`. A host restart later on October
26 removed the temporary directories before the follow-up acceptance finished.
The figures above are recorded observations from that earlier run, not
currently replayable artifacts. The experimental cache is absent from the
runtime source. New SDK-only reproduction and acceptance evidence is stored
in the persistent workspace location below. A passing rerun does not resolve
an earlier mismatch; the cache still needs validation with the readback fix
before it can be enabled or advertised as a performance improvement.

## Pageable-host readback isolation and staging fix (2026-10-26)

`examples/musa/diagnose_readback.cpp` reproduces the issue using only MUSA SDK
APIs, with no ExecuTorch, Torch, muBLAS, or GPU arithmetic. It uploads a
constant FP32 tensor once, synchronizes, and repeatedly reads immutable
device data into host memory. Every API reports success, but some ordinary
host-buffer readbacks have zero tail values; a second read sees `2.5` without
any device writes between them. Thus the failure can occur independently of
the handle-cache experiment and GEMM. This isolates a transfer/visibility
problem on the observed SDK/driver stack, not its internal vendor root cause.

On M1000, driver `5.2.2-M1000`, SDK `/usr/local/musa-5.1.0`, the standalone
probe checks 1,000, 1,001 and 1,999 FP32 elements, 100,000 copies per size:

| Raw SDK mode | Copies | Failed readbacks |
| --- | ---: | ---: |
| Pageable host, `musaMemcpyAsync` plus stream sync | 300,000 | 0 in this run |
| Pageable host, blocking `musaMemcpy` | 300,000 | 272, all at 1,001 elements |
| Pinned host, async copy plus stream sync | 300,000 | 0 |
| Pinned staging, async copy plus stream sync, then CPU byte copy | 300,000 | 0 |

This table is the persistent post-restart reproduction. The earlier run
observed 34 blocking-copy failures (23 at 1,001 elements and 11 at 1,999).
The zero-failure async run does not invalidate the earlier native async
failure: this is intermittent. Merely replacing async reads with the blocking
SDK call is demonstrably insufficient. No test retries or tolerance changes
are used to accept a failed readback.

The MUSA adapter in `extension/cuda/runtime_api.h` now detects an unregistered
host destination for explicit device-to-host copies. It copies into a pinned
staging allocation on the requested stream, waits for that stream, and copies
the completed bytes into the ordinary host destination before freeing the
staging allocation. This applies to both blocking and async adapter entry
points, so pageable D2H calls now complete synchronously. Registered/pinned
destinations retain their original SDK path and asynchronous behavior;
H2D, D2D, CUDA and ROCm paths are unchanged. Null pointers and zero-byte
requests retain the original SDK validation, and allocation/copy/sync/free
errors are propagated rather than converted into successful partial copies.

The workaround temporarily needs one extra pinned buffer of the copy size
and adds allocation/CPU-copy overhead for pageable D2H. It does not pool
buffers or promise unchanged performance. It is a correctness workaround
for the observed stack, not a claim that all MUSA versions or hardware share
the issue. The raw probe deliberately bypasses the adapter so future SDK
upgrades can be evaluated independently.

Reproduce the standalone comparison:

```bash
c++ -std=c++17 -O2 examples/musa/diagnose_readback.cpp \
  -I/usr/local/musa/include -L/usr/local/musa/lib \
  -Wl,-rpath,/usr/local/musa/lib -lmusart -o /tmp/diagnose_readback
for mode in pageable-async pageable-sync pinned-async pageable-staged; do
  /tmp/diagnose_readback "$mode" 100000 > "/tmp/$mode.jsonl" 2> "/tmp/$mode.stderr"
  echo "$mode exit=$?"
done
```

Exit status 1 denotes an observed mismatch; 2 denotes invalid arguments or
an SDK error. JSONL retains per-size counts, stderr records the first ten
failures with first/second readback values. Current evidence is in
`/home/qianfan/文档/ChatGPT/ETSupport/artifacts/musa-20261026/reports/readback-*.jsonl`
and matching `.stderr` files. The earlier
`/tmp/et-musa-profile-safe-native-20261026/summary.json` was a failed native
run after withdrawing the cache, not a passing acceptance report, and was
lost during the restart.

### Python references and exported weights

The C++ adapter does not intercept TorchMUSA's own copies. The shared
`backends/aoti/host_tensor.py` snapshot helper therefore also stages MUSA
tensors through an explicitly pinned CPU tensor, synchronizes completion,
and clones to owned, contiguous ordinary CPU storage. It detaches autograd
history and restores the selected device. Pin-allocation or synchronization
failures propagate; there is no fallback to an unsafe ordinary D2H copy.
CPU snapshots retain the previous owned-copy behavior.

Model inputs, eager reference outputs, backbone diagnostics, Llama/KV and
prefill/decode comparisons, and the runner precision tests use this helper.
KV-cache tensors used by later eager steps remain on the device: only the
comparison snapshots move to CPU. CPU tensor comparisons and argmax checks
are not CPU model execution or a new accuracy baseline.

MUSA weight serialization now uses the same pinned snapshot for each storage
chunk. The existing eight-MiB chunk bound, full-storage serialization, view
offset/stride metadata, SHA-256 digest, and failure cleanup remain unchanged.
CUDA/ROCm weight copies retain their prior path. Tests cover a strided view
with a nonzero offset, a 1,001-byte tail beyond an eight-MiB chunk, exact saved
bytes/digests, and removal of the partial file if staging fails. This avoids
protecting model outputs while leaving exported weights on the unsafe path.

### Post-restart acceptance

The toolchain, clean build, raw SDK reproduction, and follow-up results now
live under the persistent directory
`/home/qianfan/文档/ChatGPT/ETSupport/artifacts/musa-20261026`.
`environment.sh` records the local Python, CMake, flatc, patchelf, build and
SDK paths. Missing Python build helpers were installed into that directory,
not into the global TorchMUSA environment. The earlier `/tmp` artifacts are
not required to reproduce these results.

| Check | Result | Evidence relative to the persistent directory |
| --- | --- | --- |
| Native shims and device tests | 41 cases, three complete repetitions, no skips | `reports/native/summary.json`, `reports/native/native.log` |
| Snapshot and weight-metadata unit tests | 17 passed | `reports/weight-snapshots.log` |
| Final runner regression | 14 passed, no skips | `reports/runner-final/summary.json`, `reports/runner-final/runner.log` |
| Final packaging, benchmark, deployment tests | 7 + 4 + 1 passed, no skips | `reports/deployment-final/summary.json` |
| ResNet18 BF16, random weights | Four dynamic shapes, two fresh runner processes per shape | `reports/models-final/summary.json` |
| BF16 toy decoder caches and dynamic prefill | Rolling/growing caches, prompt lengths 2/3/4, two decode steps | `reports/stateful-final/summary.json`, `reports/stateful-final.log` |

The snapshot tests are included in the runner suite; these rows must not be
summed as distinct test coverage. The native source is unchanged from
`064b05b72a`. Final runner acceptance includes the weight snapshot fix
`d7af9321ba`; deployment and model acceptance also include the Buck dependency
declaration in `47af3fdf21`. Reports record their exact revisions and dirty
state; documentation was still being updated during those runs. Buck itself
was not available for a build, and the weight-metadata unit tests do not
constitute CUDA/ROCm hardware regression.

The relocated deployment test runs MLP FP32 and SimpleCNN BF16 at batches
1/2/3, with and without benchmark flags. Dynamic-loader checks reject loaded
Torch, Python, c10 and TorchMUSA Python libraries. This validates the runtime
package on the current host, not deployment to another GPU or SDK version.

ResNet18 uses one PTE for `[2,3,64,64]`, `[1,3,48,54]`, `[3,3,80,80]`, and
`[2,3,79,71]`. Maximum absolute output errors versus frozen MUSA eager are
`1.9073486328125e-6`, `0`, `0`, and `0`; each independent replay has the same
result. The original BF16 tolerances remain `atol=rtol=0.04`. Its PTE/PTD,
frozen inputs/references, runner outputs and integrity manifest are retained
in `reports/models-final/models/resnet18-bf16`. This is a random-weight
functional regression, not pretrained ImageNet accuracy. The full previous
backbone/dtype matrix has not been rerun after the readback fix.

Stateful acceptance re-exports random-weight toy decoders, not a pretrained
LLM. Rolling-cache predictions are `[3, 32]`, growing-cache predictions are
`[40, 32]`, and all match MUSA eager exactly. Dynamic prefill predicts `16`
for length three, `34` for length two, and `19` for length four; the subsequent
two decode calls predict `[40, 32]`. Maximum absolute logit error across all
these checks is `0.0078125`, with zero K/V output error. Comparisons retain
the original BF16 tolerances and do not change the GPU caches passed to later
eager steps. PTE/PTD, initial inputs and runner cache outputs remain under
`reports/stateful-final`; the summary records revision `47af3fdf21` and the
documentation-only dirty state.

The next gate at this checkpoint was to refresh the broader pretrained/dtype acceptance matrix
with these protections, then repeat handle-reuse A/B validation and unprofiled
benchmarks. The handle cache was still disabled at that checkpoint. Neither the diagnostic speedup
nor old `/tmp` evidence is an accepted performance result for the new code.

## Pretrained matrix and prediction gate (2026-10-26)

Fresh exports at `94c81ce9b7` pass the complete ResNet18/MobileNetV2 matrix
below using official pretrained state dictionaries, generated inputs and
MUSA eager references. The checkpoints were downloaded into the persistent
`artifacts/musa-20261026/checkpoints` directory and verified against the full
SHA-256 values documented in `examples/musa/README.md`; `manifest.json`
retains their URLs, paths, sizes and digests. No checkpoint was replaced by
random weights and no comparison tolerance was relaxed.

| Pretrained model | Dtype | Maximum absolute logit error across eight runs |
| --- | --- | ---: |
| ResNet18 | FP32 | `6.67572021484375e-6` |
| ResNet18 | FP16 | `0.00390625` |
| ResNet18 | BF16 | `0.0001220703125` |
| MobileNetV2 | FP32 | `2.384185791015625e-6` |
| MobileNetV2 | FP16 | `0.00048828125` |
| MobileNetV2 | BF16 | `7.62939453125e-6` |

Every model/dtype uses one PTE at `[2,3,64,64]`, `[1,3,48,54]`,
`[3,3,80,80]`, and `[2,3,79,71]`, with two fresh runner processes per shape:
48 real executions in total. Reports, exported PTE/PTD, frozen inputs and
references, output tensors and integrity manifests are retained under
`reports/pretrained-baseline`. A separate audit verifies every recorded
artifact, repeats the numerical comparison and checks exact top-1 IDs for
all 48 outputs; its results are in `reports/pretrained-predictions.json`.
All references are finite and nonzero. This proves agreement with same-dtype
MUSA eager on these inputs, not ImageNet accuracy or identical class IDs
across precisions.

Review found that the acceptance script previously checked token/class/pixel
predictions only on the first execution of each shape, and did not include
ResNet18/MobileNetV2 in the class-ID check. Commit `d545db6030` centralizes
these checks and applies them to every replay as well. The first shape's
predictions remain in the top-level report; per-shape predictions and replay
predictions are also recorded. ViT intermediate feature outputs remain
excluded from classification checks.

The focused prediction/snapshot suite passes 11 tests. Its two new methods
include 24 acceptance scenarios spanning eight model types: matching
predictions, a first-run mismatch, and a repeated-run mismatch. The changed
logits deliberately remain inside the FP32 tolerance, so rejection proves
that the semantic gate is active rather than an incidental allclose failure.
Failed acceptance does not publish a successful `result.json`.
Evidence is in `reports/prediction-checks.log`. These tests validate acceptance
logic with synthetic CPU tensors plus the existing MUSA snapshot tests; they
do not execute CPU models or add a CPU accuracy baseline.

After committing the semantic gate, MobileNetV2 FP16 was freshly exported and
executed eight more times at the same four dynamic shapes, with the checks
integrated into the actual acceptance CLI. All runs pass at `d545db6030`
(clean source at launch), maximum logit error `0.00048828125`; every replay's
class IDs match the first run and same-dtype eager. Per-shape IDs are
`[22, 127]`, `[490]`, `[92, 734, 734]`, and `[420, 23]`, respectively.
The report, PTE/PTD and tensors are retained under
`reports/pretrained-prediction-gate/models/mobilenet_v2-fp16`, with the
overall result in `reports/pretrained-prediction-gate/summary.json`.

The ResNet18/MobileNetV2 pretrained precision gate is now satisfied on the
current stack. Their retained, checksummed inputs and references can be
used for handle-reuse A/B testing without changing weights or regenerating
inputs. Handle reuse was still disabled pending that validation and fresh
unprofiled measurements. Other backbones, another GPU/SDK, and dataset-level
accuracy are not covered by this matrix.

## Thread-local muBLAS handle reuse (2026-10-27)

Commit `3faf31def4` replaces per-operation muBLAS handle creation/destruction
with one bounded cache slot per host thread. A handle is reused only when
both device and stream match; a change destroys the previous handle on its
own device before creating another. Different threads never share a handle
or its workspace. Handles use host pointer mode, retain the existing GEMM
dtype/compute/alpha/beta choices, and are invalidated after a GEMM error.
Thread exit destroys the retained handle. Matrix shims select the tensor's
device for the operation and restore the caller's device afterwards.

This applies to the muBLAS paths in addmm/mm and padded/transposed/broadcast
bmm. The muDNN paths, operators' math and export format are unchanged. The
tradeoff is that the handle/workspace stays live until a stream/device switch
or thread exit, rather than being released after every operator. There is no
unbounded per-stream map, process-global handle or rebinding of a live
workspace across streams. Multi-GPU execution and device-reset scenarios
have not been validated on this single-M1000 host.

The initial native gate passes all 42 cases in each of three complete
repetitions, without skips. Queued-call coverage now issues groups of four
operations on the same nonblocking stream before switching, exercising
actual reuse as well as stream changes. A new regression creates and
destroys 16 streams within one worker thread while running FP32/FP16/BF16
addmm and bmm. Existing coverage also exercises three worker threads,
stream switching, thread exit and immutable readback tails.

The fixed-input A/B gate alternates old/new order over 20 rounds: 40 fresh
processes each run the three attention/classifier projection regressions,
with all 120 case executions passing. Loader traces on the first pair verify
that each process actually loads its intended shim library. The candidate
then replays all six pretrained ResNet18/MobileNetV2 dtype combinations and
the fixed MLP FP32 export: 54 fresh processes, covering all recorded dynamic
shapes. For each shape, one process executes once and another executes five
times; final outputs match frozen eager references and same-dtype class IDs.
All 54 final outputs are also byte-identical to the original implementation.
The multi-execution runs compare the final output, not every intermediate
forward; queued native tests separately check every queued output.

Persistent evidence under `artifacts/musa-20261026`:

- `reports/handle-cache-native/summary.json` and `native.log` retain the clean
  candidate build and native repetitions.
- `reports/handle-cache-ab/summary.json` records commands, binary hashes,
  projection logs, artifact hashes, numerical errors and prediction checks.
- `handle-cache/validate.py` reproduces the fixed-artifact gate;
  `handle-cache/candidate.patch` is byte-identical to the source change in
  `3faf31def4` (SHA-256 `f919bc7d5be541a88714c9ff68067de8b4490d684a0b4e510eed57745eca3b62`).
- The baseline `build` directory is retained unchanged; the candidate is
  built separately in `build-handle-cache`. These are not the withdrawn
  pre-readback-fix binaries whose `/tmp` artifacts were lost.

The committed implementation also passes 16 runner tests and the 7 package,
4 benchmark and 1 relocated-deployment tests, with no skips. The deployment
test again exercises MLP FP32 and SimpleCNN BF16 at batches 1/2/3 without
loading Torch/Python runtime libraries. See
`reports/handle-cache-runner-deployment/summary.json`. No binaries were
rebuilt between the fixed-input A/B gate and the measurements below; their
hashes are checked before benchmarking. The local `environment.sh` now
selects the accepted `build-handle-cache` through `BUILD_DIR`, retaining
`BASELINE_BUILD_DIR` for the original implementation.

### Unprofiled packaged measurements

On the same M1000, driver `5.2.2-M1000`, SDK `/usr/local/musa-5.1.0`, five
rounds alternate baseline/candidate process order. Every process runs ten
unmeasured warmups and fifty measured executions. Each model/variant thus
has 250 samples from five fresh processes, using identical packaged PTE/PTD,
weights, frozen input bytes and shapes. Both variants use the pinned readback
fix; only the native handle implementation/device guard differs. All thirty
final benchmark outputs match the accepted baseline bytes. Timings include
host dispatch and device-synchronized method execution, not process startup,
model loading, first-use initialization or output readback.

| Model | Dtype and shape | Baseline min / median (ms) | Cache min / median (ms) | Median speedup |
| --- | --- | ---: | ---: | ---: |
| MLP, random weights | FP32, 2x16 | 3.496562 / 4.342714 | 0.641088 / 0.681969 | 6.368x |
| ResNet18, pretrained | BF16, 2x3x64x64 | 17.600772 / 19.083512 | 15.927030 / 16.946866 | 1.126x |
| MobileNetV2, pretrained | FP16, 2x3x64x64 | 20.404403 / 22.088569 | 19.072427 / 20.584051 | 1.073x |

The medians including input preparation are respectively
`4.349054 -> 0.685689`, `19.117333 -> 16.971087`, and
`22.115668 -> 20.615590` ms. The candidate's median is lower in every paired
round for all three models, but these are workload/host-specific observations,
not a speedup guarantee. Individual samples contain scheduling outliers;
the table reports pooled minimum/median rather than a best round or an average
of round medians. The CNNs here both use pretrained weights; do not compare
their numbers directly with a historical random-weight baseline.

`reports/handle-cache-benchmark/summary.json` retains all commands, package
fingerprints, per-process summaries, pooled statistics and device snapshots.
Subdirectories retain raw timing samples, frozen inputs/references, final
output hashes and logs. `handle-cache/benchmark.py` reproduces the paired
protocol. No profiler is active during these measurements, and intermediate
timed outputs are not compared individually.

### Fresh trace confirmation

Separate direct-binary MLP captures, with three warmups and five measured
executions, reproduce the allocation change. Every measured baseline
iteration contains four `musaMalloc` and two `musaFree` calls; every cached
iteration contains two and zero, respectively. Non-overlapping main-thread
allocation/free API durations occupy 84.5–85.5% of the profiled baseline
execution interval, versus 0.46–0.49% for the cache. These host intervals are
not added to overlapping GPU/kernel or driver durations.

`handle-cache/analyze_profile.py` joins the SQLite event-name table and
derives execution intervals from the runner's synchronization boundaries.
It verifies each of the five intervals against its corresponding timing JSON
sample within 0.05 ms, excluding load, warmup and teardown. Both profiled
outputs remain byte-identical to the accepted baseline. Full `.msys-rep`
files, commands, timing/output data and analysis are retained under
`reports/handle-cache-profile`, not `/tmp`. The profiled latencies are
diagnostic only; the accepted speedups come from the unprofiled table above.

This validates the bounded muBLAS cache on the tested stack. Further work
should profile the remaining dispatch and muDNN costs and expand stateful
and model/dtype coverage before extending handle reuse to other libraries.

## Rejected stream-ordered workspace experiment (2026-10-27)

The experiment in `11b488ded3` allocates nonempty muDNN convolution workspace with
`musaMallocAsync` and releases it with `musaFreeAsync` on the convolution's
stream. This covers forward and transposed convolution without changing
algorithm selection, dtype, accumulation or model artifacts. A device guard
selects the input device before contiguous copies and allocations and restores
the caller's device after cleanup. muDNN handles are still per operation;
there is no new process-global or cross-stream workspace cache.

**Not enabled:** `59f45a3378` restores the original convolution implementation
after the unprofiled A/B measurements below failed to show a benefit. The
queued multi-thread/multi-stream regression remains. The local default stays
`build-handle-cache`; the trial binaries and evidence are preserved separately.

Only `musaErrorNotSupported` falls back to synchronous allocation and matching
synchronous free. Allocation failures such as OOM are returned to the caller,
not retried using another allocator. Successful calls check the explicit
workspace release before publishing the output; error exits retain RAII
cleanup. Device memory-pool retention/reuse is controlled by the SDK rather
than an application-level unbounded map.

The compatibility build passes all 43 native cases in three repetitions,
without skips. The new convolution regression uses two host threads with
two nonblocking streams each, FP32/FP16/BF16, and twelve queued convolutions
per thread/dtype. It retains and checks every output against analytically
computed border/interior values, exercising both same-stream reuse and
stream switches. The standalone SDK probe also passes 100 repetitions of
async allocation, memset, pinned readback and async free on a nonblocking
stream. These checks do not establish support on other devices or SDKs.

An external allocator interposer validates the compatibility/error branches:
all 12 convolution tests pass with 63 injected `NotSupported` results and
63 synchronous fallbacks. Two injected OOM results make the queued test return
`Error::Internal`, with zero synchronous fallback allocations. Its nonzero
GoogleTest exit is expected by the fault-injection harness. The first harness
incorrectly required an error log from a build with `ET_LOG_ENABLED=0`; that
failed report is retained, and the corrected gate checks return values and
allocator call counts instead of disabled logging.

Evidence under `artifacts/musa-20261026` includes
`reports/cnn-workspace-compat-regression/native.log`,
`reports/cnn-workspace-allocator-faults-v2/summary.json`,
`cnn-workspace/test_allocator_faults.py`, and `cnn-workspace/probe_async.log`.
The retained `build-handle-cache` remains the synchronous-workspace baseline;
`build-cnn-workspace-compat` contains the withdrawn trial implementation. The
earlier `build-cnn-workspace` trial did not yet include the compatibility
fallback and is not the final candidate.

The final trial candidate also passes ten fresh processes of the complete
12-case convolution executable and 54 fixed-artifact model processes:
pretrained ResNet18 and MobileNetV2 in FP32/FP16/BF16, plus MLP FP32. Each
recorded dynamic shape is executed in separate processes with one and five
forwards. All 54 final outputs match the same-dtype MUSA eager references,
applicable prediction checks, and the baseline bytes exactly. Every queued
native output is checked, whereas these multi-forward model replays check
the final output only. No CPU model baseline or dataset-level accuracy is
claimed. `reports/cnn-workspace-compat-validation/summary.json` retains all
commands, artifact/binary hashes and comparisons. The source patch SHA-256
is `fe5cd5a6c72a94ed9fac90a5667eeab1ecfb059c6da1feac23121f1a19b8029f`.

The same binaries pass all 16 runner, 7 package, 4 benchmark and 1 relocated
deployment tests without skips. The deployment gate covers MLP FP32 and
SimpleCNN BF16 at batches 1/2/3 without Torch/Python runtime libraries;
`reports/cnn-workspace-runner-deployment/summary.json` records this gate.
These tests do not rebuild the binaries used for correctness or timing.

### A/B result and withdrawal

Five rounds alternate baseline/candidate process order; each process runs
10 warmups and 50 measured executions, yielding 250 samples per model/variant.
The baseline already includes the accepted muBLAS handle cache and pinned
readback fix. Identical packaged PTE/PTD and frozen inputs are used; all 30
final outputs match their reference bytes. No profiler runs during timing.
The table reports pooled device-synchronized method execution, excluding
process startup, model loading, first-use initialization and output readback.

| Model | Dtype and shape | Baseline min / median (ms) | Trial min / median (ms) | Median latency change |
| --- | --- | ---: | ---: | ---: |
| MLP, random weights | FP32, 2x16 | 0.598928 / 0.672329 | 0.644809 / 0.691050 | +2.8% |
| ResNet18, pretrained | BF16, 2x3x64x64 | 16.009417 / 17.221234 | 16.898470 / 18.250748 | +6.0% |
| MobileNetV2, pretrained | FP16, 2x3x64x64 | 18.635453 / 20.472178 | 20.812643 / 22.900090 | +11.9% |

Individual rounds are noisy: the trial median is higher in three of five
ResNet rounds and four of five MobileNet rounds. MLP has no convolution and
is a control, not evidence of a convolution-induced slowdown. These results
do not establish a universal SDK regression; they provide no dependable
reason to replace the simpler allocator on the tested workload/stack.
Raw samples, commands, device snapshots and package fingerprints are in
`reports/cnn-workspace-benchmark/summary.json` and its subdirectories.

Fresh traces confirm that synchronous workspace frees disappear, but host
allocation/free time does not. In every measured ResNet iteration,
`musaFree` goes from 20 to zero and `musaFreeAsync` from 43 to 63; in MobileNet
they change from 35 to zero and 106 to 141. The ResNet allocation/free
interval union is approximately 17.5-18.9 ms before versus 19.0-19.5 ms after;
MobileNet is 20.8-22.7 versus 25.5-28.9 ms. Renaming the allocation/free API
does not eliminate its observed host cost. These are main-thread profiler
interval unions, not GPU execution time or unprofiled benchmark latencies;
they are not summed with overlapping driver/kernel intervals.

`reports/cnn-workspace-profile-baseline` and
`reports/cnn-workspace-profile-candidate` retain both traces, verified output
bytes and timing JSON. `cnn-workspace/analyze_profile.py` verifies measured
intervals against runner timing samples and computes the non-overlapping
API durations. A future experiment should test bounded device/stream-local
workspace reuse that actually reduces allocator calls, including memory
retention, stream/device changes, thread exit and failure-path coverage,
rather than assume async APIs alone improve performance.

### Retained implementation and standard-size inputs

After withdrawal, the retained synchronous-workspace shim passes all 43
native cases three times, including the new queued convolution regression.
The new test executables are run with the retained `build-handle-cache`
libraries, with loader traces verifying the selected shim in all five
processes. The tested runtime source is restored exactly; trial binaries
are not overwritten or relabeled as the retained implementation.

New pretrained exports cover standard 224x224 spatial inputs at batches
1/2/3: ResNet18 BF16 and MobileNetV2 FP16. First the trial passes two fresh
runner processes for every shape; then the retained implementation replays
the same artifacts in 12 fresh processes, with one and five forwards per
shape. All final outputs match their same-dtype MUSA eager references and
class IDs and are byte-identical to the trial outputs. Maximum absolute
errors are `0.00390625` for ResNet18 and `0.00048828125` for MobileNetV2.
These are generated-input correctness checks, not ImageNet accuracy or
224x224 performance measurements. FP32 and the other low-precision pairing
are not newly validated at this size by this gate.

Exports, checksummed weights/inputs/references and trial runner outputs are
in `reports/cnn-workspace-224`; retained-runner commands, binary hashes,
loader traces, comparisons and native logs are in
`reports/cnn-workspace-retained-validation/summary.json` and its directory.
The local `environment.sh` continues selecting `build-handle-cache`.

## Bounded convolution workspace reuse experiments (2026-10-27)

The next experiment retains one workspace slot per host thread, capped at
64 MiB. It is reusable only on the same device and stream with sufficient
capacity; growth or a device/stream change retires the previous allocation.
Larger requests use a standalone allocation freed before returning. The
64 MiB limit bounds retained workspace per thread, not total process/device
memory; a large standalone request may coexist with the retained slot.
Thread exit releases the slot on its owning device. No muDNN algorithm,
precision, weights or model artifacts change.

The event-based prototype in `2653ed4927` records completion and enqueues a
wait before reuse. This protects even against a destroyed stream's raw handle
being recycled, but its first unprofiled gate is mixed. Five alternating
rounds with 10 warmups/50 measured executions per process produce 250 samples
per model/variant, using the same protocol as the earlier experiment:

| Model | Dtype and shape | Baseline min / median (ms) | Event-cache min / median (ms) |
| --- | --- | ---: | ---: |
| MLP control | FP32, 2x16 | 0.634368 / 0.731169 | 0.627048 / 0.695588 |
| Pretrained ResNet18 | BF16, 2x3x64x64 | 16.462847 / 18.770516 | 15.106711 / 17.740624 |
| Pretrained MobileNetV2 | FP16, 2x3x64x64 | 18.825517 / 22.941410 | 19.500007 / 23.958801 |

ResNet improves by 1.058x median but MobileNet regresses to 0.958x. The MLP
control also varies despite not using convolution. This is not a broadly
accepted speedup. `reports/workspace-cache-benchmark` retains all 30 runs and
their byte-exact final output checks. Fresh traces in
`reports/workspace-cache-profile` confirm zero workspace malloc/free calls
after warmup, yet substantial tensor-free costs remain. Event/API and GPU
intervals must not be added together.

### Stream identity and correctness gates

The follow-up uses `musaStreamGetId` with SDK headers at least 5.1.0. The
installed SDK documents IDs as unique for the program's lifetime, unlike raw
stream addresses. Matching device, handle and ID therefore allows same-stream
reuse without per-convolution event commands. Eviction, thread exit and
failed-call cleanup synchronize the owning device before freeing memory;
they never synchronize a potentially destroyed old stream handle. If built
with older headers, or if the runtime reports `NotSupported`, the event path
is used instead. Other identity-query errors propagate. Older SDK builds and
multi-GPU/device-reset behavior are not validated on this host.

The standalone SDK probe verifies 1,000 unique IDs across stream creation and
destruction. For 10,000 empty-stream API iterations, ID lookup takes 2.824 ms
of host time versus 757.774 ms for event record/wait pairs. This is a diagnostic
microbenchmark, not a model speedup. The installed header's event contract
also specifies that waits capture the recorded state at call time and are
not changed by subsequent recordings.

Both prototypes pass all 44 native cases in three repetitions, without skips.
The additional regression retains inputs/outputs while growing and shrinking
spatial sizes (17/33/25/49/17/41/25/17), both on a stable stream and across
stream destruction/recreation, for FP32/FP16/BF16. It checks all output values
after completion and also exercises thread-exit cleanup.

The stream-ID version passes 19 external allocator/SDK fault-injection cases:
event fallback plus the ID path. Six equal-size requests allocate/free once;
the forced growth sequence allocates/frees three times. Exactly 64 MiB is
cacheable; 64 MiB plus one byte allocates/frees on all six requests and never
creates cache events. Tracked workspace bytes reach zero at process exit.
Single injected allocation, identity, event creation/record/wait/sync/destroy,
free and device-synchronization errors propagate as `Error::Internal`; a
second GoogleTest repetition in the same process succeeds. Fault cases
intentionally produce one failed first repetition, checked by the harness.
Forced workspace sizes are never smaller than muDNN's actual request.

Evidence under `artifacts/musa-20261026` includes
`reports/workspace-cache-native-v2`,
`reports/workspace-cache-native-stream-id`,
`reports/workspace-cache-faults-stream-id`, and
`workspace-cache/probe-stream-id.log`. The interposer and reproducible gate
are `workspace-cache/faults.cpp` and `workspace-cache/test_faults.py`.
These fault tests validate the fallback on the current SDK, not installation
or behavior of an older SDK. Graph capture remains unsupported by the MUSA
backend; this experiment does not change that boundary.

### Stream-ID unprofiled measurements

Commit `50b806963e` selects the stream-ID path. On the same M1000 and SDK,
five alternating paired rounds again use 10 warmups and 50 measured
executions per process. All 30 final outputs remain byte-identical to the
saved references. The three-model first-shape correctness smoke gate passes
before timing; this timing gate alone does not replace the broader model
and deployment matrix.

| Model | Dtype and shape | Baseline min / median (ms) | Stream-ID cache min / median (ms) | Median speedup |
| --- | --- | ---: | ---: | ---: |
| MLP control | FP32, 2x16 | 0.641464 / 0.687420 | 0.612338 / 0.676605 | 1.016x |
| Pretrained ResNet18 | BF16, 2x3x64x64 | 16.068228 / 16.784857 | 14.026851 / 14.677700 | 1.144x |
| Pretrained MobileNetV2 | FP16, 2x3x64x64 | 18.776112 / 20.202919 | 17.522109 / 19.354622 | 1.044x |

Both CNNs improve in every paired round. MLP remains a non-convolution
control, and the measured variation there is not a cache-induced speedup.
Timings are pooled device-synchronized method execution, not startup, load,
first-use initialization, readback or 224x224 inference. The baseline already
contains the accepted muBLAS cache and pinned readback fix. See
`reports/workspace-cache-smoke-stream-id` and
`reports/workspace-cache-benchmark-stream-id` for commands, hashes, exact
output checks and raw timing samples. These are host/workload-specific
measurements, not a performance guarantee across models or SDKs.

### Full gates and 224x224 measurements

The committed stream-ID implementation passes the full frozen matrix:
54 fresh runner processes across all six pretrained ResNet18/MobileNetV2
FP32/FP16/BF16 combinations and MLP FP32, with every recorded dynamic shape
and one/five forwards per process. All final outputs are byte-identical to
the retained baseline and pass eager-reference and applicable prediction
checks. Three additional fresh convolution processes pass all 13 cases.
The 224x224 gate replays ResNet18 BF16 and MobileNetV2 FP16 at batches 1/2/3
in 12 fresh processes with one/five forwards, again byte-exact to the prior
accepted outputs. All 44 native cases additionally pass three repetitions
in that gate. Runner/package/benchmark/relocated-deployment tests pass
16/7/4/1 methods without skips. See `reports/workspace-cache-full-stream-id`,
`reports/workspace-cache-224-stream-id`, and
`reports/workspace-cache-deployment-stream-id`.

A separate 224x224 A/B measurement uses the same frozen exports and the same
five-round, 10-warmup/50-measured protocol, retaining 250 samples per variant:

| Model | Dtype and shape | Baseline min / median (ms) | Stream-ID cache min / median (ms) | Median speedup |
| --- | --- | ---: | ---: | ---: |
| MLP control | FP32, 2x16 | 0.604570 / 0.662511 | 0.626850 / 0.679031 | 0.976x |
| Pretrained ResNet18 | BF16, 2x3x224x224 | 21.885440 / 22.539007 | 18.160540 / 18.977291 | 1.188x |
| Pretrained MobileNetV2 | FP16, 2x3x224x224 | 35.275338 / 36.923062 | 32.480695 / 33.743177 | 1.094x |

MobileNet improves in all five paired rounds. ResNet improves in three:
its candidate process medians are 23.715, 18.764, 23.725, 18.772 and 18.895 ms,
versus baseline medians 22.748, 22.477, 22.415, 22.483 and 22.867 ms. Thus the
pooled 1.188x is not a consistent per-process speedup: two rounds regress and
the candidate p90 is slightly higher (24.328 versus 24.043 ms). No slow rounds
are discarded. The cause of this process-to-process variation is not yet
isolated. MLP's control variation is not attributed to the convolution cache.
All 30 final outputs are byte-identical to the references; raw samples,
package/binary hashes, commands and device snapshots are retained in
`reports/workspace-cache-benchmark-224`.

The source/runtime change is accepted on this tested stack with that explicit
224x224 ResNet variability limitation. The local `environment.sh` selects
`build-workspace-cache-stream-id`; `WORKSPACE_BASELINE_BUILD_DIR` retains
`build-handle-cache`, and `BASELINE_BUILD_DIR` still retains the original
pre-muBLAS-cache implementation. No validated build directory is overwritten.

### Trace confirmation and remaining work

Every measured 64x64 ResNet trace contains 43 mallocs, zero synchronous frees,
43 async frees, 20 stream-ID lookups and zero stream waits; MobileNet contains
106/0/106 allocations/frees, 35 ID lookups and zero waits. Each method still
has one non-workspace event record. Compared with the event-cache prototype,
20/35 per-convolution record/wait pairs disappear. Tensor allocation/free
costs remain; this is not an allocation-free runtime.

`reports/workspace-cache-profile-stream-id` retains traces and verified final
outputs. The first analyzer rejected the additional warmup-growth and
thread-exit device synchronizations; that failed `analysis.json` is retained.
The corrected `analysis-timing-matched.json` uniquely aligns all five measured
intervals to runner timings within 0.05 ms and excludes extra warmup/teardown
events. It finds four/two extra pre-measurement syncs for ResNet/MobileNet and
two teardown syncs each. The correction changes trace attribution only, not
runtime code, timing data or acceptance tolerances.

Next work should isolate 224x224 ResNet process variability, expand standard
resolution/dtype performance coverage, and investigate the remaining tensor
allocator costs. Retention is per thread, with no public cache-trim API;
large thread counts, memory pressure, other GPUs/SDKs and device reset require
separate validation. The event fallback has correctness coverage on this
stack but no accepted general speedup claim.

## ResNet224 variability diagnostics (2026-10-27)

The two slower 224x224 ResNet candidate processes were not uniformly slow:
their first ten measured calls had medians 18.924/18.747 ms, followed by
slower intervals; the second process returned to an 18.867 ms median in its
last ten calls. This weakens a simple fixed-per-process explanation but does
not identify the cause. No original samples are replaced or removed.

A read-only, process-local interposer records muDNN workspace-query algorithm
and size, main-thread synchronization boundaries, thread CPU time, context
switches, CPU number and sampled CPU/GPU frequency. It does not change
allocation, algorithm, synchronization mode or system governors. Eight fresh
processes alternate baseline/candidate order over four rounds, with 10
warmups and 200 measured executions each. Another four processes repeat the
experiment with the whole child process restricted to CPU 2. All twelve
final outputs match the accepted reference bytes. These are instrumented
diagnostic observations, not replacement acceptance timings.

Findings:

- Every forward has the same 20 workspace queries, using muDNN forward
  algorithm enum 0 (`IMPLICIT_GEMM`), throughout warmup and measurement in all
  twelve processes and both variants. Requested sizes are also identical;
  the maximum is 7,043,584 bytes. This rules out a change in the observed
  public algorithm/request sequence, not differences inside the SDK's kernel
  selection or execution.
- Every measured synchronization-boundary GPU-frequency sample reports
  730 MHz. Most CPU policy-0 samples report 2.65 GHz. Boundary sampling cannot
  rule out frequency changes within a method execution.
- The launcher actually inherits CPU affinity `0-3`, although the host lists
  twelve online CPUs. The previous reports did not capture this constraint.
  No affinity is widened and no global power/frequency policy is changed.
- Slower samples include increased main-thread CPU time as well as off-CPU
  time. Thread CPU time includes runtime/driver activity and possible busy
  waiting; it is not CPU model computation and is not additive to GPU time.

| Diagnostic placement | Samples per variant | Baseline median / p90 (ms) | Cache median / p90 (ms) |
| --- | ---: | ---: | ---: |
| Inherited CPU 0-3 | 800 | 22.757 / 27.164 | 19.244 / 22.953 |
| Whole child pinned to CPU 2 | 400 | 24.537 / 25.886 | 20.830 / 21.666 |

Pinning the whole process narrows the observed tail but worsens both medians;
it is not adopted as a performance fix. It also constrains SDK worker threads,
so this experiment does not isolate main-thread migration alone. The precise
cause of the original slow intervals remains open. Next attribution should
compare GPU work and host/runtime waiting within fast and slow iterations,
not assume algorithm drift, clock throttling or scheduler migration from
these observations alone.

Evidence is retained under `artifacts/musa-20261026/reports/resnet-variance-observed`
and `reports/resnet-variance-cpu2`: summaries, raw synchronization/query TSVs,
timing samples, logs and byte-exact outputs. Their `analysis.json` files align
each measured interval to the runner's timing within 0.05 ms and verify the
full algorithm/workspace sequence. `resnet-variance/observe.cpp`,
`run_observations.py` and `analyze_observations.py` reproduce the diagnostic.

### Benchmark host context

`examples/musa/benchmark_aoti.py` now records `host_before` and `host_after`
for every subprocess, plus `last_process_context` even if the runner fails.
These snapshots contain launcher CPU affinity and available CPU-frequency
policy/devfreq attributes. CPU frequencies use sysfs kHz values; devfreq
frequencies use Hz. Reads occur outside the child process's timed method
intervals; no affinity, clocks or governors are modified.

Unavailable interfaces and individual read failures are explicitly recorded,
not substituted with zero and not treated as benchmark failures. In this
environment policy4/policy8 return `Device or resource busy` for some fields;
global governor-tuning directories are not misidentified as CPU policies.
Launcher affinity is not proof of the child's eventual affinity if a wrapper
or runtime changes it. Before/after snapshots are execution context, not
continuous clock measurements or a root-cause diagnosis.

The benchmark unit suite now contains eight methods, covering readable,
missing and partially unreadable host data and retaining context on runner
failure. All eight benchmark and seven package tests pass, as do nine CI
driver tests and the relocated deployment test. A final `python -S` ResNet18
BF16 224x224 smoke checks byte-exact output and the real snapshot fields in
`reports/resnet-variance-host-metadata-final`; its five samples are a tooling
smoke, not a new performance claim. Native runtime binaries are unchanged.

### Fast/slow trace attribution and allocator trial

Four fresh `msys profile --trace=musa,osrt` captures use ResNet18 BF16,
2x3x224x224, ten warmups and 100 measured calls per process, in
baseline/cache/cache/baseline order. The baseline here already has the BLAS
handle cache but not the convolution workspace cache. All four final outputs
are byte-identical to the frozen reference. Trace synchronization boundaries
match the runner's measured intervals within 0.009 ms.

All 400 measured calls have the same ordered sequence of 118 named kernels
and five device-copy activities. The baseline makes 63 `musaMalloc`, 20
`musaFree` and 43 `musaFreeAsync` calls per forward; the workspace-cache
candidate makes 43 allocations, zero synchronous frees and 43 async frees.
This rules out changes in the captured named-kernel sequence in these runs,
not every possible SDK dispatch detail or all causes of the earlier variance.

For each process, compare the means of its fastest and slowest quartiles
(25 calls each). Partition each method interval into disjoint portions:
the union of this process's captured GPU activities, time inside free APIs
not covered by that GPU union, and the remaining time. Do not add overlapping
API, driver synchronization, OS-runtime and GPU totals.

| Process | Slow minus fast method (ms) | GPU union delta (ms) | Free without own GPU activity delta (ms) | Remaining delta (ms) |
| --- | ---: | ---: | ---: | ---: |
| Baseline 0 | 5.400 | 0.126 | 3.974 | 1.300 |
| Workspace cache 1 | 6.723 | 0.310 | 5.079 | 1.334 |
| Workspace cache 2 | 3.830 | 0.155 | 2.869 | 0.805 |
| Baseline 3 | 5.073 | 0.152 | 3.830 | 1.091 |

About 74-76% of the within-process slow-quartile excess falls inside the free
APIs without overlapping captured GPU activity from that process. This points
to the allocation/free path for the next experiment; it does not establish
whether scheduling, driver work or waiting causes that time. Nor does absence
of this process's GPU activities prove that the whole GPU is idle. All raw
samples are retained; quartile selection is diagnostic, not a performance
acceptance result. Profiled timings do not replace the unprofiled A/B results.

The embedded profiler logs contain clock-calibration warnings at capture
startup/shutdown, unavailable GPU metadata, and permission-denied lookups for
other process names. No dropped/lost/overflow message was found. This is not
proof of complete trace coverage or perfect clock correlation. The analysis
retains those warnings and excludes other processes and the separate hardware
event stream from the GPU-activity totals.

An SDK-only allocation probe also confirms memory-pool support on this M1000
stack. Legacy allocation, default-pool async allocation, and private pools with
zero/64 MiB release thresholds all pass 20 pinned-readback byte checks over
1-8 MiB buffers. Each mode then records ten warmups and 200 allocation/memset/
free/synchronize cycles. Their single-process median cycle times are
0.516/0.492/0.488/0.497 ms respectively: retaining a private pool does not show
a clear benefit in this synchronization-heavy microtest. The retained pool
reports 32 MiB reserved and zero used before trimming, and zero reserved after
trimming. The device default pool's release threshold stays zero. A release
threshold is a retention policy, not a hard memory-cap guarantee.

Trace evidence is in `reports/resnet-variance-traces/{summary,analysis}.json`
and the four `.msys-rep` files under `artifacts/musa-20261026`.
`resnet-variance/capture_traces.py` and `analyze_traces.py` reproduce capture
and analysis. `resnet-variance/allocator_probe.cpp`, `probe-*.tsv` and
`probe-*.log` retain the standalone SDK probe, raw samples and pool counters.

A separate build changes only MUSA SlimTensor allocation from synchronous
`allocate` to `allocate_async` using the default pool. It retains the accepted
workspace cache and pinned-readback fix. This is a rejected diagnostic, not
an enabled backend option:

- The 19 matrix, four BatchNorm, four activation and four existing device tests
  each pass three times. The first queued multi-thread/multi-stream convolution
  test reports error 999 from the host-to-device copy and stops making progress;
  that failing child is terminated explicitly, and the native gate fails.
- The convolution helper uses a default-stream `cudaMemcpy` after allocating
  on a selected nonblocking stream. Switching allocation semantics therefore
  requires a stream-dependency audit, not deleting this failure or assuming
  pool support implies safe drop-in replacement. The precise cause of error
  999 is not established by this experiment.
- Eight fresh, unprofiled model processes still produce byte-exact final
  outputs: each model runs baseline/trial/trial/baseline with ten warmups and
  100 measured executions. Here baseline means the accepted workspace-cache
  build, not the older baseline in the trace table. Pooled minimum/median
  ResNet18 BF16 times worsen from 18.409/19.035 to 20.056/21.220 ms, and
  MobileNetV2 FP16 from 32.674/33.961 to 37.594/40.286 ms. Both use
  2x3x224x224 inputs. These narrow model checks do not override the failed
  native gate or establish correctness for every intermediate execution.

The default-pool allocation switch is reverted. Evidence remains in
`reports/storage-async-native`, `reports/storage-async-model-diagnostic`,
`build-storage-async` and `resnet-variance/async-storage-trial.patch`.
No production build/environment path is redirected to that trial.

`MusaDeviceTest.QueuedStorageReadbackBeforeRelease` adds explicit same-stream
storage-lifetime coverage: two nonblocking streams, 36 allocations per stream,
six changing byte lengths, queued fills and pinned-host readbacks, and storage
destruction before the final synchronization. It verifies every returned byte.
It passes three repetitions in the async trial, so the primitive same-stream
path works in that limited case despite the broader convolution failure.
This is not a cross-stream ownership or multi-device guarantee.

On the retained production implementation, all five device tests pass three
repetitions, and the other 40 native cases pass three fresh-process repetitions
per suite (45 cases total, through separate device and CTest invocations).
The updated device test executable is built separately against the retained
libraries, leaving the accepted runtime and runner binaries unchanged. Logs
and compile/link commands are retained under `resnet-variance/device-retained-*`
and `resnet-variance/native-retained-test.log`.

The next allocator experiment must preserve or explicitly establish allocation,
copy and destruction dependencies, and compare a private pool's retention
policy against the default pool on actual models. Neither simply selecting an
async API nor the standalone pool probe establishes a faster, safe allocator.

### Private-pool retention follow-up

A process-local interposer redirects the rejected async-allocation trial's
`musaMallocAsync` calls into a private device pool. It verifies the stream's
device, changes only that private pool's release threshold, and checks pool
counters and the unchanged default-pool threshold. Zero and 64 MiB retention
both pass the queued storage test three times (216 allocations per process).

The initial global-destructor cleanup ran after SDK shutdown and reported
error 3; an `atexit` variant crashed at shutdown. The final diagnostic invokes
cleanup after application `main` returns but before global teardown. It checks
device synchronization, zero live allocations, trimming to zero and pool
destruction. This Linux-only interposition technique is diagnostic tooling,
not a production lifecycle implementation; a future allocator needs explicit
ownership/teardown while the SDK is still alive.

Sixteen fresh model processes compare retained synchronous allocation,
default-pool async allocation, private zero retention and private 64 MiB
retention. Each model has two rounds in opposite variant order, ten warmups
and 50 measured calls per process, at 2x3x224x224. All final outputs match
the frozen references byte-for-byte. Pooled minimum/median times are:

| Allocation variant | ResNet18 BF16 (ms) | MobileNetV2 FP16 (ms) |
| --- | ---: | ---: |
| Retained synchronous allocation | 18.571 / 20.572 | 32.628 / 34.610 |
| Default-pool async trial | 20.256 / 22.248 | 37.862 / 44.126 |
| Private pool, zero retention | 20.206 / 22.285 | 38.315 / 44.122 |
| Private pool, 64 MiB retention | 20.080 / 23.229 | 37.791 / 43.349 |

The second round is slower across variants; these limited, variable diagnostic
runs are not a stable speedup estimate. Private-pool variants also include
interposer bookkeeping, unlike the two controls. Retention has not shown
sufficient benefit to adopt it. For each private-pool ResNet/MobileNet process,
2,580/6,360 allocations succeed. The retained pool holds 32/64 MiB at cleanup,
respectively; all pools report zero used bytes, zero reserved after trimming
and successful cleanup. Default-pool thresholds remain zero. No pool option
is enabled in the production backend.

`reports/storage-private-pool-diagnostic` contains all 16 process reports,
raw timings, byte-exact outputs and cleanup counters.
`resnet-variance/private_pool.cpp`, `pool_models.py` and `pool-device-v3-*.log`
retain the final interposer, driver and primitive tests. Earlier cleanup
failures remain in `pool-device-*.log` and `pool-device-v2-*.log`.

### MUSA selected-stream blocking copies

The stream audit finds a separate correctness gap: SlimTensor allocation and
operator dispatch use the internally selected stream, but blocking storage
copies previously used only `CallerStreamGuard` or the default stream. With
a nonblocking internal stream, default-stream copies do not establish the
required ordering with work on that selected stream.

For MUSA, `DeviceTraits::memcpy` now selects an explicit caller stream first,
then the GPU device's existing internally selected stream, and otherwise
retains the default-stream fallback. It enqueues the copy and synchronizes
the selected stream to preserve blocking semantics. Looking up the internal
stream does not create one. Explicit default-stream selections remain distinct
from an absent selection. The pinned-readback workaround and synchronous
SlimTensor allocation remain enabled; CUDA/ROCm selection behavior is unchanged.

`test_musa_storage` adds three cases covering H2D, D2D, D2H and scalar readback,
both native and legacy device tags, nested caller-stream priority (including
an explicit default stream), and selected/default/unset states. Test-local
SDK forwarding wrappers record the actual copy API and stream, excluding
nested SDK implementation calls. All three cases fail against the old copy
selection and pass three repetitions after the fix. This proves routing and
returned values, not a reproduction of every possible stale-read race.

The new target is included in the native CI driver. A fresh build passes all
48 native cases three times without skips, and the nine CI-driver unit tests
pass. A final rebuild repeats the same gate in
`reports/storage-stream-native-final`; source change `2edc386d27` contains
the routing fix and CI coverage. Evidence is in `reports/storage-stream-native`,
`resnet-variance/storage-before-final-test.log`, `storage-after-test.log` and
`storage-stream-ci-unit.log`. This fix does not itself make arbitrary
cross-stream tensor sharing safe or resolve the rejected async allocator's
raw-copy convolution-test failure.

The frozen-model gate now passes all 66 fresh runner processes. This comprises
54 MLP FP32 and pretrained ResNet18/MobileNetV2 FP32/FP16/BF16 processes using
the saved dynamic-shape cases, plus 12 ResNet18 BF16/MobileNetV2 FP16 processes
at 224x224 with batches 1/2/3. Each case runs one or five forwards; each final
output is byte-identical to the previously accepted runner reference and also
passes the saved MUSA-eager numerical/prediction checks. This does not check
every intermediate output of the five-forward executions or dataset accuracy.

The delivery gate also passes all 16 runner regressions, seven package tests,
eight benchmark tests and the relocated deployment test, without skips. The
runner gate includes fresh export/compile checks for convolution-bias,
depthwise-convolution and Hardsigmoid rounding plus native/legacy metadata.
The deployment test re-exports MLP FP32 and SimpleCNN BF16, relocates their
packages and runs batches 1/2/3 without Torch/Python runtime dependencies.

Evidence is retained in `reports/storage-stream-models/summary.json`,
`reports/storage-stream-delivery/summary.json` and
`reports/storage-stream-acceptance.json`. The final audit verifies committed
source fingerprints, unchanged runtime binaries, all 48 native cases repeated
three times, and all 66 frozen output comparisons.
`resnet-variance/validate_storage_streams.py`, `verify_storage_delivery.py` and
`audit_storage_streams.py` retain the validation drivers.

The local `artifacts/musa-20261026/environment.sh` now selects
`build-storage-streams` for subsequent work and preserves
`build-workspace-cache-stream-id` as `STORAGE_BASELINE_BUILD_DIR`. Historical
builds, packages and reference outputs are not overwritten. This accepts the
selected-stream correctness fix, not an allocator performance improvement;
the rejected async/default-pool/private-pool trials remain disabled.

### Compiler-managed pool views (2026-10-27)

An allocation-origin trace of the accepted ResNet18 BF16 runner at
`2x3x224x224` records 43 successful `musaMalloc` calls in each of three
measured forwards after two warmups: 20 convolution outputs, 20 batch-norm
outputs and three direct wrapper allocations. The final output is byte-equal
to the frozen reference. This is an instrumented call-origin diagnostic, not
a latency benchmark. Its driver, stacks, synchronization alignment and hashes
are retained in `artifacts/musa-20261026/memory-planning/`, including
`resnet-allocation-analysis.json`.

The installed Inductor planner groups `AllocateLine` buffers but explicitly
does not support reuse of `ExternKernelAlloc` outputs. Consequently enabling
`memory_planning` alone cannot pool those 40 extern outputs. Reducing them
requires compiler-visible output buffers and matching out-parameter shims,
not just changing the allocator or its retention threshold.

There was also a concrete compatibility gap: exporting MLP FP32 with
`TORCHINDUCTOR_MEMORY_PLANNING=1 TORCHINDUCTOR_MEMORY_POOL=combined` emitted
`aoti_torch__alloc_from_pool`, which the SlimTensor runtime did not provide.
The development runner fell through to the libtorch implementation and
aborted when given a SlimTensor handle. The original PTE/PTD and failure log
are retained under `memory-planning/mlp-before*`.

The shim now returns a shared-storage SlimTensor view without a device
allocation or copy. Byte offsets are converted to elements of the requested
dtype; the pool must have zero storage offset. Bounds, alignment, supported
dtypes, negative sizes/strides and size arithmetic are validated before view
construction. Scalar, empty, noncontiguous and zero-stride views are covered,
as is retaining storage after the pool handle is destroyed. Non-owning pool
storage retains its existing external-owner contract.

The CNN gate also exposed an existing reinterpret ABI mismatch: its offset
is relative to the input tensor, whereas SlimTensor's `as_strided` accepts
an absolute storage offset. ResNet18's pooled average-pool output has a
nonzero offset, and flattening it with a zero increment incorrectly read
the beginning of the pool. A targeted native regression reproduced the
mismatch before the fix. The shim now adds the input storage offset and
rejects negative totals and signed overflow; direct `as_strided` semantics
are unchanged.

`test_musa_storage` includes five pool-view tests in addition to its three
copy-stream tests. The deployment regression exports both MLP FP32 and
SimpleCNN BF16 with memory planning off and on (`combined`), checks that
planned PTEs really reference the pool ABI, and exercises relocated packages
at batches 1/2/3 without Torch/Python runtime libraries. Compiler memory
planning remains opt-in; this change does not enable the rejected async
allocator or private-pool trials, nor establish a CNN speedup.

The isolated `build-memory-planning` build passes all 53 native cases three
times without skips. The original failing MLP artifact runs for five
forwards with maximum error `1.4901161193847656e-08` against its saved MUSA
eager output. The original failing pooled ResNet18 artifact now produces
byte-identical frozen-reference outputs in separate one- and five-forward
processes, without re-exporting the artifact.

Fresh pooled pretrained ResNet18 BF16 and MobileNetV2 FP16 exports at 224x224
pass batches 1/2/3 in two fresh processes per shape: all 12 outputs are
byte-identical to the previously accepted unpooled runner outputs and pass
the eager-tolerance and prediction checks. Relocated deployment passes all
four export configurations (two models, planning off/on) with no test skips,
errors or failures. The model/native/deployment results are in
`memory-planning/accepted/validation.json`; failed intermediate diagnostics
are retained separately and are not acceptance evidence.

Tracing the same pooled ResNet18 artifact after the offset fix records 41
allocations in each of three measured forwards: 20 convolution outputs,
20 batch-norm outputs and one wrapper pool. Its final output remains
byte-identical. The reduction is **43 to 41 allocations**, not elimination
of all intermediate allocations, and is not evidence of a latency win.
Compiler-visible batch-norm and convolution out buffers remain the next
substantial reuse opportunity.

The final regression gate additionally passes all 16 runner tests without
skips and all 66 frozen-model processes from the selected-stream acceptance
matrix, including FP32/FP16/BF16 CNNs and one/five-forward invocations. Every
checked final output is byte-identical to its frozen runner reference and
passes the saved eager/prediction gates; intermediate forwards are not
individually checked. Reports are
`memory-planning/accepted/final-verification.json` and
`memory-planning/accepted/frozen-models/summary.json`. Nine CI-driver tests
and 15 package/benchmark tests also pass.

The local `environment.sh` now selects `build-memory-planning` and preserves
`build-storage-streams` as `POOL_BASELINE_BUILD_DIR`. This promotes the pool
ABI and relative-view correctness fixes, not a default compiler-planning
setting or allocator-policy change. CUDA/ROCm hardware, other SDK versions
and a latency benefit have not been validated by this gate.

### Compiler-owned BatchNorm outputs (2026-10-27)

MUSA BatchNorm inference now lowers through `ExternKernelOut` and the new
`aoti_torch_musa_batch_norm_out` shim. Inductor allocates the output and can
reuse it through ordinary buffer reuse or compiler-managed pools. The old
returning shim remains exported and uses the same muDNN implementation for
previously compiled models. There is no allocator-policy change.

The out shim borrows its destination: it requires matching shape, dtype and
device, contiguous layout and sufficient storage, and rejects overlapping
input/parameter ranges. Disjoint views into the same pool are allowed,
including strided inputs whose physical extent ends before the output.
Noncontiguous inputs/parameters still require their existing contiguous
copies, so this is not a promise of zero allocations for every input layout.
The Python `executorch_musa.batch_norm.out` overload preserves the out
signature for dispatch/codegen; its eager implementation computes then
copies a result and is not the allocation-saving native runtime path.

The `build-batchnorm-out` prototype passes all 58 native cases three times,
eight BatchNorm export/dispatch tests, all 16 runner tests, and relocated
deployment without Torch/Python runtime libraries. There are no skips,
errors or failures in those gates. Its nine native BatchNorm cases include
the original four allocating-shim tests plus out-buffer coverage for ranks
2/3/4/5, FP32/FP16/BF16 with FP32 parameters, nonblocking streams, nonzero
pool offsets, shared strided pools, byte canaries, repeated use, invalid
destinations and overlapping parameters.

Fresh SimpleCNN BF16, pretrained ResNet18 BF16 and MobileNetV2 FP16 exports
pass with planning disabled and with `memory_planning=True,
memory_pool="combined"`. Each export runs batches 1/2/3 in two fresh
processes per shape: 36 model executions in total. All pass the MUSA eager
checks; the 24 ResNet18/MobileNetV2 outputs are additionally byte-identical
to their accepted frozen-reference outputs. Exported PTEs are checked for
the new out symbol and absence of the old returning BatchNorm symbol.

At batch 2 and 224x224, allocation-origin captures after two warmups record
the following counts in each of three measured forwards:

| Model | Planning disabled | Combined pool |
| --- | --- | --- |
| ResNet18 BF16 | 20 convolution + 8 wrapper = 28 | 20 convolution + 1 pool = 21 |
| MobileNetV2 FP16 | 52 convolution + 19 wrapper = 71 | 52 convolution + 1 pool = 53 |

ResNet18's corresponding pre-change counts were 43 and 41. No BatchNorm
output allocations remain in these warmed contiguous-input captures, and
the final captured outputs remain byte-identical. These are allocation
counts, not latency speedup measurements. The remaining convolution
outputs still allocate inside the returning shim.

Evidence is under `artifacts/musa-20261026/batchnorm-out/acceptance-v2/`;
`summary.json` records the passed native, export, model, trace, deployment
and runner gates. Earlier build/test-harness failures are retained outside
that accepted run. The subsequent `legacy-models/summary.json` passes all
66 frozen-model runs. `spatial-models/summary.json` passes both pretrained
models at 2x3x64x64, 1x3x48x54, 3x3x80x80 and 2x3x79x71, with two fresh
processes per shape: all 16 outputs remain byte-identical to the frozen
references.

The completed `benchmark/summary.json` records four balanced-order rounds
of the four variants below, at batch 2 and 224x224 on the same M1000. Each
process uses 10 warmups and 50 measured forwards, giving 200 samples per
model/variant. All 32 processes finish successfully with byte-identical
outputs and unchanged artifact/binary hashes. No allocation interposer is
loaded during timing.

| Model | Variant | Minimum ms | Median ms |
| --- | --- | --- | --- |
| ResNet18 BF16 | Previous, planning disabled | 27.082 | 29.970 |
| ResNet18 BF16 | Previous, combined pool | 16.352 | 29.279 |
| ResNet18 BF16 | BatchNorm out, planning disabled | 13.780 | 16.790 |
| ResNet18 BF16 | BatchNorm out, combined pool | 14.226 | 16.241 |
| MobileNetV2 FP16 | Previous, planning disabled | 34.275 | 51.716 |
| MobileNetV2 FP16 | Previous, combined pool | 33.576 | 51.696 |
| MobileNetV2 FP16 | BatchNorm out, planning disabled | 27.492 | 46.314 |
| MobileNetV2 FP16 | BatchNorm out, combined pool | 24.237 | 26.905 |

Combined-pool median speedups are 1.80x and 1.92x respectively against the
previous combined-pool artifacts. These are local forward-latency results,
not end-to-end application throughput or a guarantee across environments.
The reports retain per-round measurements and host snapshots; the
MobileNetV2 unpooled variant in particular varies across rounds. Host
affinity/governors were not changed. The separate convolution-out build
starts only after these measurements finish.

At the BatchNorm-only checkpoint, the default environment was not promoted. New
out-lowered models require a runner and shim library exporting the matching
out ABI, not the previous runtime without it.

### Compiler-owned convolution outputs (2026-10-27)

MUSA convolution now lowers through `ExternKernelOut` and
`aoti_torch_musa_convolution_out`. The compiler owns the contiguous output,
so ordinary buffer reuse and combined-pool planning can include convolution
results. The native out shim and the retained returning shim call the same
muDNN implementation, preserving fused forward bias, transposed bias,
workspace-cache behavior and selected-stream routing. This does not change
the synchronous SlimTensor allocator policy or the pinned-readback path.

The borrowed destination must match the inferred output shape, dtype and
device and have sufficient contiguous storage. Byte-range checks reject
overlap with input, weight or bias before cloning noncontiguous sources or
launching muDNN. Disjoint shared-pool views are accepted; overlapping
bounding spans of strided sources are conservatively rejected. The out
interface covers forward/transposed 1D/2D/3D, groups and optional bias;
the lowering also preserves unbatched-input handling.

The isolated `build-convolution-out` runner and shims build successfully.
All 16 native convolution tests pass three consecutive times, preserving
the 13 returning-shim tests and adding three out-path tests. The positive
matrix exercises 72 combinations of rank, FP32/FP16/BF16, forward versus
transposed, dense versus depthwise groups and bias presence. It uses a
nonblocking selected stream, a nonzero pooled output offset, three writes
per destination, byte-exact comparison with the returning shim and canary
checks. Separate tests cover strided shared-pool inputs/weights with an
analytic reference and reject malformed destinations or input/parameter
overlap without modifying output. The full native suite now passes 61
cases three times, with no skips/errors/failures. Four Python
fallback/context tests also pass. The targeted AOTI test passes all seven
pooled export cases: forward/transposed 1D/2D/3D plus unbatched Conv2d,
checking the new symbol and pool ABI and absence of the old returning
convolution symbol.

Evidence lives under `artifacts/musa-20261026/convolution-out/`.
The initial broad compile run exposed a test-harness error when inspecting
file-backed weight buffers, not a compilation failure. The corrected
symbol check reads both in-memory and file-backed data; the targeted rerun
passes. The following `acceptance-v2` run passes all 16 selected compile
methods and both SimpleCNN BF16 and pretrained ResNet18 BF16 with planning
disabled/enabled, batches 1/2/3 and two fresh processes per shape. All 12
ResNet18 outputs are byte-identical to the frozen reference.

Allocation captures at batch 2 and 224x224 show the following warmed
forward counts. Each capture records the same count in all three measured
forwards, with a byte-identical final output:

| Model | Planning disabled, BN out to convolution out | Combined pool, BN out to convolution out |
| --- | --- | --- |
| ResNet18 BF16 | 28 to 16 | 21 to 1 |
| MobileNetV2 FP16 | 71 to 41 | 53 to 1 |

No convolution or BatchNorm output allocations remain inside their native
shims. ResNet18's single compiler-side pool allocation is 8,030,208 bytes.
These counts exclude input preparation, warmup and teardown; they are not
peak-memory or latency measurements.

The initial pooled trace hit the diagnostic's timing-alignment tolerance,
not a model failure. The follow-up interposer records synchronization
callsites: the runner's begin/pre-execute/post-execute triplet repeats once
per iteration, while backend/TLS teardown has distinct call stacks.
MobileNetV2 also grows its workspace during the first warmup, causing an
additional synchronization inside the convolution shim. Allocation windows
therefore select only the three direct runner callsites, require their
exact repeating pattern, and retain all allocations inside each execution
window regardless of their call stack. Non-runner synchronizations are
recorded separately rather than assumed to number exactly two. Four
diagnostic tests cover timing jitter, nested synchronizations, malformed
runner sequences and failed runtime calls. This replaces the earlier
50-microsecond timing-match heuristic without dropping operator work.

The failed captures are retained. `acceptance-v4/summary.json` passes the
core gates: 61 native cases three times, 16 selected compile methods, all
six model exports (36 recorded runner outputs), relocated standalone
deployment and all 16 runner tests, with no skips/errors/failures. All 24
pretrained ResNet18/MobileNetV2 outputs are byte-identical to the original
references. Five previously completed exports and unchanged native/compile
results are reused after checking source identity, artifact/binary hashes,
native XML and test logs; they are not reported as freshly rerun. The new
MobileNetV2 pooled export, all four allocation captures, deployment and
runner tests execute in this final core run. A separate audit rechecks all
six model manifests, recorded outputs and current runner/shim hashes.

`test_musa_convolution_out.py` adds three end-to-end pooled decoder tests
for transposed 1D/2D/3D convolution. All pass on FP32/FP16/BF16 with dynamic
batches 1/2/3: nine exports and 27 runner processes, five forwards per
process, with final outputs checked against same-dtype MUSA eager. They
also check the out/pool ABI symbols and absence of the returning symbol.
Evidence is in `decoder-models/tests.log` and `decoder-models/summary.json`.

The subsequent acceptance is complete. All 114 frozen-artifact runs across
17 manifests, including BatchNorm-out and older pooled exports, are
byte-identical. Both pretrained models pass pooled dynamic spatial tests
at 2x3x64x64, 1x3x48x54, 3x3x80x80 and 2x3x79x71, with two processes per
shape: all 16 outputs are byte-identical to the frozen references.

The uninstrumented benchmark compares the accepted BatchNorm-out build
against convolution out within the same four balanced-order rounds. Each
process uses 10 warmups and 50 measured forwards at batch 2 and 224x224,
giving 200 samples per model/variant. All 32 processes finish successfully
with byte-identical outputs and unchanged binary/artifact hashes.

| Model | Build / planning | Minimum ms | Median ms |
| --- | --- | --- | --- |
| ResNet18 BF16 | BatchNorm out / disabled | 19.485 | 31.554 |
| ResNet18 BF16 | BatchNorm out / combined | 17.811 | 28.895 |
| ResNet18 BF16 | Convolution out / disabled | 17.896 | 29.251 |
| ResNet18 BF16 | Convolution out / combined | 11.969 | 14.589 |
| MobileNetV2 FP16 | BatchNorm out / disabled | 35.683 | 54.532 |
| MobileNetV2 FP16 | BatchNorm out / combined | 33.267 | 51.402 |
| MobileNetV2 FP16 | Convolution out / disabled | 37.752 | 46.942 |
| MobileNetV2 FP16 | Convolution out / combined | 14.183 | 19.986 |

The combined-pool variants are faster in every paired round, with aggregate
median speedups of 1.98x and 2.57x respectively. These are local
host-synchronized method timings, not application throughput guarantees.
There is appreciable variation: ResNet18's pooled per-round medians range
from 13.917 to 23.014 ms; without pooling, its small aggregate improvement
is not consistent in every round.

All before/after host snapshots in this benchmark record launcher CPU
affinity 0-3, whereas the earlier BatchNorm benchmark recorded 0-11. The
baseline binaries and model/input paths are verified identical between
the two campaigns, but host conditions differ. Do not compare their
absolute timings directly or multiply their speedup ratios. No affinity
or governor settings are changed by these tests; snapshots also retain
unreadable frequency-policy entries as errors rather than inventing values.

`final-verification.json` passes an independent recheck of native XML,
cached validation logs, all six model manifests, pretrained outputs, 114
compatibility outputs, 16 dynamic-spatial outputs, binary hashes, and all
32 benchmark processes and aggregate samples. Earlier failed diagnostic
runs remain preserved outside the accepted checkpoint.

The local `environment.sh` now selects `build-convolution-out` and retains
the previous builds as `MEMORY_PLANNING_BASELINE_BUILD_DIR` and
`BATCHNORM_BASELINE_BUILD_DIR`. `promotion-smoke/summary.json` confirms
dependency resolution against the new shim libraries and byte-identical
ResNet18/MobileNetV2 outputs after five forwards each using the promoted
runner. Compiler memory planning remains disabled by default; for the
validated pooled path, opt in with `TORCHINDUCTOR_MEMORY_PLANNING=1` and
`TORCHINDUCTOR_MEMORY_POOL=combined` during export. Fresh convolution-out
exports require this build or a newer matching runtime; the previous
BatchNorm-only build lacks the new symbol.

## Compiler-owned hard activation outputs (2026-10-27)

Hardswish and Hardsigmoid now lower through `ExternKernelOut` using the
existing `aten.hardswish.out` and `aten.hardsigmoid.out` schemas. The native
`aoti_torch_musa_hardswish_out` and `aoti_torch_musa_hardsigmoid_out` shims
borrow compiler-owned output storage, including disjoint views in a shared
memory pool. The original returning symbols remain available. Both paths
share the same muDNN invocation and Hardsigmoid parameters; this change does
not replace the previously validated rounding behavior.

The out path validates output shape, dtype, device, contiguous layout,
storage offset and capacity before launching work. Input storage spans are
bounded using sizes and strides; overlapping source/destination byte ranges
are rejected before any input cloning or output writes. Noncontiguous input
still uses a contiguous clone, so eliminating output allocations does not
imply allocation-free execution for arbitrary strided inputs.

The isolated candidate runtime is `build-activation-out` under
`/home/qianfan/文档/ChatGPT/ETSupport/artifacts/musa-20261026`.
Its evidence is in the sibling `activation-out` directory:

- `compile-tests.log`: six unittest methods passed without skips, including
  the four fallback/context checks, dynamic Hardswish compilation in three
  dtypes, and pooled Hardswish/Hardsigmoid compilation in three dtypes.
  The pooled binaries contain both new out symbols and the pool allocator,
  and do not reference the old returning activation symbols.
- `native-*.xml`: 63 native tests passed in each of three rounds. The six
  activation methods include 60 out-path combinations across two operators,
  three dtypes, contiguous/strided inputs and five shapes (including scalar
  and empty inputs). Each destination is overwritten three times on a
  selected nonblocking stream, compared byte-for-byte with the allocating
  path, and checked for pointer stability and untouched pool canaries.
  Invalid/undefined handles, shape/dtype/layout/capacity mismatches and
  overlapping or out-of-bounds views are rejected without modifying storage.
- `activation-runner.log`: two new runner test methods passed. Contiguous
  and transposed activation chains cover FP32/FP16/BF16, planning off/on,
  and dynamic batches 1/2/3: 12 exports, 36 fresh runner processes, five
  forwards per process and 72 final outputs exactly matching same-dtype
  MUSA eager. Dynamic-loader logs confirm no PyTorch/Python runtime libraries
  are loaded. The tests retain both intermediate and final outputs, checking
  that pool reuse does not overwrite a still-live activation result.
- `runner-regression.log`: all 16 existing runner/reference tests passed
  without skips, including Hardsigmoid rounding, convolution precision,
  native/legacy device metadata and reference snapshot integrity.
- `summary.json` reports all native and runner gates passed with unchanged
  binary hashes. `final-verification.json` ties that evidence to the
  implementation commit and records source/log/binary hashes. The initial
  native-test build failed on a two-element `IntArrayRef` initializer; its
  log is retained. Explicit array references fixed the test construction,
  and `rebuild.log` contains the successful build used by these gates.

These gates validate activation output reuse, not a measured model speedup.
The default environment still points to the accepted convolution-out build;
it has not been promoted to this candidate. Fresh activation-out exports
require the matching new runtime. Real-model allocation/latency measurements
and broader model compatibility checks remain separate acceptance work.

## MobileNetV3 acceptance entry points (2026-10-27)

`aoti_model_acceptance.py` and the CI `models` suite now accept
`mobilenet_v3_small` and `mobilenet_v3_large`, with local torchvision
checkpoints, configurable spatial size, bounded dynamic batches/spatial
dimensions, and strict top-1 agreement in addition to logit tolerances.
The existing `deeplabv3_mobilenet_v3_large` exporter is also exposed through
the CI model selector and retains its per-pixel label gate. The default
four-model CI matrix is unchanged. The `runner` suite now also includes
`test_musa_activation_out` and `test_musa_convolution_out` rather than leaving
these pooled-output regressions as manual-only tests.

Ten CI orchestration tests and two prediction-gate test methods passed
(`activation-out/model-ci-final.log` and `prediction-tests.log`). The latter
checks both new classifier names, including prediction mismatches that would
otherwise pass the numerical tolerance.

Both pretrained classifier correctness matrices passed at 224x224:
dynamic batches 1/2/3, two fresh processes per shape, legacy/out lowering,
and planning disabled/enabled.

| Model | Dtype | Exports | Recorded outputs | New outputs byte-identical to legacy | Largest eager logit error |
| --- | --- | ---: | ---: | ---: | ---: |
| MobileNetV3-Small | FP16 | 4 | 24 | 12 | 0.009765625 |
| MobileNetV3-Large | BF16 | 4 | 24 | 12 | 0.0078125 |

All outputs pass the unchanged MUSA eager tolerance and top-1 checks. Input
and eager-reference bytes match between paired exports; within each model,
all four PTD weight payloads have the same SHA-256 digest.
`activation-out/classifier-model-verification.json` independently rechecks
tensor manifests, model payload hashes, predictions and byte equality.
The earlier Small-only audit remains in `small-model-verification.json`.
This is execution parity on generated inputs, not ImageNet accuracy validation.

Evidence lives in `activation-out/models` under the artifact root above.
For this controlled A/B comparison, `export_variant.py` restores only the
three activation-lowering functions directly from commit `fd5f60fa18` in
isolated legacy-export processes. All other compiler code, model weights,
seeds and inputs are shared. `lowering.json` records this provenance; it is
not a claim that the entire old compiler checkout was replayed. Legacy
exports run on `build-convolution-out`; out exports run on
`build-activation-out`. Checkpoint URLs and SHA-256 digests are recorded in
`activation-out/checkpoints.json`.

DeepLab BF16 comparisons, standalone legacy artifact replay, allocation
tracing and balanced latency measurements remain separate gates. Classifier
correctness alone does not promote the candidate runtime or establish a
model speedup.

## Dynamic-spatial pool code generation (2026-10-27)

The first activation-out model campaign exceeded its 1,200-second deadline
while exporting and validating pooled BF16 DeepLabV3-MobileNetV3-Large with
dynamic batch and spatial dimensions. Its failed `models/summary.json` is
preserved; the existence of PTE/PTD and raw outputs does not make that
unfinished acceptance run successful.

Live Python stack samples identified two compiler costs, not a GPU hang:
`TemporalSplit.get_symbolic_size` constructed nested generic SymPy maxima,
invoking pairwise connectivity/factorization, and the C++ pool-offset printer
expanded those expressions again through `SizeVarAllocator.simplify`.
Skipping only the latter was insufficient; that diagnostic probe was
explicitly stopped after sampling the remaining first hotspot.

The MUSA compilation context now uses PyTorch's optimized symbolic `Max`
for temporal pool sizes, preserving the existing method cache and invalidation
contract. Pool offsets retain factored expressions when printed as C++, with
the graph's symbol replacements still applied. Other expressions and non-MUSA
graphs keep their original paths. Both patches are scoped to the existing
compilation context and restored on exceptions. This does not change pool
lifetimes, alignment, dynamic bounds, native allocation APIs, or the default
memory-planning setting.

Evidence under `activation-out` in the artifact root above:

- `pool-codegen-tests-final.log`: ten compiler-hook/fallback methods passed.
  Coverage includes CPU/CUDA isolation, symbol replacements, exception
  restoration, cache invalidation, nested maximum values, and compiling the
  factored C++ offsets for 108 batch/height/width combinations.
- `pool-codegen-v2/summary.json`: fresh pooled DeepLab legacy and out exports
  both passed. Each covers `(2,3,64,64)`, `(1,3,48,54)`, `(3,3,80,80)` and
  `(2,3,79,71)`, with two fresh runner processes per shape. All 16 outputs
  pass the unchanged MUSA eager tolerance and pixel-label checks and are
  byte-identical to their respective unplanned baseline. Both weight
  payloads have the same digest; compiler source and runtime binary hashes
  are recorded.
- The two successful export-and-validation jobs took approximately 270 and
  423 seconds. These are end-to-end acceptance durations, not isolated
  compilation timings or inference latency measurements.
- `pool-codegen-regressions/compile.log`: 13 methods passed, adding pooled
  activation, batch-normalization and convolution compilation to the hook
  tests. `pool-codegen-regressions/runner.log` also records five passing
  activation/convolution runner methods without skips (63 fresh processes).
- `pool-codegen-verification.json`: the complete regression gate passed
  with unchanged compiler source and runtime binary hashes. Four additional
  standalone processes replayed the original pre-fix DeepLab PTE/PTD, with
  five forwards per shape and byte-identical final outputs versus the new
  factored-offset export. All eight raw outputs left by the original job
  also match. This validates expression equivalence for those artifacts;
  the original producer remains recorded as `failed_timeout_preserved`.

The earlier classifier matrices used the pre-fix pool code generator.
Fresh pooled classifier exports are required before claiming a complete
common-compiler model matrix. No runtime promotion or inference speedup is
claimed by this compiler fix.
