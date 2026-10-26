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
| `deployment` | Package/benchmark validation and relocated MLP FP32/SimpleCNN BF16 execution without Torch/Python runtime libraries, including synchronized timing and fresh-process repeatability |
| `all` | All of the above, in order |

The five native CTest executables contain 41 GoogleTest cases. Each executable
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
code remains on the per-call-handle path; the separate SDK-only readback
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

The next gate is to refresh the broader pretrained/dtype acceptance matrix
with these protections, then repeat handle-reuse A/B validation and unprofiled
benchmarks. The handle cache remains disabled. Neither the diagnostic speedup
nor old `/tmp` evidence is an accepted performance result for the new code.
