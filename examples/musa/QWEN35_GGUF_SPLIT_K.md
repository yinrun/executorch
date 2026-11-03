# Long-K Split-K GEMV experiment

## Scope and decision

`EXECUTORCH_MUSA_GGUF_SPLIT_K` is a default-OFF experimental MUSA build
option. It partitions a single-vector Q4 or losslessly predecoded Q6
projection into four K ranges, then adds the four FP32 partial outputs.
It does not change weights, quantization, public shim arguments or the
existing PTE/PTD. The accepted deployment is not overwritten.

The model is Qwen3.5-4B **Q4_K_M**, with FP32 activations, recurrent state
and accumulation, not BF16 weights. Existing grouped-Q6-head, merged-FFN
and valid-prefix-attention optimizations remain in the measured deployment.
No TF32, MTP, requantization, vocabulary pruning or validation removal is
used. The 10 tokens/s target remains unmet.

This follows the rejected layout and row-reuse experiments in
[QWEN35_GGUF_LAYOUT_FOLLOWUP.md](QWEN35_GGUF_LAYOUT_FOLLOWUP.md).

## Dispatch and implementation

The experimental path requires a single vector, at least 128 output rows,
at least 4096 input columns, and 16-byte-aligned input and weight pointers.
Columns must fit a signed int and rows must fit the four-part scratch
indexing limit. The existing validator continues to enforce complete
256-element quantization blocks, storage spans, device/type consistency
and non-overlap. Unaligned views, smaller projections, Q5, original Q6
and batched inputs retain their original paths.

Q4 uses width 16; predecoded Q6 uses width 4 with two row tiles. Both reuse
the existing decoding kernels with a compile-time partition count; their
default one-part instantiations retain the original reduction order.
For partition `p`, the block range is
`[floor(blocks*p/4), floor(blocks*(p+1)/4))`, including uneven partitions.
A second kernel adds partials in order 0, 1, 2, 3. The output head has K=2560
and is not selected.

Scratch is a per-call `SlimTensor` of shape `[4, rows]`, allocated on the
input device and used on the caller's stream. There is no static pointer
cache, persistent model-specific buffer, new host readback or explicit
synchronization. The allocator owns asynchronous lifetime tracking.

Enable on an existing MUSA configuration with
`-DEXECUTORCH_MUSA_GGUF_SPLIT_K=ON`. Measurements also enable
`EXECUTORCH_MUSA_GGUF_MAX_ILP`, `EXECUTORCH_MUSA_GGUF_QUAD_ARITHMETIC` and
`EXECUTORCH_MUSA_GGUF_GROUPED_Q6_HEAD`; they are not measurements of the
Split-K flag alone on an otherwise default build.

## Correctness

Splitting K changes FP32 summation order. Byte identity to the original
unpartitioned model is not claimed. Both the prototype and the integrated
source runtime independently pass the unchanged frozen-HF gate:

| Fixture | Complete logits | Maximum error versus matched HF step |
| --- | ---: | ---: |
| Text | 232 | 0.000034094 |
| Single image | 856 | 0.000728607 |
| Multiple images | 1210 | 0.000734329 |

All 248320 logits at every step must be finite and within
`0.001 + 0.001 * abs(HF)`, with exact argmax at every prefill and decode
step. The same tolerance is also checked against the old native baseline.
All original generated token IDs/EOS, 1149 byte-exact reset pairs and
18 deliberately corrupted negative controls pass. No tolerance or argmax
exception is introduced. Frozen HF trajectories and multimodal fixtures
are reused; this is not a fresh raw-image-frontend or independent HF rerun.
The integrated-source outputs also match the prototype byte for byte at
all 2298 steps; neither is byte-identical to the unpartitioned baseline.

The 14-test native GGUF suite passes with the CMake option both ON and OFF.
Selected-path exact comparisons use an independent GPU reference: four
separate two-vector generic projections, followed by ordered FP32 addition.
Two vectors deliberately prevent that reference from entering Split-K.
Unselected paths and the OFF build retain their original byte-exact tests.
Additional tests cover K=4096/4352/9216, row tails, small-row and short-K
fallbacks, offset views, batches, full-output double-accumulated references,
nondefault streams, queued scratch lifetimes and weight updates.

## Measurements and profiling

Native microbenchmarks sweep Q4/Q5 widths and Q6 row tiles with 1/2/4/8
partitions. All 192 complete-output comparisons pass tolerance. The selected
four-part prototype gives about 1.22x speedup for Q4 FFN-down, 1.10x for Q4
SSM-output projection, and 1.11x for Q6 FFN-down.
Short-K projections do not show a broad gain and are excluded. These are
microbenchmark results, not model throughput.

The prototype's ordinary ABBA and BAAB tests each use one warmup and two
measured runs per process, giving eight measurements per variant. The
original workload has 28 prompt tokens, 89 generated IDs including EOS,
and 88 decode steps. Full output readback/checks remain enabled; profiler,
launch blocking and logits dumping are disabled. The GPU is shared with
the desktop and clocks are not locked.

The prototype's combined median is **7.3321 baseline versus 7.4446 candidate
tokens/s** (+1.53%); candidate minimum is **7.4000**. Separate order gains
are +1.20% and +1.79%. Integrated-source throughput is measured separately;
prototype measurements must not be silently attributed to its rebuilt
library. That separate integrated-source ABBA+BAAB series gives:

| Runtime | Decode minimum | Decode median | Serial prefill median |
| --- | ---: | ---: | ---: |
| Accepted baseline | 7.2471 | 7.3353 | 7.3410 |
| Integrated Split-K | 7.3956 | 7.4383 | 7.3961 |

All speeds are tokens/s. Combined decode improvement is **1.40%**; the two
individual order gains are **1.56%** and **1.12%**. This is a small repeatable
gain in these shared-desktop series, not an exclusive-device statistical
guarantee or a change to the deployment default. The integrated candidate
library SHA-256 is
`58faa6faa1f1c14a031c48fb7ed1fd4641410c870dcdf62b3cc8ae8b3ddc435d`.

The prototype global trace covers loading, serial prefill, complete decode
and validation. Every measured decode step has 48 Q4 and 16 Q6 Split-K
projections, each followed by a reduction: **876 kernel records versus
812 before**, and **67 `musaEventRecord` calls versus 3 before**. Stream
synchronizations remain **12 per step**. No additional allocation API count
appears in the steady-state trace. Async kernel labels can share intervals;
their durations must not be summed or used as reliable individual-kernel
costs. Profiled throughput is not ordinary throughput, and historical
cross-session trace timings do not establish a speedup.

## Caller-owned partials and fused epilogue PoC

A subsequent artifact-only PoC removes that extra reduction/scratch cost.
An experimental `gguf_parts` export operation returns `[4, ..., rows]`.
Its native shim writes the same four partitions directly into the caller's
output, owned by AOTI's combined memory pool. Explicit ordered additions
`((part0 + part1) + part2) + part3` remain in the exported graph, allowing
Inductor to fuse them with the following residual/RMSNorm work. There is no
per-call native scratch tensor or native reduction launch. This new export
operation and native entry point are **not integrated into the production
backend or public loader**; they live only in the local probe artifacts.

The re-export replaces 64 projections while retaining one delegate, the same
64 mutated state buffers, 32 merged-FFN calls and eight valid-prefix-attention
calls. Source GGUF hashes and loading metadata match the original export,
apart from two later-added GDN options explicitly recorded as false. Neither
GDN experiment is enabled. The first export fails because the temporary
header's repository-relative include is absent from the AOTI wrapper include
path; export v2 uses an explicit local header path and succeeds.

The new PTE/PTD and isolated native runtime pass all **2298 complete logit
comparisons**, original IDs/EOS, **1149 reset pairs** and **18 negative
controls**. Every logit also matches the preceding Split-K source runtime
**byte for byte**. HF tolerances and exact per-step argmax remain unchanged.
An independent audit rehashes outputs, sources, runtime and protected files;
it does not infer acceptance merely from the successful six-step eager probe.

Ordinary ABBA+BAAB timing uses the same workload and rules as above. Here the
baseline is the **previous qualified opt-in Split-K source runtime**, not
the original accepted grouped-head deployment:

| Runtime | Decode minimum | Decode median | Serial prefill median |
| --- | ---: | ---: | ---: |
| Previous Split-K | 7.3150 | 7.4241 | 7.4126 |
| Fused-epilogue PoC | 7.5337 | 7.5555 | 7.6150 |

All speeds are tokens/s. Combined decode improvement is **1.77%**, with
separate order gains **1.87%** and **1.90%**. Eight measured samples per
variant are collected on the shared desktop. The ordinary timing series
finishes before profiling starts. Do not multiply gains from separate
sessions and present them as a fresh comparison against the original
accepted deployment.

The new full-model trace records **812 kernels and three `musaEventRecord`
calls per measured decode step**, compared with 876 and 67 in the earlier
isolated Split-K prototype trace. It contains 48 Q4 and 16 Q6 partial
projection launches and no separate partial-reduction kernel. Stream
synchronizations remain 12 and `musaMallocAsync` calls remain eight per step.
This confirms the intended launch/lifetime change, not a per-kernel timing
attribution from overlapping async labels.

The PoC runtime SHA-256 is
`b1f7ff396d314c28f596080aad2e3592f7d793f3cef1b03702b2a0e1a484806f`.
The accepted deployment and default flags remain unchanged. Next is a
production-quality opt-in implementation with boundary/ownership tests and
fresh source-runtime qualification; the artifact's benchmark must not be
attributed to an unbuilt production implementation. Further large-matrix
optimization is still needed to reach 10 tokens/s.

## Local evidence

ETSupport evidence under `artifacts/gguf-20261029`:

- `split-k-quad-probe-v1`, `split-k-q6-probe-v1`: source, resource reports,
  native microbenchmarks and complete-output comparisons.
- `build-split-k-v1`, `split-k-validation-v1`, `split-k-benchmark-v1/v2`:
  isolated prototype, full accuracy gate and ordinary timing.
- `global-split-k-profile-v1`: runtime maps, SQLite trace and interval-union
  analysis; this trace belongs to the prototype, not the integrated build.
- `build-split-k-source-cmake-v2`: fresh CMake build and ON/OFF native tests;
  early inherited-library and old-summation-contract failures are retained.
- `build-split-k-source-v2`: integrated host/kernel objects linked into the
  protected accepted host runtime for controlled whole-model comparison.
- `split-k-source-validation-v2`, `split-k-source-benchmark-v2/v3`:
  integrated-source numerical and ordinary throughput evidence.
- `audit_split_k_source_v2.py`, `split-k-source-decision-v2.json`: source,
  build, original artifact hashes, numerical and timing audit.
- `split-k-epilogue-probe-v1`, `build_split_k_epilogue_v1.py`,
  `build-split-k-epilogue-v1`: temporary export operator, native partial-output
  shim/kernel, isolated build and protected-artifact checks.
- `export_split_k_epilogue_v2.py`, `split-k-epilogue-export-v2`:
  successful re-export after the retained v1 header-path failure.
- `split-k-epilogue-validation-v1`, `split-k-epilogue-benchmark-v1/v2`,
  `global-split-k-epilogue-profile-v1`: complete accuracy, ordinary timing
  and the separate global trace.
- `audit_split_k_epilogue_v1.py`, `split-k-epilogue-decision-v1.json`:
  independent rehashing and model/export/accuracy/performance/profile audit.
