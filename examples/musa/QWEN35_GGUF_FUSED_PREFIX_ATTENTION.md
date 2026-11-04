# Short-prefix FP32 attention fusion

This investigation returns to the global trace after the rejected
[group-tiled integer MMA experiments](QWEN35_GGUF_GROUP_TILED_MMA.md).
The model remains Qwen3.5-4B Q4_K_M with FP32 activations, recurrent state and
accumulation. The qualified runtime and original Split-K PTE/PTD are not
overwritten. No TF32, MTP, reduced-precision inputs or relaxed gate is used.

## Global motivation

A read-only follow-up of the [wait-policy trace](QWEN35_GGUF_WAIT_POLICY.md)
checks all 176 measured decode windows, 88 each for auto and blocking waits.
Every step has eight async allocations, eight async frees and twelve stream
synchronizations. Summed `musaMallocAsync` host API duration has median
1.1007/1.1755 ms per step; the union of device and pinned-host allocation/free
API intervals has median 1.4633/1.7358 ms. This is not removable critical-path
time: host intervals can overlap device work, and API duration includes waits.
The trace does not contain source-stack attribution for individual calls.

Source inspection identifies a scratch allocation in each valid-prefix
attention call. The FP32 path materializes QK, scaled scores and probabilities,
using separate muDNN operations for QK, scaling, accurate softmax and PV.
Fusion can avoid these intermediate buffers and launches. It does not remove
the checked position/mask readback or any of its synchronization barriers.
The dominant packed Linear work remains a separate bottleneck; attention
microbenchmark ratios must not be applied to whole-model throughput.

## Isolated kernels and microbenchmarks

The probe computes 16 query heads, four KV heads and head dimension 256.
All attention math is FP32. It stages queries and scores in shared memory,
uses max-subtracted softmax and reads only the valid KV prefix. No future
cache entry is touched. It accepts a caller-supplied stream.

The first 128-thread implementation is slower at longer prefixes. Moving
normalization out of the PV loop does not fix that. Eight/sixteen-component
PV tiles improve it, but do not produce a broad win. Four independent prefix
partitions plus a second merge kernel help at length 115, but need caller-owned
scratch and remain slower at long prefixes. The best short-prefix variant
instead uses one 512-thread block per query head, eight PV components per
thread and a 16-lane reduction. It needs no global scratch or second kernel.
Compiler reports show 28 temporary registers, 3156 shared-memory bytes and
zero local/spill bytes; these do not establish occupancy or stall causes.

Seven variants each run ten capacity/length cases in both orders. Each process
compares four complete outputs on changing synthetic inputs, including zero
queries and higher-magnitude queries, with poisoned future KV entries, output
canaries and a nondefault stream. It then uses three seconds of alternating
warmup and retains 100 alternating event-timing pairs. The reference uses the
same FP32 muDNN operation sequence and disabled TF32 as the shim, but reuses
descriptors and preallocated scratch. Prefix validation/readback is excluded
from both sides. This is not a full shim or model benchmark.

A separate eight-process identity control runs muDNN on both sides. Together,
the probes compare 592 complete 4096-float outputs, check 4736 canaries and
444 invalid calls. Maximum absolute error is 2.0862e-7, with no local tolerance,
finite or argmax failures; identity-control outputs are byte-exact. Synthetic
attention argmax checks are not model-token acceptance. The initial host build
fails on an ambiguous `SetNdInfo` overload; an explicit int64 dimension-count
cast fixes it, and the failed build is retained.

| Capacity / valid length | 512-thread forward ratio | Reverse ratio |
| --- | ---: | ---: |
| 512 / 28 | 1.4197 | 1.5178 |
| 512 / 63 | 1.3695 | 1.3859 |
| 512 / 115 | 1.1863 | 1.1914 |
| 512 / 128 | 1.1339 | 1.1438 |
| 4096 / 115 | 1.1714 | 1.1677 |
| 512 / 256 | 0.9664 | 0.9498 |
| 512 / 512 | 0.7405 | 0.7740 |

Ratios are reference latency divided by candidate latency. Identity controls
range from 0.99989 to 1.00647. All frequency samples and regressions remain;
the selected variant samples 730 MHz 1800 times and 1.4 GHz 200 times.
Clocks and other desktop applications are not changed. These results support
a bounded short-prefix experiment, not replacing long-prefix attention.

## Isolated runtime and model qualification

The prototype dispatches only grouped FP32 attention with valid length at
most 128 and a 16-byte-aligned value pointer. BF16, longer prefixes and
unaligned values retain the original muDNN path. Existing dtype, geometry,
device, span, alias, position and complete causal-mask validation is unchanged
and precedes dispatch. This first experiment uses an isolated prototype; the
subsequent default-off source option is described below.

The first integration attempt detects that today's host source does not
reproduce the historical object; it stops before linking. The corrected
attempt retrieves the exact historical source from commit `0ff9cfc820`,
checks its recorded SHA-256 and reproduces the original host object byte for
byte before introducing the fused branch. Only that object and the new kernel
are changed in the isolated link. Other historical objects, runner, extension,
weights, export and qualified shim remain protected.

The original text trajectory passes 232 complete 248320-float logit checks,
116 byte-identical reset pairs and six negative controls. All original
generated IDs/EOS, finite checks, exact per-step argmax and the frozen HF
`0.001 + 0.001 * abs(reference)` gate pass. Maximum absolute error versus HF
is 0.0000262260, maximum normalized error 0.0187488, and maximum additional
error versus the qualified local-first baseline is 0.0000143051.
HF is not rerun. The historical native attention suite passes eleven tests;
one second-device test retains its unavailable-device skip.

Single-image and multi-image trajectories subsequently pass the same gate,
without any tolerance change. Together with text, all 2298 complete logits,
1149 byte-identical reset pairs and eighteen negative controls pass.

| Case | Complete outputs | Maximum HF absolute error | Maximum normalized HF error |
| --- | ---: | ---: | ---: |
| Text | 232 | 0.0000262260 | 0.018749 |
| Single image | 856 | 0.0006752014 | 0.334109 |
| Multiple images | 1210 | 0.0008490086 | 0.573804 |

Normalized error must remain at most one for every element. All original
IDs/EOS and exact argmax checks pass; outputs are not byte-identical to the
previous runtime. Prepared image features and frozen HF references are reused,
not rerun through the raw-image frontend. This does not establish a full
4096-token trajectory or concurrent-service behavior. Source-option coverage
is qualified separately below.

## Ordinary whole-model timing

The unchanged runner/export executes the original 28-token prompt and 88
decode steps to EOS. Eight processes use ABBA followed by BAAB order, with
one complete warmup and two measured generations per process. All original
IDs/EOS, full output readback and validation are retained. There is no profiler,
launch blocking, logit dumping, concurrent investigation GPU work or compilation
during timing. The device remains shared and unlocked.

| Series | Baseline median | Candidate median | Candidate minimum | Candidate maximum |
| --- | ---: | ---: | ---: | ---: |
| ABBA | 10.2867 | 10.4156 | 10.1922 | 10.7066 |
| BAAB | 9.1731 | 10.3648 | 10.1506 | 10.6056 |
| Combined, eight samples per variant | 10.0925 | 10.4156 | 10.1506 | 10.7066 |

Units are decode tokens/s. Serial-prefill medians are 10.1758 baseline and
10.5494 candidate tokens/s; this is the original serial prefill, not batched
prefill. The candidate's eight measured samples all exceed 10, meeting the
minimum-based timing check in this particular series. Its 10.7066 peak exceeds
the earlier recorded 10.5812, but neither peak is a stability guarantee.

All slow baseline samples remain: the second BAAB baseline process records
8.2780 and 8.3027 tokens/s. ABBA's median ratio is 1.01253, whereas BAAB's is
1.12991 and the pooled ratio 1.03202. These inconsistent ratios do not prove a
causal 13% or 3.2% improvement. The subsequent same-process comparison below
does not confirm a stable minimum of ten. No new default or export is installed
from this favorable series.

## Same-process alternating comparison

A benchmark-only runner switches between the original muDNN path and the
qualified fused kernel between complete synchronized generations. Both modes
share the same loaded model, buffers and process. Counters verify eight calls
per executed step in the selected branch and zero calls in the other branch.
These hooks are not part of the source option or production runner.

Before timing, four complete text generations compare 464 full logit outputs
byte for byte with the independently HF-qualified runtimes, including 232
reset pairs. This instrumented harness does not rerun image trajectories.
Two processes then use opposite sequences, `0101101001` and `1010010110`;
the first two complete generations warm up each mode once. All eight subsequent
generations per process are retained, giving eight measured samples per mode.

| Mode | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Original muDNN | 8.6849 | 10.0307 | 10.5892 |
| Fused prefix | 8.4903 | 10.1675 | 10.7272 |

Units are decode tokens/s. Six of eight adjacent pairs improve; the paired
median ratio is 1.01157 and geometric mean ratio 1.00640. The first measured
generation in each process is slow, but belongs to opposite modes. This is an
observation, not a diagnosis of clocks, allocator behavior or desktop activity.
All low samples remain. The instrumented 10.7272 peak is not an ordinary-runner
record, and the 8.4903 minimum fails the stability target.

## Default-off source option

The MUSA backend now supports `EXECUTORCH_MUSA_FUSED_PREFIX_ATTENTION`, default
`OFF`. Add `-DEXECUTORCH_MUSA_FUSED_PREFIX_ATTENTION=ON` to an existing qualified
MUSA build configuration (`EXECUTORCH_BUILD_MUSA=ON`) to enable it. The separate
`.mu` object uses the configured MUSA architecture and `-ffp-contract=off`,
without applying GGUF-specific MAX_ILP flags. CUDA and ROCm paths are unchanged.

Dispatch occurs after the existing checks and current-stream/device selection,
but before scratch allocation. Only grouped FP32, valid prefix lengths 1--128
and 16-byte-aligned values use the fused kernel. BF16, longer prefixes and
misaligned values retain muDNN. Both checked attention entry points share this
dispatch. No runtime experiment switch, counters or tolerance changes enter
the production source.

A fresh CMake tree is configured in default-OFF, ON, OFF-again and ON-again
order. The attention and GGUF suites pass in all four states: 31 native tests
pass and one unavailable second-device test is skipped. OFF/ON objects and
libraries reproduce byte for byte on revisiting each configuration. The added
boundary test covers 48 combinations of capacities 129/4096, prefix lengths
1/7/31/63/115/127/128/129, aligned/misaligned values, output canaries, poisoned
future cache entries and unchanged complete KV storage. It retains the existing
independent double-reference unit bound of `3e-5`.

Removing formatter-added namespace comments subsequently reproduces both test
binaries, the attention objects and the CMake shared library byte for byte;
CTest is rerun. The current source objects are also linked into an isolated
copy of the historically qualified runtime, leaving all other objects intact.
That isolated source link passes all three full-model cases: 2298 complete
logits are byte-identical to the qualified fused prototype, with the unchanged
HF gate, original IDs/EOS, 1149 reset pairs and eighteen negative controls.
This is not a full-model qualification of every freshly rebuilt CMake object.

The source-linked runtime receives another ordinary ABBA+BAAB series using the
unchanged runner, one full warmup and two measured generations per process:

| Mode | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Original local-first runtime | 8.0835 | 8.4762 | 10.5297 |
| Source-linked fused prefix | 8.6756 | 10.4421 | 10.7215 |

The candidate's serial-prefill median is 9.3282 tokens/s. All samples and
original trajectories are retained. The pooled ratio of 1.23193 is not evidence
of a causal 23% improvement: baseline low samples dominate this series and the
same-process result is much smaller. The candidate again fails the minimum-ten
check. No concurrent agent GPU work, compilation or heavy audit runs during
timing. A user-requested historical-results summary causes lightweight reads
and a response during the run, so the intended wait-only protocol is not
strictly achieved; no causal effect is inferred. The option remains off by
default and stable ten tokens/s remains unproven.

## Fresh global capture

A separate nonblocking capture covers loading, complete warmup, serial prefill,
full measured decode and output checks. Library maps and original trajectories
are verified. Every one of the 232 execution windows contains eight fused
attention kernels. Kernel labels decrease from the recorded baseline's 812 to
788 per step, and async allocations/frees decrease from eight each to zero.
Stream synchronizations remain twelve: the checked host prefix is not removed.

The measured decode host-window median is 137.020 ms and model Compute-track
union median 123.347 ms. These profiled values are not normal throughput.
Async kernel labels still share intervals, so neither their summed durations
nor host waits should be added to engine times. Compute intervals are clipped
to execution-window boundaries. Counts demonstrate actual fast-path execution
and removal of intermediate allocations; they do not establish a causal
whole-model speed ratio or exclusive-device performance.

## Evidence

Under ETSupport `artifacts/gguf-20261029`:

- `analyze_attention_api_windows_v1.py`, `attention-api-windows-v1.json`:
  allocation/control API analysis of the existing hashed global trace.
- `fused-prefix-attention-probe-v1/v2`, `fused-prefix-attention-probe-v3-w0/w8/w16`,
  `fused-prefix-attention-probe-v4-split`, `fused-prefix-attention-probe-v5-t256/t512`:
  retained failure, sources, compiler logs, outputs and unfiltered timings.
- `fused-prefix-attention-control-v1`: unchanged-muDNN timing control.
- `audit_group_tile_and_fused_attention_v1.py`,
  `group-tile-and-fused-attention-decision-v1.json`: numerical, timing and hash audit.
- `fused-prefix-attention-integration-v1/v2`, `build-fused-prefix-attention-v1/v2`:
  original-object mismatch, exact-source control and isolated candidate runtime.
- `validate_fused_prefix_attention_v1.py`, `fused-prefix-attention-text-validation-v1`,
  `fused-prefix-attention-image-validation-v1`, `test-fused-prefix-attention-native-v1.log`,
  `test-fused-prefix-attention-native-loader-v1.log`: complete model outputs,
  native tests and loader confirmation of the candidate shim.
- `benchmark_fused_prefix_attention_v1.py`, `fused-prefix-attention-benchmark-v1`:
  complete ordinary ABBA/BAAB series, protected hashes and per-process maps.
- `profile_global_fused_prefix_attention_v1.py`,
  `analyze_global_fused_prefix_attention_v1.py`, `global-fused-prefix-attention-profile-v1`:
  raw trace, original trajectory checks, per-window counts and engine intervals.
- `audit_fused_prefix_runtime_v2.py`, `fused-prefix-runtime-decision-v1.json`:
  independent rechecks of every raw logit, references, reset equality, ordinary
  rates, protected artifacts and all 232 raw-trace kernel/allocation counts.
  The first audit script has a retained syntax failure; the corrected script
  changes no model data, numerical bound or measurement.
- `alternating-attention-probe-v1`, `build-alternating-attention-v1`,
  `alternating-attention-validation-v1`, `alternating-attention-benchmark-v1`,
  `alternating-attention-decision-v1.json`: benchmark-only switching harness,
  branch counts, full text qualification and independently checked paired rates.
- `build_fused_prefix_source_v1.py`, `build-fused-prefix-source-cmake-v1`,
  `build-fused-prefix-source-v1`, `fused-prefix-source-clean-build-v1.json`:
  four-configuration native tests, isolated source link and clean-source binary
  reproduction. Pre-cleanup source snapshots retain the initial build hashes.
- `validate_fused_prefix_source_v1.py`, `fused-prefix-source-validation-v1`,
  `benchmark_fused_prefix_source_v1.py`, `fused-prefix-source-benchmark-v1`:
  source-linked full-model precision and unfiltered ordinary timing.
- `audit_fused_prefix_source_v1.py`, `fused-prefix-source-decision-v1.json`:
  independent current-source/build hash checks, full raw-logit and reference
  rechecks, original trajectories, process maps and recomputed timing summaries.
