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
and precedes dispatch. This is not yet a production build option.

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
4096-token trajectory, concurrent-service behavior or source-option coverage.

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
causal 13% or 3.2% improvement. A same-process alternating comparison and wider
qualification remain necessary before treating this as stable deployment
performance. No new default or export is installed from this favorable series.

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
