# Packed-kernel counters, warmup and block grouping

This follows the [dequantization instruction investigation](QWEN35_GGUF_INSTRUCTION_FOLLOWUP.md)
and the [host wait-policy experiment](QWEN35_GGUF_WAIT_POLICY.md). It keeps the
qualified Qwen3.5-4B Q4_K_M local-first runtime and FP32 arithmetic unchanged.
All new kernels and harnesses are isolated artifacts. No variant is promoted,
and stable 10 decode tokens/s is not established.

## Fresh counters on the merged projection

Moore Perf Compute 1.3.0 profiles the real merged gate/up matrix, 18432 by
2560. A mangled-name filter selects the standalone candidate kernel rather
than the anonymous-namespace reference. Each of the three configurations
captures one launch after skipping 90 matching launches, with three
application replays and the SpeedOfLight, LaunchStats and
MemoryWorkloadAnalysis sections. The grid has 1152 blocks of 128 threads.

| Configuration | Compute(MP) throughput | L1 throughput | Memory pipes busy | L1 hit rate | Registers/thread | Profiled duration, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local-first control | 72.29% | 39.37% | 24.33% | 49.79% | 107 | 1.57 |
| Vector dequantization contraction | 94.28% | 51.22% | 29.32% | 47.10% | 107 | 1.19 |
| Contraction plus vector metadata | 70.27% | 38.32% | 22.72% | 49.78% | 108 | 1.59 |

All report three waves per MP and an MP frequency of 0.73 GHz. That frequency
is the profiler's metric, not an independent simultaneous sysfs measurement.
No clock-control command is issued. The GPU remains shared. Compute(MP) is not
FLOP utilization, and the separately replayed durations do not establish a
causal 32% speedup. In particular, ordinary alternating microbenchmarks below
do not reproduce that apparent benefit. Static instruction-count reduction,
counter percentages and elapsed time are not interchangeable.

The control and contraction variants have the same reported register count;
the metadata variant adds one register. These observations do not diagnose
occupancy or a specific stall cause. MP22's available metric list does not
include the MP31 DDR byte counters, so these reports do not establish measured
DRAM bandwidth or a full-model memory-bound/compute-bound classification.

## Bounded active warmup

A fresh native harness adds at least three seconds of alternating reference
and candidate launches before event timing. It does not request or lock a
frequency. It preserves three complete byte comparisons outside timing,
then records all 100 alternating timing pairs and a read-only GPU frequency
sample after each pair. Two initial timing pairs remain excluded as warmups.
No profiler, preload or launch blocking is present.

Six merged-projection runs use configuration order `0,2,3,3,2,0`. All **18
complete outputs** remain byte-identical to the qualified runtime:

| Configuration | First reference/candidate median-time ratio | Reverse ratio |
| --- | ---: | ---: |
| Local-first control | 1.003984 | 1.004668 |
| Vector contraction | 0.984557 | 0.983165 |
| Contraction plus vector metadata | 1.010079 | 1.007529 |

A ratio above one favors the candidate. Of 600 sampled frequencies, 530 are
1.4 GHz and 70 are 730 MHz; all lower samples occur in the first control run.
The remaining five runs sample 1.4 GHz throughout. Thus the contraction-only
regression persists with high sampled frequency, while the combined variant
has only a small improvement on this one matrix. This is not a model-level
gain or justification to multiply the historical peak throughput by a ratio.
Three seconds of active warmup is not a guaranteed frequency condition.

## Block-grouping experiment

A separate probe tests 128, 256 and 512 threads per block with the existing
local-first arithmetic. Each eight-lane subgroup still computes one row;
only the number of rows assigned to a block and the launch grid change.
The FP32 accumulation/reduction order, packed weights and Split-K partition
boundaries are unchanged. This is not the older vector-kernel launch sweep:
it tests the current packed-quad local-first implementation.

Eight real Q4/Q5 matrices, three block sizes and opposite matrix/variant
orders produce 48 processes. Each uses the bounded warmup and 100 alternating
timing pairs above. All **144 complete outputs** match byte for byte. The
128-thread control's complete device fatbin is identical to the previous
standalone dequantization control, not a claim of whole-library identity to
the production shim. All variants report zero local-memory/spill bytes and
the same compiler temporary-register counts for corresponding kernels.

| Matrix | 128-thread control ratios | 256-thread ratios | 512-thread ratios |
| --- | ---: | ---: | ---: |
| Attention gate | 1.0013 / 0.9874 | 0.9801 / 1.0411 | 0.9840 / 0.9297 |
| Q5 QKV | 0.9988 / 0.9790 | 0.9800 / 0.9865 | 0.9799 / 0.9870 |
| FFN gate | 0.9874 / 1.0050 | 0.9984 / 0.9836 | 0.9876 / 0.9836 |
| SSM output | 0.9866 / 0.9966 | 0.9995 / 0.9843 | 0.9942 / 0.9890 |
| Attention K | 1.1428 / 1.0054 | 1.0260 / 1.1421 | 0.7849 / 0.9533 |
| Attention Q | 1.0178 / 1.0083 | 0.9829 / 0.9919 | 1.0069 / 0.9922 |
| Q4 FFN down | 0.9992 / 1.0037 | 1.0043 / 1.0046 | 0.9869 / 0.9998 |
| Merged gate/up | 1.0018 / 1.0005 | 1.0065 / 1.0066 | 0.9950 / 0.9951 |

The larger blocks provide no broad improvement. The 256-thread merged case
improves about 0.65% against the reference in both orders, but several other
shapes regress or change direction. Even the unchanged standalone control
has substantial variation on the small attention-K case. These measurements
do not justify selecting favorable samples or introducing a general block-size
change. Long-K timings include the separate partial-reduction launch on both
sides; they are not timings of the exported caller-owned-parts path alone.

All 4800 timing pairs are retained: 4347 frequency samples read 1.4 GHz and
453 read 730 MHz. Neither low-frequency samples nor regressions are discarded.
Sampling is outside the timed events and cannot exclude intra-pair frequency
changes. These are microbenchmarks, not full-model HF/logit qualification,
tail-shape coverage or normal model-throughput acceptance.

## Evidence and decision

The independent audit verifies protected hashes, replay selection, complete
micro-output checks, every timing pair, frequency counts, both matrix orders,
compiler resources and the unchanged control device image. It does not
reinterpret counter timings as ordinary performance.

Under ETSupport `artifacts/gguf-20261029`:

- `mcu-packed-dequant-v1`: three raw counter reports, replay logs and launches.
- `packed-hot-probe-v1`: bounded-warmup native harness, build and six runs.
- `packed-block-probe-v1`: isolated kernels, native harness, builds and all
  48 runs with unfiltered samples.
- `audit_packed_followup_counters_v1.py`,
  `packed-counter-followup-decision-v1.json`: independent combined audit.

The current evidence does not justify replacing the qualified runtime.
Future promotion still requires a reproducible representative gain followed
by the complete model numerical gate and ordinary end-to-end timing; a
favorable counter capture or one matrix is insufficient.

The subsequent [lane-width and integer-dot investigation](QWEN35_GGUF_LANE_WIDTH_AND_DOT.md)
tests the current packet kernel's per-lane work distribution and checks the
SDK's signed/unsigned integer primitives before considering a new algorithm.
