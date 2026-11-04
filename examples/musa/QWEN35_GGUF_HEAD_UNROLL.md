# Q6 output-head loop scheduling follow-up

This follows the default-off [short-prefix attention option](QWEN35_GGUF_FUSED_PREFIX_ATTENTION.md).
The model remains Qwen3.5-4B Q4_K_M with FP32 activations, recurrent state and
accumulation. No requantization, TF32, MTP, vocabulary pruning or tolerance
change is introduced. All head variants are isolated experiments; no head
source option or deployment default changes.

## Current source-runtime attribution

A new blocking `msys --trace=musa,osrt` capture runs the source-linked fused
attention runtime with the unchanged Split-K export and original runner.
It covers 28 serial-prefill steps and eight decode steps, preserving the first
nine generated IDs. Library maps, trace PID, model/runner/runtime hashes and
all 36 execution windows are checked. There is no full-generation warmup in
this diagnostic. The earlier complete nonblocking capture remains separate.

Every window contains 788 distinct, nonoverlapping kernel-label intervals,
including eight fused attention kernels and one grouped Q6 output-head kernel.
Async allocations/frees remain zero and stream synchronizations remain twelve.
Blocking supports bounded per-kernel attribution, not normal throughput or an
estimate of removable host-wait time.

Across eight decode steps, median summed kernel duration is 125.259 ms;
packed Linear accounts for 106.431 ms, approximately 85%. The grouped Q6 head
alone has median 15.330 ms, approximately 12% of kernel time. Non-Linear
kernels account for 18.836 ms. Category medians need not add to the total
median. Host-window median is 226.996 ms and compute-track union median
125.356 ms; waits and GPU intervals overlap and must not be added.

| Kernel family | Calls per step | Median aggregate time, ms |
| --- | ---: | ---: |
| Packet Q4, single part | 75 | 43.6300 |
| Packet Q4, four parts | 48 | 17.6780 |
| Grouped Q6 output head | 1 | 15.3301 |
| Packet Q5 | 24 | 14.5064 |
| Prepared Q6, four parts | 16 | 12.8177 |
| GDN state-update pointwise kernel `_9` | 24 | 3.7411 |
| GDN reduction kernels `_10` / `_7` | 24 each | 2.4367 / 2.3702 |
| Fused prefix attention | 8 | 0.4472 |

This prioritizes projections rather than another broad attention rewrite.
The previous wider-GDN fusion already reduced launches without demonstrating a
reliable model gain; these numbers do not invalidate that negative result.
The shared, unlocked desktop GPU and blocking profiler affect absolute times.

## Head-only unroll experiment

The qualified head uses four row tiles per subgroup, 16 lanes per row and ten
256-column blocks. Only the compiler unroll directive on the outer ten-block
loop changes. The packed representation, per-block FP32 FMA, scale application,
row mapping and final reduction remain unchanged. This is not the older unroll
sweep on a Q4/Q5 or pre-grouped Q6 kernel.

Five settings run in forward/reverse order, with all builds completed before
timing. Each of ten processes uses the real 248320-by-2560 output matrix and
three changing synthetic inputs. All **30 complete outputs are byte-identical**
to the qualified source-attention runtime, finite, with **240 output canaries**
preserved. Compiler-default and explicit-one complete device images are identical.

Each process has at least three seconds of alternating active warmup, then
100 recorded reference/candidate event pairs after two initial pairs. All
1000 pairs and frequency samples remain: 955 read 1.4 GHz and 45 read 730 MHz.
No clock is requested or locked; samples cannot exclude changes within a pair.
Preparation/repacking and comparisons are outside timed events. No other
investigation GPU job or compilation overlaps timing.

| Outer-loop setting | Forward ratio | Reverse ratio | Temp registers | Local/spill bytes |
| --- | ---: | ---: | ---: | ---: |
| Compiler default | 1.000307 | 0.999552 | 110 | 0 |
| Explicit one | 1.000138 | 0.996805 | 110 | 0 |
| Two | 1.028984 | 1.038336 | 190 | 0 |
| Five | 0.294992 | 0.292550 | 254 | 488 |
| Ten | 0.239258 | 0.240572 | 254 | 924 |

Ratios are reference median latency divided by candidate median latency. The
two-block candidate measures 8.9069/8.7808 ms versus references at
9.1651/9.1174 ms. Larger factors regress severely and report spills. Resource
reports do not measure occupancy or stall causes. A 3% head microbenchmark
gain is not a 3% whole-model improvement and cannot be applied directly to
historical decode throughput.

## Qualification boundary and evidence

The two-block candidate proceeds to full-model qualification rather than being
promoted from microbenchmarks. An isolated link replaces only the GGUF device object in
the qualified source-attention runtime. The unchanged-control fatbin reproduces
the protected original device image, and the selected object reproduces the
measured two-block image exactly. The initial build helper mistakenly treats
the SONAME flag as an input filename and stops before compilation; the corrected
helper excludes flags. Both scripts/logs remain; no runtime is overwritten.

Text, single-image and multi-image trajectories subsequently pass **2298 complete
248320-float logit comparisons**, **1149 byte-identical reset pairs**, original
IDs/EOS, exact per-step argmax and the unchanged frozen-HF elementwise gate
`0.001 + 0.001 * abs(reference)`. All outputs are byte-identical to the
qualified source-attention runtime. Eighteen negative controls are rejected.
Independent auditing repeats the raw-buffer, HF-reference, process-map and
original-trajectory checks; it also verifies that the isolated source differs
from the repository kernel only by the bounded unroll directive.

| Case | Complete outputs | Maximum absolute error vs frozen HF |
| --- | ---: | ---: |
| Text | 232 | 0.0000262260 |
| Single image | 856 | 0.0006752014 |
| Multiple images | 1210 | 0.0008490086 |

Prepared image features and frozen HF references are reused, not rerun through
the raw-image frontend. This does not qualify arbitrary shapes, long-context
service or a new production build option. The subsequent timing below does not
establish stable ten tokens/s or justify changing the deployed default.

## Full-generation timing follow-up

A benchmark-only combined library retains both head kernel bodies. A
thread-local selector changes modes only between complete synchronized
generations. Branch counters require exactly one selected head launch per
executed step and zero launches of the other mode. Both modes use the same
loaded model, buffers, export and fused prefix attention; the protected ordinary
runner and deployment runtime are not overwritten.

Before timing, four complete text generations in order `0110` compare 464 full
logit buffers byte for byte against the corresponding independently qualified
runtimes, with 232 within-mode reset pairs and original IDs/EOS. The switching
harness does not rerun image trajectories. Two processes then warm each mode
once and measure eight complete generations each, using opposite sequences
`0101101001` and `1010010110`. All eight samples per mode and eight adjacent
pairs remain; switches and counter collection are outside generation timing.

| Same-process mode | Minimum tok/s | Median tok/s | Maximum tok/s |
| --- | ---: | ---: | ---: |
| Qualified original head | 10.1046 | 10.1433 | 10.2550 |
| Two-block unroll | 10.1378 | 10.1997 | 10.6953 |

Five of eight pairs improve. The paired median ratio is 1.00604 and geometric
mean ratio 1.00930; the latter includes one 1.04773 pair. No high or low sample
is excluded. This supports at most a small observed gain in this series, not
the full head microbenchmark ratio applied to the whole model. Both modes
exceed ten in this instrumented series, but ordinary runs below contradict a
broader minimum-ten claim.

The unchanged ordinary runner subsequently executes eight ABBA+BAAB processes,
each with one full warmup and two measured original generations. Prefill remains
serial, with 28 prompt tokens and 88 decode steps to the original EOS. No
profiler, launch blocking, logit dumping, compilation, heavy audit or concurrent
investigation GPU work runs during either timing series. The agent only waits;
the desktop GPU remains shared and unlocked.

| Ordinary mode | Minimum tok/s | Median tok/s | Maximum tok/s |
| --- | ---: | ---: | ---: |
| Qualified original head | 9.1156 | 10.2675 | 10.7046 |
| Two-block unroll | 8.5350 | 10.0907 | 10.7474 |

Candidate serial-prefill median is 9.1212 tok/s, versus 10.4471 for the control.
ABBA and BAAB median ratios are 0.94491 and 0.98278; pooled ratio is 0.98278.
All low results, including 8.8506 and 8.5350 in one candidate process, remain.
The 10.7474 ordinary peak is a new observation, not stable deployment speed.
These ordinary measurements do not confirm an end-to-end improvement, despite
the small favorable same-process result. The candidate remains experimental;
no new head option or default is added on this evidence.

## Bounded scheduling follow-up

A separate six-configuration sweep checks whether lower register pressure or
the opposite compiler scheduling policy improves the current grouped head.
It compares four-row-tile default/two-block unroll with MAX_ILP on/off, and
two-row-tile two-block unroll with MAX_ILP on/off. This is head-only timing:
other kernels compiled under changed flags are not model-qualified and must
not be installed through a wholesale replacement of the GGUF object.

All builds finish before the twelve forward/reverse timing processes. The
harness retains three-second alternating warmup, 100 pairs per process, full
output buffers and canaries. All **36 complete outputs are byte-identical**,
with **288 canaries** intact. All 1200 frequency samples read 1.4 GHz; no clock
is requested or locked, and between-sample changes remain possible. The two
controls reproduce the previous default and two-block complete device images.

| Row tiles / unroll / MAX_ILP | Forward ratio | Reverse ratio | Temp registers |
| --- | ---: | ---: | ---: |
| 4 / default / on, original control | 1.000347 | 1.000181 | 110 |
| 4 / 2 / on, existing candidate | 1.040789 | 1.039957 | 190 |
| 4 / default / off | 0.977912 | 0.977888 | 81 |
| 4 / 2 / off | 0.909584 | 0.909220 | 140 |
| 2 / 2 / on | 0.900635 | 0.900129 | 116 |
| 2 / 2 / off | 0.923446 | 0.923639 | 86 |

Ratios again compare each candidate with the qualified original runtime in
alternating event timing. All six configurations report zero local/spill bytes.
Reducing registers does not improve this workload: every new configuration
regresses in both orders. The existing candidate's roughly 4% head-only result
is reproduced, but does not supersede its inconsistent end-to-end evidence.
No new scheduling variant warrants full-model promotion. The global trace
still prioritizes the much larger Q4/Q5 projection cost over further marginal
head-only tuning; the minimum-ten goal remains unproven.

Local evidence under ETSupport `artifacts/gguf-20261029`:

- `profile_fused_prefix_source_blocking_v1.py`,
  `fused-prefix-source-blocking-profile-v1`: raw short trace, original prefix,
  maps and per-window analysis using the existing blocking analyzer.
- `generate_q6_head_unroll_v1.py`, `q6-head-unroll-probe-v1`: isolated sources,
  five builds, resources, ten processes, complete buffers and unfiltered timing.
- `audit_q6_head_unroll_v1.py`, `q6-head-unroll-decision-v1.json`: independent
  raw-buffer/hash checks, recomputed timing, control identity, resources and
  all 36 raw-trace kernel/allocation/synchronization checks.
- `build_q6_head_unroll_runtime_v1.py`, `build_q6_head_unroll_runtime_v2.py`,
  `build-q6-head-unroll-v2`: retained preflight failure and corrected isolated
  runtime with protected link inputs and exact device-image checks.
- `validate_q6_head_unroll_v1.py`, `q6-head-unroll-validation-v1`,
  `audit_q6_head_unroll_runtime_v1.py`, `q6-head-unroll-runtime-decision-v1.json`:
  complete three-case qualification and independent numerical/provenance audit.
- `alternating-head-probe-v1`, `build-alternating-head-v1`,
  `alternating-head-validation-v1`, `alternating-head-benchmark-v1`,
  `alternating-head-decision-v1.json`: benchmark-only switching, full text
  identity, branch counts and independently recomputed paired timing.
- `benchmark_q6_head_unroll_v1.py`, `q6-head-unroll-benchmark-v1`,
  `audit_q6_head_ordinary_v1.py`, `q6-head-unroll-ordinary-decision-v1.json`:
  original-runner timing, all trajectories/maps and independent raw-rate audit.
- `q6-head-schedule-probe-v1`, `audit_q6_head_schedule_v1.py`,
  `q6-head-schedule-decision-v1.json`: six configurations, retained regressions,
  raw outputs/timings, resources, frequency samples and independent audit.

Stable ten decode tokens/s and ordinary whole-model speedup are not established.
