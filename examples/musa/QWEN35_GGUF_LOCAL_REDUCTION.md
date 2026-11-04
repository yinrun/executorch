# Local-first Q4/Q5 reduction investigation

## Decision

An isolated local-first reduction prototype passes the complete frozen-HF
model gate. Separate-process timing is inconsistent; the subsequent
same-process comparison below supports a small, approximately one-percent
benefit, not stable 10 tokens/s. It is not integrated into default or opt-in
production dispatch. The previous
[packed scheduling source build](QWEN35_GGUF_PACKED_SCHEDULE.md) remains intact.
The 10 tokens/s goal is not accepted from one favorable series or a peak.

This remains Qwen3.5-4B Q4_K_M with FP32 activations, recurrent state and
accumulation. There is no TF32, MTP, requantization, vocabulary pruning,
output-check removal or model re-export. The caller-owned Split-K PTE/PTD,
native runner, extension and historical host link objects are unchanged.

## Refreshed bottleneck attribution

The source-build nonblocking global trace covers the complete original
generation, but asynchronous kernel labels share intervals. A separate
short blocking trace therefore examines 28 prompt tokens and two decode
steps, retaining the original generated prefix and verifying native library
maps and trace PID. It is not an ordinary throughput measurement.

Each decode step has 812 distinct, nonoverlapping kernel-label intervals.
Median summed kernel time is 129.902 ms, of which packed Linear accounts for
110.226 ms, approximately 84.85%. CPU waits and GPU time must not be added.

| Kernel family | Calls per decode | Mean time across two decode steps, ms |
| --- | ---: | ---: |
| Packet/quad Q4, single part | 75 | 45.456 |
| Packet/quad Q4, four parts | 48 | 19.046 |
| Grouped prepared-Q6 output head | 1 | 15.408 |
| Packet/quad Q5 | 24 | 14.879 |
| Prepared Q6, four parts | 16 | 12.956 |
| Narrow Q4 | 48 | 1.893 |
| Prepared Q6, single part | 5 | 0.589 |

The blocking trace supports prioritizing Q4/Q5 work; its absolute timings
must not be substituted for normal asynchronous execution. The device is
shared with desktop applications and clocks are unlocked.

## Mechanism and native microbenchmarks

The previous packet kernel shuffles sixteen component accumulators across
an eight-lane subgroup before combining components. The candidate first
combines components within each lane, then reduces one value across lanes.
The source-level subgroup-shuffle expressions decrease from 48 to 3 per
lane. An intermediate four-component variant uses 12. These are source-level
counts, not a final-ISA or dynamic-cycle measurement.

The original per-block dequantization, FMA accumulation, packed layout,
eight-lane subgroup, row mapping and four-part output ABI remain unchanged.
Only the final reduction tree changes. FP32 addition is not associative,
so this is explicitly a rounding-changing optimization, not a byte-exact one.
It does not move scale factors outside the per-weight dot product.

Three variants, including the unchanged sixteen-component control, run in
forward and reverse order on eight real Q4/Q5 geometries with three changing
inputs. All **144 complete outputs** pass the existing local tolerance;
the control's 48 outputs are additionally byte-identical to the qualified
source runtime. Maximum error over the family is 0.00000143051, with maximum
normalized local-gate error 0.000738012. This is not the full-model gate.

| Selected one-component variant | Forward speed ratio | Reverse speed ratio |
| --- | ---: | ---: |
| Q4 attention gate | 1.0411 | 1.0396 |
| Q5 QKV | 1.0166 | 1.0181 |
| Q4 FFN gate | 1.0362 | 1.0312 |
| Q4 SSM output | 1.0761 | 1.0898 |
| Q4 attention key | 1.0366 | 1.0324 |
| Q4 attention query | 1.0316 | 1.0337 |
| Q4 FFN down | 1.0421 | 1.0396 |
| Merged Q4 gate/up | 1.0260 | 1.0280 |

Ratios are baseline latency divided by candidate latency. Native event
timing alternates both sides, retaining 100 samples after two warmups.
Long-K microbenchmarks include a separate reduction on both sides; the
model's caller-owned partial-output path does not. All compiled variants
report zero local memory and spill bytes. They use the same reported
temporary-register counts for corresponding Q4/Q5 instantiations; these
reports do not establish measured occupancy or stall causes.

## Complete model precision

Only the selected Q4/Q5 kernel function differs in the isolated full-model
source copy. The unchanged historical host objects are relinked with its
kernel object. The prototype runtime SHA-256 is
`cec93027f0c289166410d80ce1a772bd9ce4fbfd600546762dfbd8ed5852decf`.

All **2298 complete 248320-float outputs**, **1149 byte-exact reset pairs**
and **18 negative controls** pass. Every output must be finite, retain exact
per-step argmax and match the original generated IDs/EOS. The unchanged
elementwise bound is `0.001 + 0.001 * abs(reference)` against frozen HF;
the same additional bound is checked against both the old native baseline
and the qualified packet-scheduling source runtime.

| Case | Maximum error vs HF | Maximum normalized HF error | Maximum additional error vs packet source |
| --- | ---: | ---: | ---: |
| Text | 0.0000307560 | 0.0209715 | 0.0000131130 |
| Single image | 0.000627518 | 0.311311 | 0.000165462 |
| Multiple images | 0.000710011 | 0.488506 | 0.000247955 |

The normalized gate limit remains one, with no outliers or argmax exceptions.
Outputs are not byte-identical to the preceding runtime. Smaller observed
errors against HF are not claimed as a general quality improvement. Frozen
HF and prepared image features are reused, not independently rerun. Native
microbenchmarks and this model gate do not replace future production coverage
for arbitrary offsets, row tails, streams or build-option toggling.

## Ordinary whole-model timing

The direct control is the qualified packet-scheduling source runtime, using
the same export. Each process has one full warmup and two measured runs.
ABBA and BAAB retain the original 28-token prompt, 88 decode steps and 89
generated IDs including EOS. Full readback, finite checks and argmax remain;
there is no profiler, launch blocking or logit dumping during timing.
Compilation, heavy audits and investigation GPU workloads are outside timing.

| Series | Baseline median tok/s | Candidate median tok/s | Median change |
| --- | ---: | ---: | ---: |
| ABBA | 9.8401 | 10.0125 | +1.7521% |
| BAAB | 10.2711 | 9.5167 | -7.3444% |
| Pooled, eight measured samples each | 9.9636 | 9.9771 | +0.1349% |

The candidate range is 8.5207--10.2893, versus 9.6695--10.4297 for the control.
The reverse series begins with candidate samples 8.5207 and 9.0054; both
remain in the report. Later candidate samples are 10.2893 and 10.0281.
These measurements do not identify the cause of the variation, and removing
the slower samples would bias the conclusion. Neither the first median above
10 nor the 10.2893 peak establishes stable 10 tokens/s or a reproducible gain.

These inconsistent results motivate the tighter alternating full-generation
comparison below. No successful microbenchmark alone changes the deployed
default, and instrumented timings must remain distinct from the protected
runner's deployment measurements.

## Separate candidate global trace

A fresh nonblocking trace verifies the intended runtime maps and complete
original trajectories. All 88 measured decode steps retain 812 kernel labels,
147 packet/quad projections, twelve stream synchronizations, three event
records and eight async allocations. This confirms that the local reduction
does not change model launch/lifetime counts.

Median host execution is 139.407 ms, compute-track interval union 124.232 ms,
compute-plus-copy union 124.881 ms and post-execution validation gap 0.852 ms.
These profiled intervals are neither normal throughput nor a paired speedup
against the earlier source-build trace. Async kernel labels still share
intervals and cannot be summed to attribute individual kernel costs.

## Same-process alternating follow-up

An artifact-only runner and combined kernel library hold both original
kernel function bodies. A benchmark-only setter chooses the implementation
between complete generations, after synchronization and before timing begins.
The same model instance, packed weights, historical host link objects and
PTE/PTD are reused. The runner is rebuilt with an instrumented generation
header; it is not the protected deployment executable. Full output readback,
finite checks, argmax, original token checks and state reset remain unchanged.

Every generation records its selected mode and separate host launch counters.
Exactly `147 * steps` Q4/Q5 calls must use that mode, with zero calls to the
other mode. This avoids inferring active dispatch from an environment variable
alone. The mutable selector is a sequential benchmark facility, not a public
runtime option or support for concurrent callers/models.

Before timing, each text/image case runs modes baseline/candidate/candidate/
baseline. All **4596 complete outputs** match their respective independently
qualified runtime's recorded output bytes, including **2298 within-mode reset
pairs**. The original IDs/EOS and every per-generation dispatch count match.
Independent auditing rehashes outputs and references against each earlier
qualification report. This reuses the previous frozen-HF acceptance rather
than claiming a new independent HF execution.

Each of two timing processes warms both modes once and measures eight complete
generations. Their measured orders are ABBA/BAAB and BAAB/ABBA, producing
eight adjacent baseline/candidate pairs and eight observations per mode.
Switching and counter collection are outside per-token timing; neither
model computation nor the existing output checks are removed from it.
No compilation, heavy audit or other investigation GPU work overlaps timing.

| Process | Baseline median tok/s | Candidate median tok/s | Paired geometric-mean change |
| --- | ---: | ---: | ---: |
| First balanced order | 7.7780 | 7.7323 | +0.4466% |
| Reverse balanced order | 7.7609 | 7.8954 | +1.4910% |
| Combined, eight samples per mode | 7.7693 | 7.8751 | +0.9675% |

The median adjacent-pair speed ratio is **1.01290**; **seven of eight pairs**
improve. The negative pair's ratio **0.97047** remains included. Unpaired
median improvement is 1.3618%, a different statistic from the paired values.
The candidate range is 7.5593--7.8985 versus 7.4222--7.7991 for the control.
The first process's lower candidate median despite positive paired aggregate
shows that temporal variation still matters. These results support a modest
benefit under this instrumented comparison, not statistical significance,
exclusive-GPU throughput or a guarantee for every generation. They must not
be multiplied by the earlier approximately 10 tokens/s measurements.

A separate full-generation trace verifies both modes' actual kernel names:
all 232 warmup/measured steps retain 812 kernel records and exactly 147 calls
to the chosen packet or local-first kernel, with none to the other. The
analyzer explicitly excludes the added generation-boundary synchronizations
from its per-step boundary pairing. Candidate measured-decode medians are
136.930 ms host execution and 112.491 ms compute-track interval union. These
are profiled intervals, not ordinary throughput or a paired attribution.
The profiler's exceptional Pixel-event warning is retained; selected model
compute intervals pass monotonic checks.

The instrumented runtime SHA-256 is
`e6fe684df153641425d28b1366d5ee0e5d479ba283a222a53dd449abfdcce998`;
its runner SHA-256 is
`6c50cd6e0a79880769189be5fca3659798b7ccf3e7944e69b22f5fa7804ad2a1`.
Protected deployment files and production source are unchanged. The next
kernel investigation tests fixed matrix dimensions while preserving FP32
reduction order; the approximately one-percent reduction benefit alone is
not sufficient evidence of reaching 10 tokens/s.

## Local evidence

ETSupport artifacts under `artifacts/gguf-20261029`:

- `profile_packed_schedule_blocking_v1.py`,
  `analyze_packed_schedule_blocking_v1.py`, `packed-schedule-blocking-profile-v1`:
  new short blocking attribution with native maps and original prefix checks.
- `local-reduction-probe-v1`: three kernel variants, resource logs, native
  harness, forward/reverse results and the isolated full-model source copy.
- `build_local_reduction_v1.py`, `build-local-reduction-v1`:
  protected-host relink and source/runtime hashes, not a production build.
- `validate_local_reduction_v1.py`, `local-reduction-validation-v1`:
  unchanged full-model gate, reset checks and full logits.
- `benchmark_local_reduction_v1.py`, `benchmark_local_reduction_v2.py`,
  `local-reduction-benchmark-v1/v2`: both ordinary timing orders and all samples.
- `profile_global_local_reduction_v1.py`, `analyze_global_local_reduction_v1.py`,
  `global-local-reduction-profile-v1`: separate complete-generation capture.
- `audit_local_reduction_v1.py`, `local-reduction-decision-v1.json`:
  independent output/reference rechecks, protected hashes and timing audit.
- `alternating-reduction-probe-v1`, `build_alternating_reduction_v1.py`,
  `build-alternating-reduction-v1`: isolated dual-kernel source, explicit
  generation-boundary selector, counters and instrumented runner build.
- `alternating_reduction_common.py`, `validate_alternating_reduction_v1.py`,
  `alternating-reduction-validation-v1`: complete per-mode output and reset gate.
- `benchmark_alternating_reduction_v1.py`, `alternating-reduction-benchmark-v1`:
  both within-process balanced orders and every adjacent pair.
- `profile_global_alternating_reduction_v1.py`,
  `analyze_global_alternating_reduction_v1.py`,
  `global-alternating-reduction-profile-v1`: mode-specific dispatch verification.
- `audit_alternating_reduction_v1.py`, `alternating-reduction-decision-v1.json`:
  independent byte/hash/trajectory/counter checks and paired timing analysis.
