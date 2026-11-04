# Stream priority and in-generation telemetry

This is an isolated diagnostic follow-up to the mixed ordinary timings of
[local-first reduction](QWEN35_GGUF_LOCAL_REDUCTION.md), not a deployment change.
The model remains Qwen3.5-4B Q4_K_M with FP32 activations/state/accumulation.
The qualified local-first shim, original extension, Split-K PTE/PTD and
complete output checks are unchanged. The 10 decode tokens/s goal remains unmet.

## Why test scheduling

Small kernel improvements are obscured by substantial shared-device variation.
The previous global capture still shows most of the execution window covered
by GPU compute. Changing stream scheduling is a separate hypothesis, not a
claim that the desktop or GPU clock caused those earlier slow measurements.

On this M1000/SDK combination, `musaDeviceGetStreamPriorityRange` returns zero
for least priority and -2 for greatest priority. An artifact-only runner tests:

1. The backend's original per-thread stream, with no caller guard.
2. An explicitly owned nonblocking stream with priority zero.
3. An explicitly owned nonblocking stream with priority -2.

Both owned streams are created once, queried for actual flags/priority, and
destroyed after synchronization. `CallerStreamGuard` selects them only across
complete sequential generations. Mode switches and the added pre-generation
synchronization occur outside the generation timer. There is no global stream
override, clock change, process affinity/priority change, or termination of a
desktop application. This does not establish concurrent-model correctness.

## Correctness and actual routing

The six-generation `012210` text sequence passes **696 complete 248320-float
outputs**, all byte-identical to the qualified local-first corpus. Each mode
has two reset generations, or 116 reset pairs. Original IDs/EOS, finite checks
and argmax validation remain enabled. Image cases and the raw-image frontend
are not rerun; this is text-only qualification, not the full three-case gate.

A separate untimed interposer checks the stream passed to both runtime and
driver kernel-launch APIs against the active caller selection, and queries
the owned stream's actual priority. Each full text generation observes 47,560
runtime launches and 94,192 driver launches: 410 and 812 per step respectively.
All three modes pass. These counts overlap because runtime launches can use
the driver API underneath; they must not be added. The interposer is absent
from every timing process, as verified by environment and runtime mappings.

Two unsuccessful routing checks are retained and excluded from qualification.
The first hooked an AOTI stream getter that this export did not call and saw
zero observations. The next interposer initially failed to compile because
the runtime launch API is overloaded; an incorrectly started run without its
library was terminated. The corrected harness requires the preload file to
exist and verifies its actual mapping before accepting a run. The successful
runtime/driver hook is a separate fresh run, not a selected result from either
failed attempt.

## Balanced diagnostic timing

Two processes run opposite balanced orders. Each warms all three modes once,
then measures four complete generations per mode: eight measurements per mode
in total. Each uses the original 28-token prompt and 88 decode steps to EOS,
with full readback/finite/argmax checks and no profiler, launch blocking or
logit dumps. The runner is instrumented and a host-side sampler reads sysfs
and `/proc` approximately every 200 ms. These are diagnostic timings, not a
replacement for historical ordinary protected-runner benchmarks.

| Mode | Minimum tok/s | Median tok/s | Maximum tok/s |
| --- | ---: | ---: | ---: |
| Original per-thread stream | 7.4643 | 7.5038 | 7.6034 |
| Nonblocking, priority zero | 7.4238 | 7.5203 | 7.5687 |
| Nonblocking, priority -2 | 7.4674 | 7.5337 | 7.5826 |

Across eight balanced three-generation groups, highest priority versus the
original stream has a paired geometric-mean ratio of **1.001745**, about
**+0.17%**; five groups improve and three regress. Highest versus ordinary
nonblocking priority is +0.32%; ordinary nonblocking versus the original is
-0.15%. All samples remain in the reports. These tiny mixed differences do
not establish a useful scheduling improvement or justify a production option.
Do not multiply them by the historical 10.58 tok/s peak.

## What telemetry establishes

The two processes retain 1203 and 1192 samples. Every sampled GPU frequency
inside a measured generation is **1.4 GHz**, not the 730 MHz observed before
launch. Therefore the idle snapshot is not evidence of inference running at
730 MHz. Sampling does not exclude changes between samples or explain the
older 8--10 tok/s variation. CPU frequency varies across the three policies;
no frequencies or governors are modified.

Sampler read durations sum to 1.297 and 1.294 seconds across the respective
processes; these are observation durations, not a measured additive slowdown.
The native main thread's sampled scheduler CPU time has a median of 15.255
seconds for generations whose median wall time is 15.441 seconds. Its median
sampled runqueue wait is only 0.00218 seconds, with a maximum of 0.03080 seconds.
CPU-time windows omit the edges between samples, so these are approximate
generation-level observations rather than per-token CPU profiles.

Near-continuous CPU residency is compatible with active runtime waiting; it
does not mean inference arithmetic moved to CPU, nor prove polling is the
bottleneck. A next controlled experiment can compare the runtime's host wait
policies and inspect CPU wait behavior, preserving the same GPU math and
output checks. That follow-up is now complete in
[host wait-policy investigation](QWEN35_GGUF_WAIT_POLICY.md): blocking saves
substantial CPU time but slightly reduces throughput. Merely increasing GPU
stream priority did not solve this case.

## Evidence

Under ETSupport `artifacts/gguf-20261029`:

- `stream-priority-probe-v1`, `build_stream_priority_v1.py`,
  `build-stream-priority-v1`: instrumented runner, routing probes and build hashes.
- `stream_priority_common.py`, `validate_stream_priority_text_v1.py`,
  `stream-priority-text-validation-v1`: complete three-mode text outputs.
- `stream-priority-route-validation-v1/v2`: excluded routing attempts.
- `stream-priority-route-validation-v3`: verified runtime/driver routing.
- `benchmark_stream_priority_v1.py`, `stream-priority-benchmark-v1`:
  balanced orders, every raw timing, runtime maps and timestamped telemetry.
- `audit_stream_priority_v1.py`, `stream-priority-decision-v1.json`:
  independent complete-output rehash, routing and timing/telemetry audit.
