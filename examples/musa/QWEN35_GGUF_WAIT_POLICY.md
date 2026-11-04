# Host wait-policy investigation

This follows the [stream-priority experiment](QWEN35_GGUF_STREAM_PRIORITY.md).
The model remains Qwen3.5-4B Q4_K_M, with FP32 activations, state and
accumulation. The qualified local-first runtime, original extension,
Split-K PTE/PTD, GPU arithmetic and output checks are unchanged. Blocking
host waits greatly reduce CPU consumption but do not improve decode
throughput. No production default or deployment is changed, and stable
10 decode tokens/s remains unproved.

## Controlled policy changes

An artifact-only native runner selects auto, spin, yield or blocking host
waiting before each complete generation. The SDK 5.1 runtime permits updating
these flags on an initialized current device. The controller synchronizes
before changing them, preserves every non-scheduling bit, verifies the result
through both `musaGetDeviceFlags` and driver `muCtxGetFlags`, and restores and
checks the initial flags at the end. These operations are outside the
generation timer. The original backend per-thread stream remains selected.

The observed scheduling bits are respectively 0, 1, 2 and 4. Initial flags
are zero. No OS scheduler, process priority, GPU clock or governor is changed,
and no other application is stopped. This is a sequential single-context
experiment, not concurrent-service qualification.

## Text correctness

The eight-generation sequence `01233210` passes **928 complete 248320-float
outputs**, byte-identical to the qualified local-first text corpus. Each of
the four policies has two generations and 116 reset pairs. Original token
IDs/EOS, finite checks and exact argmax remain enabled. Images and the
raw-image frontend are not rerun; this is not a new full three-case HF gate.

The original validation producer accidentally retains a status label naming
three modes. The raw report is preserved. An independent audit verifies all
eight runs, all four actual runtime/driver flag values and every complete
output byte; its status explicitly names four policies. The timing and final
audits require that independent report, not the stale label.

## Balanced diagnostic timing

Two processes each warm all four policies, then measure four generations per
policy in opposite balanced orders. Eight four-generation comparison groups
and eight measurements per policy are retained, including all low results.
Each generation uses the original 28-token prompt, 88 decode steps and EOS.
There is no profiler, launch blocking or logit dumping during this timing.

The instrumented runner records main-thread CPU time over the complete
generation using `CLOCK_THREAD_CPUTIME_ID`. A separate read-only sampler reads
GPU/CPU frequency and process scheduling data about every 200 ms. These are
diagnostic timings, not replacements for ordinary protected-runner records.

| Host wait policy | Minimum tok/s | Median tok/s | Maximum tok/s | Median thread CPU seconds | Median CPU/wall ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Auto | 7.8016 | 7.8513 | 7.8790 | 14.7246 | 0.9974 |
| Spin | 7.8163 | 7.8578 | 7.9066 | 14.7258 | 0.9974 |
| Yield | 7.8167 | 7.8446 | 7.9111 | 14.7284 | 0.9959 |
| Blocking | 7.6780 | 7.7649 | 7.8187 | 1.9241 | 0.1285 |

Paired geometric-mean throughput ratios against auto are 1.001842 for spin,
1.000701 for yield and **0.988650 for blocking**. Spin and yield each improve
four of eight groups; blocking improves none. Blocking reduces median
generation CPU time by **86.93%**, but decode throughput falls about **1.13%**.
CPU seconds cover serial prefill and decode together, not decode alone.

The processes retain 1511 and 1517 telemetry samples; sampling read durations
sum to 1.587 and 1.605 seconds, not measured additive slowdowns. Measured-run
GPU frequency samples are 1.4 GHz except one blocking generation that includes
1.3 GHz. Sampling does not exclude changes between observations or establish
the cause of timing differences. The device remains shared and unlocked.

## Complete-generation global trace

A separate `msys --trace=musa,osrt` capture runs auto/blocking warmups followed
by one measured generation of each. All four original trajectories and
runtime/driver policies pass. This profile is not ordinary throughput data.

| Measured decode statistic | Auto | Blocking |
| --- | ---: | ---: |
| Profiled decode tok/s | 7.1979 | 7.1711 |
| Median host execution window, ms | 138.061 | 138.530 |
| Median compute-track union, ms | 113.560 | 112.373 |
| Median output-validation gap, ms | 1.314 | 1.321 |

Every step has 812 kernel labels, including 147 packed-quad labels. Each
measured step retains 12 stream synchronizations, eight async allocations and
three event records. The main thread consumes 15.820 versus 2.659 CPU seconds
over the two profiled complete generations. CPU waits and GPU intervals
overlap and must not be added. Async kernel labels can share intervals, so
their summed durations are not reliable per-kernel attribution.

The exporter reports two inverted intervals, one Compute and one Pixel.
Both timestamps precede the first native model runtime/device event, and
neither overlaps any warmup or measured generation. SQLite export contains no
negative-duration intervals. The warning text does not identify their PIDs;
no attribution to a particular application is made. Model-window analysis is
retained, but this trace must not be used for complete pre-model device
accounting. The independent audit records warning timestamps and this limit.

The [all-process device-overlap analysis](QWEN35_GGUF_DEVICE_OVERLAP.md)
subsequently checks every exported engine on the same GPU and independently
verifies all 464 windows. Substantial graphics overlap is observed, but is
not interpreted as additive lost time or an exclusive-device speed estimate.

## Interpretation and evidence

Near-continuous CPU residency in the default mode largely disappears when
the runtime blocks instead of actively waiting, without a corresponding GPU
throughput improvement. This supports active waiting as a major CPU-time
consumer, not CPU inference arithmetic as the decode bottleneck. Reducing
CPU utilization is useful independently, but is not a solution to 10 tok/s.
Further throughput work should address GPU kernels and execution overhead,
not infer a speedup from CPU utilization alone.

Local evidence under ETSupport `artifacts/gguf-20261029`:

- `wait-policy-probe-v1`, `build-wait-policy-v1`: isolated runner source,
  build commands and protected input hashes.
- `wait_policy_common.py`: actual flag, library-map, trajectory and telemetry
  checks; no preload in timing runs.
- `wait-policy-text-validation-v1`, `wait-policy-text-validation-audit-v1.json`:
  retained raw report and independent four-policy full-output verification.
- `wait-policy-benchmark-v1`: both native runs, all samples and telemetry.
- `global-wait-policy-profile-v1`: raw trace, warnings, native results and
  interval-union analysis over all 464 steps.
- `audit_wait_policy_v1.py`, `wait-policy-decision-v1.json`: independent
  hash, output, timing, flag, frequency and trace-warning audit.
