# Q6 four-part projection input reuse

This follows [shared-input block grouping](QWEN35_GGUF_SHARED_BLOCKS.md) and
[Q6 output-head input reuse](QWEN35_GGUF_Q6_HEAD_INPUT.md). The qualified
512-thread Q4/Q5 blocking trace still attributes about 12.763 ms per decode
step to sixteen Q6 four-part projections. This experiment changes that path,
not the head or single-part Q6 path. It retains the original Qwen3.5-4B Q4_K_M
weights, prepared Q6 representation, FP32 input/state/accumulation and exact
FMA and reduction order. No TF32, MTP, requantization or relaxed checks are used.

## Isolated variants

Each lane retains four components, each 32-lane group computes two output rows,
and each launch retains four partitions. Shared-input candidates stage only
their partition once per CTA, synchronize before any row-boundary return, and
load the same float4 values from shared memory. The staging allocation is
bounded by 4096 floats; larger partitions use the original kernel. Input is
reloaded on the calling stream for every invocation, with no persistent cache.

| Mode | Threads | Input access | Output rows per CTA |
| --- | ---: | --- | ---: |
| 0 | 128 | Original global/read-only | 8 |
| 1 | 128 | Shared | 8 |
| 2 | 256 | Shared | 16 |
| 3 | 512 | Shared | 32 |
| 4 | 256 | Global/read-only control | 16 |
| 5 | 512 | Global/read-only control | 32 |

The real 2560-by-9216 FFN-down matrix has nine quantization blocks in each
partition, so each shared CTA stages 2304 floats, or 9 KiB. At 128/256/512
threads, total logical staging is 11.25/5.625/2.8125 MiB per projection.
These are logical input loads, not measured DRAM traffic or a bandwidth claim.

The original production source and installed artifacts are unchanged. The
mode-zero complete device image matches the qualified 512-thread source image.
All five modified images preserve all fifty original device-function bodies
and all twenty-six original kernel-metadata records byte for byte; only the
new candidate is added and selected behind the existing guarded parts entry.

## Audited paired microbenchmark

The six variants run in forward and reverse order on four cases: the real
FFN-down matrix, its first 129 and 159 rows, and a synthetic 129-by-4352 reshape
of valid Q6 blocks to exercise unequal partition lengths. The latter is not
a production-model shape or performance claim. Each process retains three
seconds of active warmup, 100 alternating timing pairs, three complete output
comparisons and 24 boundary-canary checks. Packing is outside kernel timing.

Independent auditing verifies all 144 complete four-part vectors, not just
their summed outputs: every value is finite and byte-exact to the qualified
runtime. All 1152 canary checks and 4800 timing pairs are retained, including
negative results. Frequency endpoints comprise 4328 samples at 730 MHz and
472 at 1400 MHz; clocks and desktop applications are not changed, and endpoint
observations cannot establish constant frequency or explain causation.

| Mode | Real FFN forward ratio | Real FFN reverse ratio | Tail 129 ratios | Tail 159 ratios | Unequal-part ratios |
| --- | ---: | ---: | --- | --- | --- |
| Original 128 | 1.00089 | 1.00079 | 0.99837 / 1.00194 | 0.99924 / 0.99772 | 1.00018 / 0.99751 |
| Shared 128 | 1.08810 | 1.09037 | 0.86951 / 0.86987 | 0.90671 / 0.90375 | 0.98447 / 1.00458 |
| Shared 256 | 1.13880 | 1.13477 | 1.01358 / 1.01420 | 1.04009 / 1.03980 | 0.98848 / 0.98920 |
| Shared 512 | 1.11712 | 1.30829 | 0.97645 / 0.97853 | 1.02708 / 1.02649 | 1.00207 / 0.99143 |
| Global 256 | 0.99202 | 1.00034 | 0.99486 / 1.00098 | 1.00258 / 1.00363 | 1.00185 / 1.00420 |
| Global 512 | 0.99643 | 0.99417 | 0.95573 / 0.95093 | 0.97560 / 0.97408 | 0.99718 / 0.99830 |

Ratios are baseline median divided by candidate median within each paired
process, not comparisons of raw timings between processes. Shared 256 gives
the more consistent real-matrix improvement; shared 512 is also positive in
both orders but less uniform. Both warrant full-model qualification. Shared
128 regresses on row tails; enlarging global-input blocks alone provides no
general gain. No candidate is claimed universally faster across shapes.

The original and global controls use 56 temporary registers; all shared modes
use 55. Static shared-memory accounting is 20 bytes for global and 32 for shared,
in addition to the shared candidates' dynamic allocation. All variants report
zero local memory and zero spill loads/stores.

## Native and complete-model qualification

Both 256- and 512-thread shared candidates have separately linked runtimes,
each replacing only the GGUF device object. Each passes 18 GGUF native tests
and 15 attention tests; the unavailable second-device test is skipped. The
isolated object image matches the corresponding measured microbenchmark image.
Protected original runners, exports and other link inputs remain unchanged.

Independent audits reread every complete output, reset counterpart and frozen
HF reference for both candidates:

| Case | Complete outputs per candidate | Reset pairs per candidate | Maximum absolute error against HF |
| --- | ---: | ---: | ---: |
| Text | 232 | 116 | 0.0000262260 |
| Single image | 856 | 428 | 0.0006752014 |
| Multiple images | 1210 | 605 | 0.0008490086 |

Each candidate has 2298 byte-exact outputs, 1149 reset pairs, eighteen negative
controls and unchanged original IDs/EOS. The totals across both are 4596 full
outputs and 2298 reset pairs. The unchanged gate is
`abs(actual-reference) <= 0.001 + 0.001*abs(reference)`, with finite values and
matching argmax. Prepared image features and frozen HF references are reused;
this is not a fresh raw-image frontend or HF execution test.

## Blocking and complete nonblocking traces

Each candidate has an independently audited blocking capture of 28 prefill
and eight decode windows. Each step retains 788 kernels, 147 shared Q4/Q5
calls, sixteen new Q6-part calls, one original Q6 head, eight fused attention
calls, twelve stream synchronizations and no asynchronous allocations/frees.
The original non-shared Q6 parts kernel does not execute in these windows.

| Blocking decode median (ms) | Prior 512-thread Q4/Q5 source path | Shared Q6 256 | Shared Q6 512 |
| --- | ---: | ---: | ---: |
| Sum of nonoverlapping kernels | 106.1287 | 104.4483 | 104.4910 |
| Packed Linear aggregate | 87.1365 | 85.5049 | 85.5597 |
| Q4 single-part aggregate | 32.0014 | 32.0574 | 31.9953 |
| Q4 four-part aggregate | 14.3685 | 14.3954 | 14.3475 |
| Q5 aggregate | 10.1974 | 10.0970 | 10.1617 |
| Q6 four-part aggregate | 12.7634 | 11.1910 | 11.2694 |
| Unchanged Q6 output head | 15.3248 | 15.3291 | 15.3233 |

These separate instrumented captures show the intended Q6-part path and about
1.5--1.6 ms less attributed Q6-part time. They are not a causal ordinary-model
speedup estimate. The Q4 single-part aggregate and output head remain large
individual targets, alongside about 18.97 ms of other kernels.

Each candidate also has a fresh complete nonblocking trace of one warmup and
one measured generation: 232 windows, original 28-token prompt, 89 generated
IDs including EOS, and 88 decode steps per generation. Independent SQLite
audits verify every window, runtime mapping, kernel/API count and all phase
minimum/median/maximum/total summaries. The two candidates' measured decode
medians are:

| Nonblocking quantity (ms) | Shared Q6 256 | Shared Q6 512 |
| --- | ---: | ---: |
| Host execute window | 117.1468 | 117.1570 |
| Compute-track interval union | 103.0167 | 103.0597 |
| Copy-label interval union | 0.6770 | 0.6804 |
| Post-execute validation gap | 0.8501 | 0.8499 |

Async kernel labels overlap and are not suitable for summed kernel attribution.
The compute-track union is independently reconstructed; CPU, GPU and copy
durations must not be added. None of these profile values is ordinary TPS.

## Ordinary-model comparison and decision

The original uninstrumented runner compares the qualified 512-thread Q4/Q5
source against both Q6-part candidates. Two opposite three-mode orders keep
twelve full warmups and all 24 measured generations, eight per mode. Agent
activity during native timing is waiting only. There is no profiling, frequency
sampling, blocking launch mode, logit dump, model reexport or token/EOS change.

| Runtime | Decode minimum | Decode median | Decode maximum | Serial prefill median |
| --- | ---: | ---: | ---: | ---: |
| Qualified 512-thread Q4/Q5 source | 9.3232 | 11.8400 | 12.9243 | 10.2120 |
| Plus shared Q6 256 | 9.9536 | 12.2061 | 13.3905 | 11.4794 |
| Plus shared Q6 512 | 9.5670 | 12.9193 | 13.4882 | 12.9382 |

All rates are tokens/s. Unpaired median ratios to this batch's control are
1.03092 and 1.09116, respectively; these are not isolated causal speedups.
The first series has candidate minima 10.5006 and 12.7828, but the second has
9.9536 and 9.5670. Both series and every low observation remain in acceptance.
In particular, 9.9536 is below ten and must not be rounded into a pass.

The 13.4882 observation is a new historical ordinary-model peak. The 512-thread
Q6 candidate also has the higher batch median, while 256 has the higher minimum
and more consistent microbenchmark gain. Neither is yet a stable-ten result.
Retain both isolated candidates for same-process full-model pairing before
selecting a source option; do not select from the peak alone. No default or
installed deployment is changed by this experiment.

## Evidence

Under ETSupport `artifacts/gguf-20261029`:

- `q6-parts-input-reuse-probe-v1/{kernels.mu,bench.cpp,build.py,run.py}`:
  all six isolated variants, raw complete partial vectors and timing pairs.
- `audit_q6_parts_input_reuse_v1.py`, `q6-parts-input-reuse-decision-v1.json`:
  independent output, resource, device-function/metadata and timing audit.
- `build_q6_parts_input_{256,512}_v1.py`, `build-q6-parts-input-{256,512}-v1`,
  `validate_q6_parts_input_{256,512}_v1.py`,
  `q6-parts-input-{256,512}-validation-v1`,
  `audit_q6_parts_input_{256,512}_runtime_v1.py`,
  `q6-parts-input-{256,512}-runtime-decision-v1.json`: both isolated builds,
  native tests and independent complete three-case precision qualification.
- `profile_q6_parts_input_{256,512}_blocking_v1.py`,
  `q6-parts-input-{256,512}-blocking-profile-v1`,
  `analyze_q6_parts_input_{256,512}_blocking_v1.py`,
  `audit_q6_parts_input_{256,512}_profile_v1.py`,
  `q6-parts-input-{256,512}-profile-decision-v1.json`: short blocking captures,
  all-window kernel/API counts, interval reconstruction and independent audits.
- `profile_global_q6_parts_input_{256,512}_v1.py`,
  `global-q6-parts-input-{256,512}-profile-v1`,
  `analyze_global_q6_parts_input_{256,512}_v1.py`,
  `audit_q6_parts_input_{256,512}_global_v1.py`,
  `global-q6-parts-input-{256,512}-profile-decision-v1.json`: complete nonblocking
  traces and independently reconstructed window/phase summaries.
- `benchmark_q6_parts_staging_v1.py`, `q6-parts-staging-benchmark-v1`,
  `audit_q6_parts_staging_ordinary_v1.py`,
  `q6-parts-staging-ordinary-decision-v1.json`: all ordinary timings, warmups,
  original IDs/EOS, runtime mappings and protected inputs.

These are kernel-level measurements, not ordinary-model throughput. Stable
minimum model decode throughput of ten tokens/s remains unproven.
