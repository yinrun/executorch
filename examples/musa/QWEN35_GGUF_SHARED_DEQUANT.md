# Shared-input GEMV dequantization follow-up

This follows the [shared-input experiment](QWEN35_GGUF_INPUT_REUSE.md).
The guarded source blocking trace still attributes 99.763 of 118.678 ms of
median kernel time to packed Linear. The experiment changes instruction
expression within the existing Q4/Q5 packed GEMV body, not weights, layout,
accumulation order or shared-memory capacity. Q6 kernels are unchanged.

The workload remains Qwen3.5-4B Q4_K_M on M1000 MP22 with FP32 activation,
state and accumulation. No TF32, requantization, MTP, vocabulary pruning,
tolerance relaxation or output-check removal is used. The initial isolated
experiments are followed by the default-off source option documented below;
deployment defaults do not change.

## Variants

All four variants use the qualified transposed shared-input source path:

| Mode | Change relative to shared-input source |
| --- | --- |
| 0 | None; complete device image matches source ON |
| 1 | Convert four integer scale/minimum factors and multiply as vectors |
| 2 | Permit FP contraction only in the vector dequantization helper |
| 3 | Combine modes 1 and 2 |

The compiler command retains `-ffp-contract=off`; the helper alone has
`#pragma clang fp contract(fast)`. The four accumulation vectors still use the
same explicit FMA sequence and local-first reduction. For finite GGUF half
scales, multiplying an 11-bit significand by a six-bit group scale and a
five-bit level fits within FP32 precision. This motivates testing contraction
without assuming an arbitrary FP expression is exact. Real outputs and native
boundary tests remain the acceptance criteria. The experiment changes the
common body, including its non-shared fallback; it is not a shared-only source
feature. Existing capacity, shape, type and stream guards remain intact.

An earlier vector-contraction experiment on the original global-input path did
not establish a useful general speedup. This retests the interaction with the
new shared-input scheduling, rather than reclassifying that earlier negative
result.

## Paired microbenchmark

Eight real matrices, four modes and two opposite traversal orders produce 64
processes. Every process has at least three seconds of active warmup, 100
alternating timing pairs, three complete finite byte-exact output comparisons
and 24 guard checks. Independent auditing verifies all 192 complete outputs,
1,536 guard checks and 6,400 timing pairs, plus input/source/binary hashes.
No output mismatch or spill occurs. Frequency endpoint samples comprise 4,335
at 730 MHz and 2,065 at 1,400 MHz; no sample is discarded.

Ratios below are baseline median divided by candidate median, in forward /
reverse traversal order. These are microbenchmarks, not model throughput:

| Matrix | Vector metadata | Contraction | Combined |
| --- | ---: | ---: | ---: |
| Q4 attention gate, 4096 x 2560 | 1.0424 / 1.0019 | 1.0288 / 0.9395 | 0.9547 / 1.0087 |
| Q5 QKV, 8192 x 2560 | 1.0596 / 1.0564 | 1.0207 / 1.0035 | 1.0702 / 1.0987 |
| Q4 FFN gate, 9216 x 2560 | 1.0259 / 1.0665 | 1.0206 / 1.0251 | 1.0958 / 1.0480 |
| Q4 SSM out, 2560 x 4096 | 1.0192 / 1.0393 | 1.0306 / 1.0117 | 1.0440 / 1.0304 |
| Q4 attention K, 1024 x 2560 | 1.0246 / 1.0336 | 1.0216 / 1.0157 | 1.0430 / 1.0456 |
| Q4 attention Q, 8192 x 2560 | 1.0244 / 1.0640 | 1.0186 / 1.0220 | 1.0917 / 1.0508 |
| Q4 FFN down, 2560 x 9216 | 1.0287 / 1.0302 | 1.0170 / 1.0174 | 1.0354 / 1.0474 |
| Q4 merged gate/up, 18432 x 2560 | 1.0263 / 1.0434 | 1.0237 / 1.0220 | 1.0515 / 1.0751 |

The identical-code mode-0 attention-gate control itself measures 0.9244 /
0.9914, so that matrix's inconsistent candidate result is retained rather than
used to justify a shape-specific exception. Combined mode improves both orders
on seven of eight matrices and is selected for isolated full-model testing,
not deployment. Long-K microbenchmarks include the same separate part reduction
on both sides; the model uses its existing caller-owned parts path.

Q4 single-part / Q4 four-part / Q5 register counts are 103 / 106 / 109 for
modes 0 and 2, and 104 / 107 / 110 for modes 1 and 3. All three relevant kernels
in every mode have zero local memory and zero spill loads/stores. Resource
counts alone are not a performance prediction.

## Isolated runtime qualification

The combined-mode object is linked into a separate runtime by replacing only
the GGUF device object in the qualified fused-attention link. Its device image
matches the measured mode-3 image. All protected runner, model, baseline and
other link-input hashes remain unchanged. This is an isolated link experiment,
not a clean whole-project rebuild or an installed production runtime.

The current guarded-source native test binaries load this candidate by an
explicit runtime path: all 17 GGUF and 15 attention tests pass. The second-device
restoration test skips on this single-device system. Full-model validation then
retains the original IDs/EOS, finite logits, unchanged frozen-HF elementwise
bound `0.001 + 0.001 * abs(reference)`, and byte-exact reset comparisons:

| Case | Complete output vectors | Reset pairs | Maximum absolute error vs frozen HF |
| --- | ---: | ---: | ---: |
| Text | 232 | 116 | 0.0000262260 |
| Single image | 856 | 428 | 0.0006752014 |
| Multiple images | 1210 | 605 | 0.0008490086 |

All 2,298 complete 248,320-element vectors are byte-identical to the qualified
fused-attention baseline. There are 1,149 reset pairs and 18 negative controls.
An independent audit rereads every raw output and reference, verifies loaded
libraries, compile/link provenance, native test logs and all original token
trajectories. Prepared image features and frozen HF outputs are reused; the raw
image frontend and HF model are not rerun. Long-context and concurrent-service
acceptance are not implied.

## Whole-model attribution

A separate launch-blocking capture retains 28 prefill steps and eight decode
steps with the original nine-token prefix. Independent SQLite auditing checks
all 36 windows. Each has 788 distinct nonoverlapping kernels, 147 shared-input
GEMVs (75 Q4 single-part, 48 Q4 four-part, 24 Q5), eight fused-attention kernels,
twelve stream synchronizations and no async allocation/free. There are no
original packed-GEMV fallbacks in this workload.

| Blocking decode attribution | Shared-input source trace | Combined experiment trace |
| --- | ---: | ---: |
| Total kernel median (ms) | 118.678 | 113.355 |
| Packed Linear aggregate median (ms) | 99.763 | 94.370 |
| Other kernels aggregate median (ms) | 18.939 | 18.965 |
| Q4 single-part aggregate median (ms) | 41.745 | 38.030 |
| Q4 four-part aggregate median (ms) | 15.895 | 15.134 |
| Q5 aggregate median (ms) | 11.568 | 10.671 |
| Q6 head median (ms) | 15.329 | 15.326 |
| Q6 four-part aggregate median (ms) | 12.774 | 12.767 |

These are separate instrumented captures, not a causal speedup estimate. The
combined trace has a 216.913 ms host execution-window median and 113.435 ms
compute-track union; neither is ordinary-runner throughput. Independent
medians need not add exactly. Packed Linear remains about 83% of the measured
kernel total and is still the main target.

## Ordinary full-generation timing

The original uninstrumented runner compares qualified shared input against
combined mode in ABBA followed by BAAB order. Each of eight processes retains
one complete warmup and two measured generations: eight measured generations
per mode, each with 28 prompt tokens, 89 generated IDs including EOS and 88
decode steps. No profiler, launch blocking, logit dump or clock manipulation is
enabled. The agent performs no competing work during timing.

| Runtime | Minimum | Median | Maximum decode tokens/s | Serial-prefill median |
| --- | ---: | ---: | ---: | ---: |
| Shared-input source | 9.0077 | 10.3612 | 12.1569 | 9.7159 |
| Shared input plus vector metadata/contraction | 9.0205 | 10.8044 | 12.3928 | 11.2235 |

The pooled median ratio is 1.04278; separate ABBA and BAAB ratios are 1.04795
and 1.01865. The eight candidate decode rates, in measurement order, are
9.9794, 9.3134, 12.2976, 11.6295, 12.3928, 11.8713, 9.0205 and 9.7168.
All lows, warmups and original output checks remain in the evidence. Because
these are separate processes on a shared GPU, the ratios are not claimed as
causal performance gains. Serial prefill is not batched prefill.

The 12.3928 observation is a new ordinary-runner peak, not a guaranteed
deployment rate. The minimum 9.0205 still fails the ten-token/s objective;
this observed slow generation averages about 110.859 ms per decode step and
would need roughly another 9.8% time reduction to reach 100 ms. This arithmetic
does not predict performance across other workloads or GPU conditions.

## Same-process generation pairing

A separate switching runner compares shared input with the combined variant
inside one process, switching only between complete generations. The device
audit verifies all 50 baseline function bodies and 44 executed candidate
function bodies against their qualified images, plus 49 kernel metadata records
after entrypoint-name normalization. Six candidate fallback functions are
explicitly excluded: the switching probe retains the original fallback for
out-of-capacity shapes. All 147 intended GEMV calls per model step are checked
to use the requested shared-input variant, so this exception is not exercised
by the measured workload. This is not complete-image or host-library identity.

Both modes first pass 464 full text-output comparisons, including 232 reset
pairs, against their independently qualified ordinary runtimes. Two opposite
sequences then retain one full warmup per mode per process and eight measured
generations per mode, forming eight neighboring pairs:

| Instrumented mode | Minimum | Median | Maximum decode tokens/s |
| --- | ---: | ---: | ---: |
| Shared-input source | 10.9699 | 11.3636 | 12.1311 |
| Shared input plus vector metadata/contraction | 10.5191 | 11.5864 | 12.1933 |

Seven of eight pairs improve. The paired median ratio is 1.01920, but the
geometric mean is only 1.00223 because one pair measures 0.86816. That candidate
generation has 25 decode steps with both frequency endpoints at 730 MHz and
13 with transitioning endpoints. The baseline's 704 measured steps all have
both endpoints at 1400 MHz; the candidate has 666 such steps. Every low and
transition sample is retained. Endpoint association does not establish the
frequency throughout a step or prove causation.

Frequency reads are excluded from step timers and cost a median 0.1577% of loop
time; cadence effects are not excluded. This instrumented batch has no
generation below ten, but it cannot override the ordinary 9.0205 minimum or
establish stable deployment throughput. No clocks or other applications are
changed. The modest paired result supports only a default-off source
experiment, not a default change or a claimed 4.3% causal gain.

## Default-off source integration

`EXECUTORCH_MUSA_GGUF_VECTOR_DEQUANT` is a default-OFF CMake option passed only
to the GGUF device compilation. It enables vector scale/minimum processing and
the locally scoped contraction helper. The global `-ffp-contract=off` remains
unchanged, and accumulation/reduction expressions, weights, runtime stream and
shape/alignment/capacity guards are untouched. It operates in the packed body
selected by `PACKED_SCHEDULE` or `LOCAL_REDUCTION`; without either, there is no
affected path. Shared staging is an independent option. Timings here use both
options, with the existing qualified reduction and compiler settings.

A fresh CMake build tests six configurations: default OFF, ON, OFF again, ON
again, ON with shared staging disabled, and restored ON with staging. Every
configuration passes all 18 GGUF and 15 attention tests; the single unavailable
second-device test skips. The new finite-half-scale test covers signed zeros,
minimum/maximum subnormals, minimum normals, values adjacent to one, and maximum
finite positive/negative scales and minima. Random packed groups and levels
exercise Q4/Q5 at 129 rows and K values 256, 2560, 4096, 4352, 9216, 16384 and
16640, including single/four-part and shared-capacity boundaries. Complete
results match the existing embedding-decode/local-FMA reference byte for byte,
with finite-output and guard checks. This is boundary coverage, not exhaustive
testing of every half-scale/minimum combination.

With shared input enabled, the complete OFF device image is byte-identical to
the previous qualified shared-input source; the complete ON image matches the
measured combined prototype. Repeated settings reproduce object, runtime and
device-image hashes exactly. The no-shared configuration passes correctness
tests but has no new performance claim.

The ON object is linked into an isolated source runtime, again replacing only
the GGUF device object. A fresh three-case run and independent audit verify
another 2,298 complete byte-exact vectors, 1,149 reset pairs, 18 negative
controls and all original IDs/EOS with unchanged frozen-HF bounds. The source
option is numerically qualified on these cases, not enabled in deployment.

## Source ordinary timing

A separate original-runner ABBA/BAAB series uses the same warmup, full-generation
checks and eight measured generations per mode as the prototype series:

| Source runtime | Minimum | Median | Maximum decode tokens/s | Serial-prefill median |
| --- | ---: | ---: | ---: | ---: |
| Qualified shared input | 9.1207 | 11.4150 | 12.1088 | 11.5816 |
| Shared input plus vector dequantization | 9.2093 | 9.4969 | 10.3759 | 9.1463 |

The pooled median ratio is 0.83197; ABBA and BAAB ratios are 0.82445 and 0.91811.
All eight candidate measurements are retained: 9.5177, 9.5149, 9.2478, 9.4789,
9.4621, 10.3759, 9.6454 and 9.2093. No profiler, frequency instrumentation,
launch blocking or competing agent work is active during ordinary timing.

This unfavorable source series is not replaced by the better prototype or
instrumented results. Device-image equality excludes an observed change in GPU
instructions, but does not establish host-library identity or a clock-related
cause. The 12.3928 historical ordinary prototype peak remains a peak, not a
source deployment guarantee. The source option has not demonstrated a robust
ordinary end-to-end gain and must remain default OFF.

## Decision

Combined mode now has same-process pairing, binary equivalence and a tested
default-off source option. The microbenchmarks, full precision checks and
blocking attribution support continued experiments, but the mixed ordinary
timings do not justify default enablement or a stable-throughput claim. Keep
the inconsistent attention-gate matrix and all ordinary low samples. Further
work must reduce the packed Linear cost rather than remove checks or select
only high-frequency runs.

The next [shared-input block-grouping experiment](QWEN35_GGUF_SHARED_BLOCKS.md)
tests the new staging path with larger independent row groups, rather than
repeating the old global-input block-size experiment unchanged.

## Evidence

Under ETSupport `artifacts/gguf-20261029`:

- `shared-input-dequant-probe-v1/{kernels.mu,bench.cpp,build.py,run.py}`:
  source, compiler commands and complete raw paired measurements.
- `audit_shared_input_dequant_v1.py`,
  `shared-input-dequant-probe-v1-decision.json`: independent micro-output,
  timing, guard, resource and device-image audit.
- `build_shared_input_dequant_v1.py`, `build-shared-input-dequant-v1`,
  `validate_shared_input_dequant_v1.py`, `shared-input-dequant-validation-v1`,
  `audit_shared_input_dequant_runtime_v1.py`,
  `shared-input-dequant-runtime-decision-v1.json`: isolated runtime, native
  tests, complete three-case logits and independent precision/provenance audit.
- `profile_shared_input_dequant_blocking_v1.py`,
  `shared-input-dequant-blocking-profile-v1`,
  `analyze_shared_input_dequant_blocking_v1.py`,
  `audit_shared_input_dequant_profile_v1.py`,
  `shared-input-dequant-profile-decision-v1.json`: whole-model blocking trace,
  all windows, intervals, kernel counts and independent SQL audit.
- `benchmark_shared_input_dequant_v1.py`, `shared-input-dequant-benchmark-v1`,
  `audit_shared_input_dequant_ordinary_v1.py`,
  `shared-input-dequant-ordinary-decision-v1.json`: original-runner ABBA/BAAB,
  all original IDs, retained lows and protected artifacts.
- `alternating-shared-dequant-probe-v1`,
  `build_alternating_shared_dequant_v1.py`,
  `alternating-shared-dequant-device-audit-v1`,
  `audit_alternating_shared_dequant_devices_v1.py`: generation-switching
  source and independently verified device-function/metadata identity.
- `alternating-shared-dequant-validation-v1`,
  `alternating-shared-dequant-benchmark-v1`,
  `audit_alternating_shared_dequant_v1.py`,
  `alternating-shared-dequant-decision-v1.json`: exact text outputs, dispatch
  counters, all eight pairs and read-only frequency endpoints.
- `build_shared_dequant_source_v1.py`, `build-shared-dequant-source-cmake-v1`,
  `build-shared-dequant-source-v1`, `validate_shared_dequant_source_v1.py`,
  `shared-dequant-source-validation-v1`, `audit_shared_dequant_source_v1.py`,
  `shared-dequant-source-decision-v1.json`: six source configurations, exact
  images, finite-half boundary tests and full three-case model qualification.
- `benchmark_shared_dequant_source_v1.py`, `shared-dequant-source-benchmark-v1`,
  `audit_shared_dequant_source_ordinary_v1.py`,
  `shared-dequant-source-ordinary-decision-v1.json`: retained ordinary source
  timing, including the unfavorable series and all low samples.

Stable minimum decode throughput of ten tokens/s remains unproven.
