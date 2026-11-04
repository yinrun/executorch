# Shared-input GEMV dequantization follow-up

This follows the [shared-input experiment](QWEN35_GGUF_INPUT_REUSE.md).
The guarded source blocking trace still attributes 99.763 of 118.678 ms of
median kernel time to packed Linear. The experiment changes instruction
expression within the existing Q4/Q5 packed GEMV body, not weights, layout,
accumulation order or shared-memory capacity. Q6 kernels are unchanged.

The workload remains Qwen3.5-4B Q4_K_M on M1000 MP22 with FP32 activation,
state and accumulation. No TF32, requantization, MTP, vocabulary pruning,
tolerance relaxation or output-check removal is used. These are isolated
artifacts, not a new production option or default.

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

## Decision

Combined mode warrants source-based same-process pairing and binary-equivalence
checks before introducing another opt-in production flag. The microbenchmarks,
full precision checks and blocking attribution support further investigation,
but do not justify default enablement or a stable-throughput claim. Keep the
inconsistent attention-gate matrix and all ordinary low samples. Further work
must reduce the packed Linear cost rather than remove checks or select only
high-frequency runs.

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

Stable minimum decode throughput of ten tokens/s remains unproven.
