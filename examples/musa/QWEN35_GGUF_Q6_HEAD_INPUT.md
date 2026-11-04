# Q6 output-head input reuse

This follows [shared-input block grouping](QWEN35_GGUF_SHARED_BLOCKS.md).
The qualified 512-thread Q4/Q5 candidate's blocking trace attributes about
15.325 ms to the unchanged Q6 output head and 12.763 ms to Q6 four-part
projections. This experiment targets only the output head. It keeps the
248320-by-2560 matrix, 128-thread/32-row head geometry, existing prepared Q6
weights, grouped FP32 dot-product order, reduction and complete vocabulary.
The Q4/Q5 path remains the qualified 512-thread shared/vector variant.

The model is still Qwen3.5-4B Q4_K_M with FP32 activation, state and accumulation.
No TF32, requantization, MTP, pruning or tolerance relaxation is used. The head
experiment is isolated and adds no production source option or default.

## Variants

| Mode | Head input access |
| --- | --- |
| 0 | Original global/read-only loads |
| 1 | Stage all 2560 inputs once per CTA in their original layout |
| 2 | Stage once per CTA with four-float vectors transposed by word/group |
| 3 | First sixteen lanes of each warp load, then broadcast to the other sixteen |

The two staging variants use 10 KiB of input storage and synchronize before
reading it. They do not change the weight loads or any FP32 FMA/reduction
expression. Every invocation reloads input on the calling stream; no persistent
input cache is introduced. Original head shape/type/stream dispatch guards
remain unchanged, as do all other kernels.

## Paired head microbenchmark

Four modes and opposite traversal orders produce eight processes, each retaining
at least three seconds of active warmup, 100 alternating timing pairs, three
complete output comparisons and 24 guard checks. Independent auditing verifies
24 complete finite 248320-element vectors byte for byte, 192 guard checks and
all 800 timing pairs. Input weights, the existing lossless repacking library,
source, harness and binaries are protected by hashes. Packing is outside the
kernel timing; no new weight layout is introduced.

The control's complete device image matches the newly built 512-thread source
image. All 800 frequency endpoint samples read 1400 MHz; no sample is removed,
and endpoints cannot establish constant frequency throughout a pair.

| Mode | Forward ratio | Reverse ratio | Forward candidate median (ms) | Reverse candidate median (ms) |
| --- | ---: | ---: | ---: | ---: |
| Original control | 0.999850 | 0.999597 | 8.92374 | 8.93284 |
| Plain shared input | 1.038970 | 1.039007 | 8.59728 | 8.59844 |
| Transposed shared input | 1.039029 | 1.038483 | 8.59626 | 8.60390 |
| Warp broadcast | 0.994762 | 0.994646 | 8.99550 | 8.97970 |

Ratios are baseline median divided by candidate median. The two shared layouts
are essentially tied here; the simpler plain layout is selected for full-model
qualification. Warp broadcast regresses slightly and is not selected. The
roughly 3.9% head-kernel improvement is not a 3.9% model-speedup claim.

Compiler temporary registers are 110/104/104/111 for modes 0/1/2/3. Reported
static shared bytes are 20/10272/10272/20, including compiler overhead; all
modes have zero local memory and zero spill loads/stores. These resource
counts do not by themselves explain the measured gain.

## Full-model qualification and profiling

Only the GGUF device object is replaced in an isolated runtime. The control
device image matches the qualified 512-thread source image and the candidate
matches mode 1. Native tests pass 18 GGUF and 15 attention cases, with the
unavailable second-device case skipped. A separate model audit verifies 2298
complete byte-exact output vectors, 1149 reset pairs, 18 negative controls and
all original IDs/EOS across text, single-image and multi-image cases. Maximum
absolute errors against the unchanged frozen HF references remain 0.0000262260,
0.0006752014 and 0.0008490086. Prepared image features and frozen references are
reused; this is not a fresh raw-image frontend or HF execution test.

The separate blocking trace contains 28 prefill and eight decode windows,
each with 788 kernels, 147 shared Q4/Q5 calls, eight fused attention calls,
twelve stream synchronizations and no asynchronous allocations/frees.
Independent SQL auditing gives these decode medians:

| Quantity | Median (ms) |
| --- | ---: |
| Sum of nonoverlapping kernels | 105.7842 |
| Packed Linear aggregate | 86.8332 |
| Q6 output head | 15.1522 |
| Q6 four-part projections | 12.7721 |
| Other kernels | 18.9342 |

The preceding 512-thread trace's head median was 15.3248 ms. These are separate
instrumented captures, not an ordinary-throughput or causal model-speedup claim.

## Ordinary-model decision

An uninstrumented three-mode benchmark retains eight measured generations per
mode and twelve full warmups in two opposite orders. Original IDs/EOS and all
low observations are independently verified. Decode minimum/median/maximum are
10.1224/12.3788/13.0030 tokens/s for the 512-thread source and
9.9777/11.2519/12.8348 for the added Q6 head staging. The 128-thread control is
9.5891/11.8339/12.4482. The Q6 candidate's first-series median is 12.4550, but
its second-series median is 10.1047; neither series nor low samples are removed.

The head-only microbenchmark gain has not established a reproducible full-model
benefit. Keep this as an isolated experiment, not a production source option or
default. Prefer the qualified selectable 512-thread source for further work.
The earlier 512-thread prototype's 9.7423 ordinary low also remains relevant;
one all-above-ten source batch does not prove stable minimum throughput.

## Evidence

Under ETSupport `artifacts/gguf-20261029`:

- `q6-head-input-reuse-probe-v1/{kernels.mu,bench.cpp,build.py,run.py}`:
  four variants, complete raw outputs, all timing pairs and protected inputs.
- `audit_q6_head_input_reuse_v1.py`, `q6-head-input-reuse-decision-v1.json`:
  independent micro-output, resource, timing and control-image audit.
- `build_q6_head_input_reuse_v1.py`, `build-q6-head-input-reuse-v1`,
  `validate_q6_head_input_reuse_v1.py`, `q6-head-input-reuse-validation-v1`,
  `audit_q6_head_input_reuse_runtime_v1.py`,
  `q6-head-input-reuse-runtime-decision-v1.json`: isolated build, native tests
  and independent three-case full-model precision/provenance qualification.
- `profile_q6_head_input_reuse_blocking_v1.py`,
  `q6-head-input-reuse-blocking-profile-v1`,
  `analyze_q6_head_input_reuse_blocking_v1.py`,
  `audit_q6_head_input_reuse_profile_v1.py`,
  `q6-head-input-reuse-profile-decision-v1.json`: fresh blocking trace and
  independent full-window kernel/interval audit.
- `benchmark_source_head_staging_v1.py`, `source-head-staging-benchmark-v1`,
  `audit_source_head_staging_ordinary_v1.py`,
  `source-head-staging-ordinary-decision-v1.json`: all ordinary timings,
  warmups, protected input/runtime hashes and original token trajectories.

Stable minimum model decode throughput of ten tokens/s remains unproven.
