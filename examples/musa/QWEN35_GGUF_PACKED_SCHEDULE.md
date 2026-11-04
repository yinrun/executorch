# Packed GEMV preloading and scheduling investigation

## Scope

This follows the rejected [dense FP32 cache probe](QWEN35_GGUF_DENSE_FP32.md).
Keep the original Qwen3.5-4B Q4_K_M weights, FP32 activations/state and the
qualified caller-owned Split-K export. Investigate scheduling within the
compressed Q4/Q5 GEMV instead of increasing weight traffic. No TF32,
requantization, MTP, vocabulary pruning or relaxed numerical gate is used.

The preloading kernels remain isolated prototypes. The no-preload scheduling
candidate now has an opt-in source implementation described below. Default
dispatch, accepted export and runtime are not replaced. The 10 tokens/s
goal remains unmet; microbenchmark gains are not whole-model gains.

## Software preloading

The one-block-ahead variants retain the same block order, FMA order,
subgroup reduction, final component tree and four-part combination order.
They test metadata plus packed payload preloading, additionally preloading
activations, and metadata-only preloading. A no-preload control uses the
same packet representation and quad accumulators, fetching inputs closer
to their consumption. Every variant retains the original compressed layout.

Eight real matrices include Q5 QKV, Q4 attention projections, SSM output,
FFN down, FFN gate and the actual 18432-by-2560 merged gate/up projection.
The initial sweep and reverse-order follow-up pass **144 complete,
byte-exact outputs** over three changing inputs per matrix/variant.
Native event timings alternate reference and candidate, discarding two
warmup rounds and retaining 100 samples per side. The desktop GPU is shared.
Long-K micro controls and candidates both include a separate reduction;
the model's caller-owned partial-output path does not.

| Variant | Q4 temp registers | Q5 temp registers | Micro result |
| --- | ---: | ---: | --- |
| No-preload packet/quad control | 111 | 113 | Small shape-dependent gains |
| Next metadata and packed payload | 135 | 145 | Mostly slower |
| Next metadata, payload and inputs | 199 | 209 | Slower on every tested matrix |
| Next metadata only | 119 | 123 | No consistent gain |

All compiled variants report zero local memory and zero spill loads/stores.
These resource reports do not establish measured occupancy or prove a
particular stall mechanism. Preloading is rejected, not promoted based on
isolated near-parity results. The installed SDK's one-argument prefetch-only
helper is guarded for architectures 310/320, not this MP22 target; its
existence in the headers is not evidence of usable MP22 prefetch-only support.

The no-preload control warrants a whole-model probe: forward/reverse
native/candidate speed ratios are 1.0336/1.0506 for SSM output,
1.0287/1.0252 for Q4 FFN down, 1.0346/1.0264 for attention Q and
1.0349/1.0214 for merged gate/up. Other matrices are closer to parity and
some regress slightly. These are two micro sweeps, not a deployment claim.

## Isolated model qualification

The candidate replaces only the kernel object in the qualified Split-K
runtime link. The host objects, runner, extension and PTE/PTD stay unchanged.
Only width-16 Q4/Q5 dispatch and Q4 four-part dispatch use the no-preload
packet/quad kernel. Width-4 fallbacks, Q6 projections, the grouped output
head and the caller-owned partial-output ABI remain unchanged.

All **16 native GGUF tests pass**, including offsets, tails, bit planes,
calling streams, invalid inputs and caller-owned partials. The first attempt
used the previous CMake directory's OFF-configured test executable against
an ON runtime, producing three expected contract mismatches. Its log is
retained. Recompiling that same test source with the matching ON definition
in an isolated directory passes the full suite; no test is disabled or
acceptance condition loosened.

Text, single-image and multi-image trajectories, each run twice with reset,
pass **2298 complete logits**, 1149 reset pairs and 18 negative controls.
Every output byte matches the qualified caller-owned Split-K result.
The unchanged frozen-HF gate, finite checks, exact argmax and original
IDs/EOS also pass. Maximum HF errors remain 0.0000340939, 0.0007286072 and
0.0007343292 respectively. HF and the raw-image frontend are frozen
references, not rerun. This does not establish a 4096-token trajectory or
concurrent-service correctness.

## Whole-model timing

Ordinary ABBA and BAAB comparisons use the qualified Split-K runtime as
the control, the same PTE/PTD, original 28-token prompt and 88 decode steps,
full output checks, one warmup plus two measured generations per process,
and no profiler, launch blocking or logit dumping. Compilation and heavy
auditing run outside timing. The original generated IDs/EOS match in all
runs, and runtime maps verify the intended native libraries without
libtorch/libpython. No desktop application is stopped or clock locked.

| Series | Baseline median | Candidate median | Median change |
| --- | ---: | ---: | ---: |
| ABBA | 7.5548 | 7.5960 | +0.5445% |
| BAAB | 7.5123 | 7.6346 | +1.6274% |
| Combined, eight samples each | 7.5310 | 7.6239 | +1.2342% |

Units are decode tokens/s. Combined minima are 7.4653 and 7.5777; maxima
are 7.8515 and 7.6490. The high baseline sample is retained. These are
small measured improvements in both orders on a shared GPU, not a claim
of exclusive-GPU throughput, statistical significance or 10 tokens/s.
Serial-prefill medians are mixed between orders and are not claimed as
an improvement. Do not multiply this ratio by historical measurements.

The isolated candidate runtime SHA-256 is
`4e42c7f25a23f2690b231028f70021b675cc964b5401fbd145dc050310c05d9e`.
Independent auditing rehashes all 2298 outputs against recorded qualified
Split-K hashes, build inputs, protected libraries and ordinary-run artifacts.
This table measures the isolated prototype, not the subsequent source build.

## Separate global trace

A fresh nonblocking trace covers loading, serial prefill, a full warmup and
the original measured decode trajectory. It confirms 812 kernel labels per
step, including 147 packet/quad kernels: 75 single-part Q4, 48 four-part Q4
and 24 Q5. Twelve stream synchronizations, three event records and eight
async allocations remain. There is no standalone caller-owned-part reduction.

Measured decode host-window median is 141.883 ms; compute-track interval
union is 117.618 ms, compute-plus-copy union 118.481 ms, and post-execution
validation gap 1.317 ms. These profiled intervals are not ordinary token
latencies. Async labels share intervals; do not sum their durations or add
CPU waits to GPU intervals. An exceptional Pixel event warning is retained;
the selected model compute intervals pass the analyzer's monotonic checks.
The trace verifies dispatch, not a causal per-kernel speedup attribution.

## Opt-in source implementation

For a configured MUSA build, enable:

```sh
cmake -S . -B <build> -DEXECUTORCH_MUSA_GGUF_PACKED_SCHEDULE=ON
cmake --build <build> --target aoti_cuda_shims test_musa_gguf test_musa_causal_attention -j8
```

The option defaults to OFF. It selects `linear_packed_quad_decode_kernel`
only at the existing aligned width-16 Q4/Q5 dispatch and, when Split-K is
enabled, the existing Q4 partial-output dispatch. It changes no host shim,
operator schema, output ownership or shape/type/alignment validation.
The Q6 path, grouped head and width-4/generic fallbacks are unchanged.
The source implementation removes the rejected preloading modes rather
than carrying runtime policy branches or unused activation buffers.

No re-export is required for this kernel-only option. The benchmarked
configuration additionally retains the existing MAX_ILP, QUAD_ARITHMETIC,
GROUPED_Q6_HEAD and SPLIT_K build options and the caller-owned Split-K
PTE/PTD. This scheduling option does not itself enable those other options.

A fresh CMake build passes both GGUF and attention suites in five states:
default OFF, ON, OFF again, ON with QUAD_ARITHMETIC/SPLIT_K OFF, and ON again
with those two options restored. The repeated ON objects are identical,
as are the repeated OFF objects; ON differs from OFF. The native regression
matrix now also covers 4352 columns and a 143-row, fifteen-row tail, including
the existing offset/alignment and packed-bit-plane checks. The attention
suite retains its existing second-device skip on this single-device system.

The source kernel object is also linked into the protected host runtime
for controlled full-model validation and timing. This is not a throughput
measurement of every newly compiled CMake host object. Full-model source
qualification and the fresh source-build benchmark are in progress; the
prototype numbers above must not be substituted for those results.

## Local evidence

ETSupport artifacts under `artifacts/gguf-20261029`:

- `pipelined-packed-probe-v1`, `pipelined-metadata-probe-v1`: sources,
  native harness, resource logs, forward and reverse micro sweeps. The
  initial pre-launch matrix-selection assertion failure is retained.
- `audit_pipelined_micro_v1.py`, `pipelined-micro-decision-v1.json`:
  independent source/library rehashes and all 144 byte-exact checks.
- `packed-reorder-probe-v1`, `build_packed_reorder_v1.py`,
  `build-packed-reorder-v1`: isolated candidate source, build and link.
- `test_packed_reorder_v1.py`, `packed-reorder-native-tests-v1.log`,
  `packed-reorder-native-tests-on-v1.log`: retained configuration mismatch
  and corrected full native-suite execution.
- `validate_packed_reorder_v1.py`, `packed-reorder-validation-v1`:
  complete original trajectories, logit files and model gate.
- `benchmark_packed_reorder_v1.py`, `benchmark_packed_reorder_v2.py`:
  ordinary paired comparison against the qualified Split-K source runtime.
- `packed-reorder-benchmark-v1/v2`, `audit_packed_reorder_v1.py`,
  `packed-reorder-decision-v1.json`: complete ordinary timing and independent
  model/build/output audit.
- `profile_global_packed_reorder_v1.py`, `analyze_global_packed_reorder_v1.py`,
  `global-packed-reorder-profile-v1`: separate global capture and
  interval-union analysis, including original IDs and runtime-map checks.
- `build_packed_schedule_source_v1.py`, `build-packed-schedule-source-cmake-v1`,
  `build-packed-schedule-source-v1`: integrated kernel source, five CMake
  configurations and controlled kernel-object replacement in the runtime.
