# Wider GDN fusion investigation

## Decision

The wider Triton GDN prototype passes the unchanged full-model accuracy gate,
but does **not** establish a reliable throughput gain. Keep the accepted
grouped-Q6-head deployment and the production loader unchanged. The prototype
is an artifact-only experiment, not a new enabled backend feature. The
10 tokens/s target remains unmet.

The model is Qwen3.5-4B **Q4_K_M**, with FP32 activations, recurrent state and
accumulation. This is not a BF16-weight experiment. Existing lossless Q6
preprocessing, merged FFN and valid-prefix attention stay enabled. TF32,
MTP, requantization, vocabulary pruning and shared-prefix attention stay off.

This follows the core-only experiment in
[QWEN35_GGUF_GDN_RECURRENCE.md](QWEN35_GGUF_GDN_RECURRENCE.md).

## Fusion boundary and correctness

The experimental `triton_op` includes query/key normalization and head repeat,
beta sigmoid, recurrent state update, output RMSNorm, norm weight and SiLU
gating. Convolution, projections, muDNN softplus/exp, state reset and explicit
state copy remain outside. It uses the existing accepted native runtime;
there is no new shim ABI and no libtorch/libpython dependency in the runner.

The reference CUDA DeltaNet kernel is not a drop-in replacement. This GGUF
path repeats key heads with `head % key_heads`, uses
`rsqrt(sum(square) + 1e-6)` for query/key normalization, and supplies the
model metadata epsilon for output RMSNorm. FP32 contraction remains disabled.
The recurrence retains the separately verified memory/output reduction trees.

A generic Triton norm reduction initially passes local tolerance tests but
fails exact argmax at multi-image prefill position 135: 440 rather than 12391.
No argmax waiver or position-specific fallback is introduced. Inspection and
1024 native muDNN sample rows identify the required normalization order:
serial addition of each contiguous group of four, followed by a descending
balanced reduction over 32 partials. Both sum and mean match byte for byte.

Fully spelling out that second tree with nested reshape/split operations
causes a compiler-layout regression. The backend reports 244 registers and
2374 spills; LLVM IR expands from 1157 lines in the generic prototype to 9354.
A diagnostic full-model run is stopped rather than completing a slow gate.
Its corrected position 135 alone is not full numerical acceptance.

Keeping only the serial four-element groups explicit and using `tl.sum` for
the 32 partials avoids the spill regression: 107 registers, zero reported
spills and 1625 LLVM lines. The corresponding Python-dispatched event
microbenchmark falls from 15.7112 to 0.5092 ms. These numbers include Python
launch overhead and are **not native whole-model throughput**. A proposed
`tl.gather` implementation is unavailable in the installed Triton API.

The corrected prototype passes 96 recurrent GPU steps across 16/32 heads and
zero/small/normal input magnitudes, including complete output/state checks
and preservation of input state. A standalone stateful AOTI PTE passes three
native executions. Its integration fixture clones the returned state to avoid
unsupported mutation/user-output aliasing; the model itself does not expose
state as a user output.

## Full-model qualification

The fresh export has 2934 graph nodes, 24 wide GDN calls, 32 merged FFN calls,
eight valid-prefix attention calls, one delegate and the same 64 mutated
buffers. All six eager export probes pass. The artifact loader uses an
explicit temporary subclass patch; the repository loader is not modified.

Text, single-image and multi-image tests pass all **2298 complete
248320-float logits**, 1149 byte-exact reset pairs, original generated IDs/EOS,
per-step exact argmax and 18 negative controls. The HF tolerance remains
`0.001 + 0.001 * abs(reference)` with finite checks. Independent rehashing
finds all 2298 outputs **byte-identical to the accepted grouped-head runtime**.
They are not byte-identical to the older predecoded baseline, reflecting the
already accepted grouped-head rounding change. Maximum HF errors are unchanged:
0.0000226498, 0.000739098 and 0.000905991. HF and raw-image feature generation
are frozen references, not rerun in this experiment.

## Ordinary timing and global profiling

Two ordinary ABBA/BAAB series each use one complete warmup and two measured
generations per process. The original 28-token prompt, 89 generated IDs/EOS
and 88 decode steps remain unchanged. Timing has no profiler, launch blocking,
logit dumping, compilation, heavy audit or other investigation GPU job.
Full readback/finite/argmax checks stay enabled. The desktop GPU is shared,
clocks are unlocked and no application is terminated.

| Series | Baseline median | Candidate median | Median change |
| --- | ---: | ---: | ---: |
| ABBA | 7.3242 | 7.3819 | +0.7873% |
| BAAB | 7.3553 | 7.3418 | -0.1839% |
| Combined, eight measurements each | 7.3478 | 7.3643 | +0.2250% |

Units are decode tokens/s. Combined minima are 7.2625 and 7.0064 respectively;
the low candidate sample and high baseline sample are retained. Opposite
series directions and a small aggregate difference do not justify promotion.
The historical accepted deployment measurement remains 7.3423 median tokens/s
in its own earlier series, not the control of this new comparison.

A separate corrected-candidate global profile verifies **692 kernel labels
per decode step versus 812** in the accepted graph: 120 fewer, including
24 wide kernels. Twelve stream synchronizations remain. The measured decode
host-window median is 142.062 ms, compute-track union 121.233 ms, copy-label
union 0.997 ms and post-execution validation gap 1.191 ms. These profiled
intervals are not ordinary timing. Shared asynchronous labels make summed
kernel durations unreliable; do not add CPU waits to GPU intervals or infer
causal per-kernel savings from this trace. Fewer launches alone do not improve
end-to-end throughput.

## Lossless Q4/Q5 predecode follow-up

A separate probe combines the earlier full-byte level predecode layout with
the newer four-component byte conversion/FMA implementation and MP22 ILP
scheduling. It preserves arithmetic order and checks three complete outputs
for each of eight real matrices across six width/tile variants: **144
byte-exact comparisons** pass against the currently accepted runtime.

The 320-byte block expands Q4 traffic from 144 bytes and Q5 from 176 bytes.
All seven non-tiny matrices regress in every tested variant: their best
baseline/candidate speed ratios range from 0.7180 to 0.9043. Only the tiny
32-row alpha projection has a single 1.0338 ratio. This does not establish a
whole-model opportunity; no model is reexported with this representation.
Do not generalize that rejection to all possible compressed predecode layouts.

The [layout and graph replay follow-up](QWEN35_GGUF_LAYOUT_FOLLOWUP.md)
tests lower-growth metadata formats, unchanged-size interleaving, paired-row
input reuse and SDK graph replay. None establishes a new deployment gain.

## Reproduction and evidence

Local ETSupport evidence under `artifacts/gguf-20261029`:

- `gdn-wide-triton-probe-v1`: generic-norm kernel, recurrent probes and native
  PTE; `decode-gdn-wide-export-v2`, `gdn-wide-validation-v1` and
  `global-gdn-wide-profile-v1`: export, exact-argmax rejection and first trace.
- `gdn-norm-order-probe-v1`: native muDNN bridge, reduction-order search,
  frozen sample inputs and exact native outputs.
- `gdn-wide-triton-probe-v2`, `decode-gdn-wide-export-v3` and
  `gdn-wide-validation-v2`: spill-heavy explicit-tree experiment and its
  deliberately aborted full-model run; not a completed accuracy gate.
- `gdn-wide-triton-probe-v3`: compatible corrected kernel, exact norm checks,
  96-step probe, native stateful PTE and compiler/microbenchmark evidence.
  Two/eight-warp timings are exploratory only, not model-qualified variants.
- `export_gdn_wide_v4.py`, `decode-gdn-wide-export-v4`,
  `gdn_wide_common_v3.py` and `validate_gdn_wide_v3.py`: corrected full export
  and strict validation; results in `gdn-wide-validation-v3`.
- `benchmark_gdn_wide_v4.py`, `benchmark_gdn_wide_v5.py`,
  `gdn-wide-benchmark-v4` and `gdn-wide-benchmark-v5`: ordinary ABBA/BAAB.
- `profile_global_gdn_wide_v3.py`, `analyze_global_gdn_wide_v3.py` and
  `global-gdn-wide-profile-v3`: corrected candidate trace and analysis.
- `level-predecode-quad-probe-v1`: six compiled prototypes, source, alternating
  native microbenchmarks and full-output checks; no deployment replacement.
- `audit_gdn_wide_v3.py` and `gdn-wide-decision-v3.json`: independent output
  identity, provenance, profile-count and timing audit. No accepted PTE/PTD,
  native library or default execution path is replaced.
