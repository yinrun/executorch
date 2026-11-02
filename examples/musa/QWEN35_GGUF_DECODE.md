# Qwen3.5 GGUF decode optimization

The 2026-10-31 grouped-attention milestone is described below. Earlier measurements
remain scoped to their original model packages and test runs.

The first optimization targets the packed Q4_K/Q5_K/Q6_K linear kernels for
single-token FP32 decode. It does not change the raw quantized weights, shared
Q6_K output head, activations, recurrent state, model package, or attention path.
The generic batched implementation remains available.

## Reduction order

Each 128-thread block now computes four output rows. Each group of 32 threads
keeps four independent accumulators for the original lanes `lane`, `lane + 32`,
`lane + 64`, and `lane + 96`. Processing each 256-element quantization block
preserves the original sequence of fused multiply-add operations for each
accumulator. Combining `(partial[0] + partial[2]) + (partial[1] + partial[3])`
preserves the original first two reduction stages. The remaining stages use
width-32 shuffles in the same order, without shared-memory barriers.

The implementation uses explicit width-32 subgroups rather than assuming that a
reported hardware warp size is the supported shuffle width. Partial row tiles
and byte-offset packed storage are covered by native tests. Column sizes beyond
the signed 32-bit indexing range retain the generic implementation.

## Validation on M1000

The 2026-10-30 validation used unchanged Qwen3.5-4B Q4_K_M PTE/PTD files and the
original runner/extension, replacing only `libaoti_cuda_shims.so`:

- Six native tests passed, including 90 new dtype/width/row-count combinations.
- Eleven real quantized matrix geometries, including the full output head,
  produced byte-identical outputs for three activation vectors each.
- Original text, single-image, and multi-image prepared requests produced
  232, 856, and 1210 byte-identical full-logit outputs, respectively. This covers
  1149 reset pairs and matching autonomous generated tokens/EOS.
- The existing frozen native results were rehashed and compared; HF was not
  rerun. Existing HF batch-comparison failures are not relabeled as passing.
- MP21 compilation also succeeded; runtime testing was on MP22/M1000 only.

The image cases reuse previously validated vision features. This is a text
decoder regression, not a new raw-image or vision benchmark.

## Performance scope

A normal, nonblocking A/B/B/A measurement used the original 28-token text prompt
and 17 generated tokens (16 decode steps). Each process had one warmup generation
and two measured generations. No profiler, logits dump, or launch blocking was
enabled. Inputs, model package, runner, extension, and output validation were
identical; only the shim library changed.

| Runtime | Minimum decode tok/s | Median decode tok/s |
| --- | ---: | ---: |
| Original packed kernels | 1.084 | 1.089 |
| Order-preserving subgroup kernels | 2.315 | 2.336 |

The median improvement is approximately 2.15x on this shared, unlocked desktop
GPU. These are short-sequence measurements, not replacements for full-case
throughput results. The 10 tok/s target is **not achieved**. The original frozen
deployment package is unchanged; the optimized library is a validated candidate.

Larger per-subgroup row tiles were also tested and were slower than this
candidate; they are not part of the implementation.

## Reproduction and next work

Build and run the focused native suite in an independently configured MUSA build:

```sh
cmake --build "$BUILD_DIR" --target executor_runner test_musa_gguf -j8
LD_LIBRARY_PATH="$BUILD_DIR/backends/cuda:$BUILD_DIR/extension/cuda:$MUSA_HOME/lib" \
  "$BUILD_DIR/backends/cuda/test_musa_gguf"
```

Local validation artifacts are grouped under `artifacts/gguf-20261029` in the
ETSupport workspace: `build-decode-gemv-v1`, `decode-gemv-validation-v1`,
`decode-gemv-validation-audit-v1.json`, and `decode-gemv-benchmark-v1`.
Their scripts and raw outputs retain the commands, input paths, and checks.

The next priority after the attention milestone is further packed GEMV work.
Capacity remains 4096. Any
lossless weight preprocessing must disclose its extra memory cost; converting
activations or the output head to lower precision is not an equivalent
optimization. Future acceptance must compare complete logits against the
matched reference and measure normal full-case decode, not just isolated kernels.

## Grouped valid-prefix attention (2026-10-31)

`load_gguf_text(..., valid_length_attention=True)` opts into FP32 attention with
Q `[1,16,1,256]` and KV `[1,4,capacity,256]`, for capacities 1 through 4096.
The default loader path is unchanged. Q is interpreted as four groups of four
heads, sharing the corresponding KV head without materializing repeated KV.
Only the prefix ending at the supplied position is read. Position and the exact
causal mask are checked together with one host synchronization per attention.
TF32 remains disabled. The existing BF16 geometry and operation order remain
supported.

This requires a newly exported PTE/PTD and the updated shim. Replacing only the
shim in an old package does not enable the new graph. The candidate has eight
opaque attention calls, one delegate and the same 64 mutable state buffers;
its exported graph has 4630 nodes rather than 4710. All 426 original weight
payloads were rehashed in the new PTD, with the shared Q6_K head stored once.

Validation on MP22/M1000 passed 11 native attention tests (the two-device test
was skipped on this single-device system), six GGUF tests, and 11 Python
metadata/eager/export/runner tests. FP32 coverage includes grouped head mapping,
nonzero scores, valid length 4096, poisoned future KV, offsets and invalid
controls/storage/aliases. BF16 regression and planned/unplanned memory passed.

The original text, single-image and multi-image prepared cases produced 232,
856 and 1210 full-logit outputs, respectively. Every element passed the existing
`atol=rtol=0.001` comparison against rehashed, frozen matched-HF-step references;
maximum absolute errors were 0.000030041, 0.000896454 and 0.000934601.
All argmax values, autonomous generated tokens, EOS and step counts matched.
The 1149 reset pairs were byte-identical. Outputs are not claimed byte-identical
to the old attention implementation. HF and vision were not rerun, historical
HF batch failures remain failures, and the new model's full 4096-step trajectory
and relocation were not revalidated in this milestone.

A new normal A/B/B/A test used the same 28-token prompt and 17 generated tokens,
with one warmup and two measured runs per process, no profiler/blocking/dumps,
and unchanged full output checking:

| Runtime | Minimum decode tok/s | Median decode tok/s | Maximum decode tok/s |
| --- | ---: | ---: | ---: |
| Previous GEMV candidate, old attention | 2.710 | 2.759 | 2.774 |
| GEMV plus grouped valid-prefix attention | 3.975 | 4.015 | 4.132 |

The paired median improvement is 1.455x. The older 2.336 measurement is not this
run's baseline: the desktop GPU is shared and clocks are not locked. This is
still a short-sequence result, and the 10 tok/s target is **not achieved**.
The original frozen deployment package is unchanged.

A separate blocking profile after measurement attributes 93.27% of kernel time
to packed Linear, including 17.02% for the full Q6_K head. These profiler times
are not normal throughput. The remaining priorities are the Q4_K FFN matrices
and Q6_K head, not further attention-only tuning.

New local artifacts under `artifacts/gguf-20261029` are
`build-decode-attention-v1`, `decode-attention-export-v1`,
`decode-attention-validation-v1`, `decode-attention-benchmark-v1`,
`decode-attention-final-audit-v1.json` and
`decode-attention-profile-blocking-v1`. Their adjacent build/export/validation/
benchmark/audit/profile scripts retain the exact commands and scope.

## Vectorized packed decode (2026-10-31)

The next candidate vectorizes the existing `gguf_linear` MUSA implementation;
it does not change the operator or model mathematics. Contiguous input groups
use `float4` loads. Q4/Q5 use 32-bit or 128-bit packed reads and byte-lane
bit-plane decoding, reusing scale/minimum calculations. Q6 uses paired 16-bit
reads because its 210-byte blocks are not necessarily 32-bit aligned.

Q4/Q5 use width 16 for at least 128 rows with 16-byte-aligned packed storage,
otherwise width 4; Q6 uses width 4. Each thread holds consecutive original
reduction lanes. Subgroup shuffles followed by component reduction preserve
the original 128-lane addition tree and per-lane FP32 FMA order. Misaligned
input/storage retains the scalar decode fallback; batched input retains the
generic path. No weight preprocessing, persistent cache, extra weight copy,
quantization change, TF32, or reduced-precision activation is introduced.

Validation on MP22/M1000 passed eight native GGUF tests, including 1620
dtype/shape/offset/bit-plane combinations with bitwise output comparisons and
a nondefault-stream partial-row-tile test. Eleven attention tests passed;
the two-device test remained skipped. MP21 compilation passed, but MP21
runtime behavior was not tested. Eleven real matrix geometries with three
inputs each produced 863328 byte-identical output values. Large Q4 kernels
were approximately 2x faster and the full Q6 head was 1.585x faster in this
microbenchmark; these are not model throughput measurements.

The three original prepared cases produced 2298 full-logit outputs that were
byte-identical to the preceding grouped-attention candidate, including all
1149 reset pairs. Autonomous tokens, EOS and step counts matched. Rehashed
frozen matched-HF-step references still passed the unchanged elementwise
`atol=rtol=0.001` gate, with zero outliers or argmax mismatches. HF and vision
were not rerun; historical HF batch failures remain failures. The candidate's
full 4096-step model trajectory and relocation remain unvalidated.

A fresh normal A/B/B/A test used the same grouped-attention PTE/PTD on both
sides, changing only the shim. The 28-token prompt, 17 generated tokens,
16 decode steps, one warmup and two measured runs per process, and complete
output checking were unchanged. Profiling, launch blocking and logits dumps
were disabled:

| Runtime | Minimum decode tok/s | Median decode tok/s | Maximum decode tok/s |
| --- | ---: | ---: | ---: |
| Grouped attention, scalar packed decode | 4.074 | 4.113 | 4.117 |
| Grouped attention, vectorized packed decode | 7.042 | 7.250 | 7.262 |

The paired median gain is 1.763x on the shared, unlocked desktop GPU. This
short-sequence result does not establish full-case or long-context throughput.
The 10 tok/s target is **not achieved**. A separate blocking profile attributes
87.53% of kernel time to packed Linear, including 17.20% for the full Q6 head;
those times are not normal token latency. Large Q4 matrices and the Q6 head
remain the next performance priorities. Whole-input shared-memory staging and
wider Q6 variants were slower and are not part of the implementation.

Only the shim needs replacement relative to the grouped-attention candidate;
that candidate's PTE/PTD is reused unchanged. The original frozen package is
not overwritten. Local artifacts under `artifacts/gguf-20261029` include
`build-decode-vector-v1`, `gemv-vector-v1`, `decode-vector-validation-v1`,
`decode-vector-benchmark-v1`, `decode-vector-final-audit-v1.json` and
`decode-vector-profile-blocking-v1`, with adjacent reproduction scripts.

## Read-only Q4 loads and tiled Q6 heads (2026-11-01/02)

Q4 vector loads now use the read-only cache hint. A separate Q6 kernel reuses
each input group across two rows for aligned single-batch matrices with 2560
columns and at least 4096 rows, including the full output head. Ordinary Q6
and Q5 retain their original vector kernel, with existing alignment/batch
fallbacks. FP32 FMA and reduction order, quantization, PTE/PTD, and weight
storage are unchanged. No persistent weight cache or extra weight copy is
introduced. Only the shim changes relative to the grouped-attention package.

Nine GGUF native tests passed, retaining 1620 offset/bit-plane combinations
and adding 18 Q6 dispatch/tail/sentinel cases. The calling-stream test also
checks input and weight updates after an earlier invocation. Eleven attention
tests passed; the two-device test remained skipped. MP21 compilation passed;
runtime testing remains MP22/M1000 only. Eleven real matrix geometries produced
863328 byte-identical output values. All 2298 full-logit outputs and 1149 reset
pairs matched the preceding vector candidate byte for byte, with unchanged
matched-HF-step tolerance, tokens and EOS. HF, vision, full 4096-step model
execution and relocation were not rerun.

An initial generic row-tile implementation regressed ordinary Q6 down kernels
by approximately 7% and was superseded. The accepted implementation preserves
that kernel and specializes only the large Q6 path. Larger blocks, wider
vectors and unrolled loops were generally not beneficial. Q5's final isolated
microbenchmark was 0.953x; not every geometry is claimed to improve.

Normal A/B/B/A tests retain full output checking, with one warmup and two
measured generations per process, no profiling/blocking/logits dumps:

| Workload and runtime | Minimum tok/s | Median tok/s | Maximum tok/s |
| --- | ---: | ---: | ---: |
| 16 decode steps, previous vector | 6.002 | 6.651 | 7.341 |
| 16 decode steps, cache/tile | 7.336 | 7.633 | 7.768 |
| Original text to EOS, previous vector | 6.754 | 6.846 | 6.910 |
| Original text to EOS, cache/tile | 7.155 | 7.206 | 7.224 |

The full text uses the original 28-token prompt and generates 89 tokens to EOS,
with 88 decode steps. All 12 trajectories match. Its paired median gain is
1.053x, not the noisier short test's 1.148x. The continuous full-text series
took 250.044 seconds and records realtime/monotonic/boottime boundaries.
An earlier full-text series spanned more than 15 hours and was excluded from
continuous performance acceptance; its slower candidate results and timestamp
evidence are preserved separately, not overwritten. The desktop GPU remains
shared and unlocked. The 10 tok/s target is **not achieved**.

A separate blocking profile attributes 86.72% of kernel time to packed Linear,
including 17.34% for the full Q6 head. These are not normal token latencies.
Further work should evaluate lossless layout/metadata preprocessing with
explicit lifetime and extra-memory accounting, and retain full-logit checks.

Local artifacts include `gemv-schedule-v1`, `build-decode-cache-v2`,
`decode-cache-validation-v2`, `decode-cache-benchmark-v2`,
`decode-cache-fulltext-benchmark-v2`, `decode-cache-final-audit-v2.json`,
`decode-cache-acceptance-v2.json` and `decode-cache-profile-blocking-v2`.
The original package is unchanged; adjacent scripts retain reproduction commands.
