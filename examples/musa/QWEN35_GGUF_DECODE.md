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
