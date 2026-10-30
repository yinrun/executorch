# Qwen3.5 GGUF decode optimization

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

The next priorities are FP32 valid-prefix attention without materialized GQA
expansion, followed by further packed GEMV work. Capacity remains 4096. Any
lossless weight preprocessing must disclose its extra memory cost; converting
activations or the output head to lower precision is not an equivalent
optimization. Future acceptance must compare complete logits against the
matched reference and measure normal full-case decode, not just isolated kernels.
