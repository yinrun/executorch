# Native Q4/Q5 byte conversion

`EXECUTORCH_MUSA_GGUF_BYTE_CONVERT=ON` enables an optional MUSA-only byte-vector
conversion expression in the existing Q4/Q5 vector GEMV. It does not add a
new operator, change the packed weights, alter scale arithmetic or reorder
the explicit FP32 FMA/reduction tree. Q6 and its predecoded output head are
unchanged. The option defaults to **OFF**; CUDA/ROCm are unaffected.

The subsequent optional
[four-component arithmetic experiment](QWEN35_GGUF_QUAD_ARITHMETIC.md)
also optimizes predecoded Q6 and preserves this implementation as a fallback.

The tested combination uses MUSA SDK 5.1, MP22 and
`EXECUTORCH_MUSA_GGUF_MAX_ILP=ON`. Other SDKs, architectures and scheduling
combinations are not performance-qualified. The compiler builtins are inside
the optional preprocessor branch, so disabling the option retains the original
implementation without requiring those builtins.

For an already configured MUSA build:

```sh
cmake -S . -B <build> \
  -DEXECUTORCH_MUSA_GGUF_MAX_ILP=ON \
  -DEXECUTORCH_MUSA_GGUF_BYTE_CONVERT=ON
cmake --build <build> --target aoti_cuda_shims test_musa_gguf test_musa_causal_attention -j8
```

## Why this expression

Unsigned bytes are exactly representable in FP32. Expressing the conversion
as an unsigned four-byte vector exposes byte conversion to this compiler,
instead of independently extracting a byte into a signed integer and then
converting it. Post-register-allocation machine IR for the wide Q4 K-block
loop replaces 24 shift/AND pairs with 24 combined bit operations. Excluding
`IMPLICIT_DEF` pseudo-operations, the static loop count falls from 239 to 215.
There are still 32 conversion operations; this is not a four-results-per-
instruction claim or final binary disassembly. See
[QWEN35_GGUF_COUNTERS.md](QWEN35_GGUF_COUNTERS.md) for hardware counters,
rejected alternatives and interpretation limits.

## Normal whole-model timing

The model is **Qwen3.5-4B Q4_K_M**, not BF16, with FP32 activations and
accumulation. Both variants use the same FFN-merged PTE/PTD, lossless Q6
predecode, protected native runner and bundle extension. Neither GDN merging,
shared-prefix attention, TF32, MTP, vocabulary pruning nor requantization is
enabled. The baseline already includes ILP scheduling.

ABBA and BAAB series each contain four processes, each with one complete
warmup and two measured generations. All original 28 prompt tokens, 88 decode
steps, 89 generated IDs and EOS are retained. Timed runs have no profiler,
launch blocking or logit dump; complete readback, finite checks and argmax
remain enabled. Runtime maps verify the selected shim and no libtorch or
libpython. Builds, other investigation GPU jobs and heavy audits run outside
the timed series.

| Variant | Decode minimum | Decode median | Decode maximum | Serial prefill median |
| --- | ---: | ---: | ---: | ---: |
| ILP baseline | 6.9380 | 7.0136 | 7.2034 | 7.0055 |
| ILP + Q4/Q5 byte conversion | 7.0543 | 7.0926 | 7.5399 | 7.0874 |

Units are tokens/s, with eight measured runs per variant. Pooled decode
median improves **1.13%**; independent series improve **1.16%** and **1.14%**.
GPU access is shared and clocks are not locked. All samples, including
transient high readings, remain in the table; the maximum is not sustained
throughput. Prefill is still the serial-token path, not batched prefill.
The separate earlier ILP result of 7.030 tokens/s is not the paired baseline
for this comparison. **The 10 tokens/s target remains unmet.**

## Numerical and build boundaries

The isolated candidate relinks only the GGUF kernel object into the accepted
historical-host shim. Eleven native GGUF tests pass with the actual loader
path verified. The original text, single-image and multiple-image cases each
run twice with reset. All **2298 complete 248320-float logits** and **1149 reset
pairs** match the accepted baseline byte for byte. Frozen matched HF references
pass the unchanged `0.001 + 0.001 * abs(reference)` gate, with zero outliers and
zero argmax mismatches. Maximum absolute errors are 0.0000228882, 0.000739098
and 0.000905991 respectively. Corrupted-output and perturbed-reference negative
controls fail as expected.

A fresh current-source CMake build passes ON/OFF/ON toggling without deleting
the object manually. Both ON objects match; the OFF object matches the
accepted ILP object byte for byte. The ON device fatbin exactly matches the
isolated measured candidate. All 21 non-GGUF link-input entries match the
previously validated current-source ILP CMake build. Eleven GGUF and fourteen
attention tests pass, with one second-device test skipped. Nevertheless, the
complete three-case model test and timing above belong to the **isolated
relink**, not a rerun of the current-source CMake shim or a rebuilt runner.

The first native test attempt started before the candidate build completed,
without a verified candidate loader path; that failed log is retained, not
counted as candidate validation. The corrected run verifies the loader path.
The first BAAB attempt also stopped during loader-map readiness checking,
before any process completed: the shim had appeared before the extension.
The retry waits for both libraries and retains all original path/no-libtorch
checks. Its entire BAAB series is rerun; no measured samples are selected from
the failed attempt.

No original model or deployment file is overwritten. HF and the raw-image
frontend are not rerun; this does not establish full 4096-token, concurrency,
other-model or other-device coverage. The optional conversion remains OFF by
default despite passing this fixed-workload acceptance.

## Local evidence

Under ETSupport `artifacts/gguf-20261029`:

- `ilp-vector-convert-probe-v1`: four variants, eleven real geometries,
  132 complete byte-exact micro-output comparisons and compiler machine IR.
- `build-decode-vector-convert-v1`, `test-vector-convert-v2.log`,
  `decode-vector-convert-validation-v1`: isolated candidate and numerical gate.
- `decode-vector-convert-benchmark-v1`, `decode-vector-convert-benchmark-v3`:
  normal ABBA/BAAB timing; failed v2 is retained separately.
- `build-decode-byte-convert-cmake-v1`: fresh build, toggle results, native
  tests and `host-inputs-comparison-v2.json`.
- `ilp-followup-experiments-v2.json`: 648 exact micro-output comparisons across
  the broader investigation; 144 grouped-dot tolerance comparisons are
  separate and not promoted.
- `decode-vector-convert-acceptance-v1.json`: independent rehash, complete
  output/reference check, numerical gate, timing and build-provenance audit.
