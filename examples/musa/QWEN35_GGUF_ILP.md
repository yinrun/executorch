# Native GGUF compiler scheduling experiment

The MUSA SDK 5.1 compiler exposes
`-mllvm -mtgpu-enable-max-ilp-scheduling-strategy=true`. On the unchanged
metadata/tile GGUF kernel source, this changes instruction/register scheduling,
not the packed format, dequantization formula, explicit FMA/reduction tree or
`-ffp-contract=off` contract. It is a version-sensitive experimental backend
option, not a generic CUDA/ROCm flag or a recommendation for other MUSA SDKs.

The MUSA-only CMake option `EXECUTORCH_MUSA_GGUF_MAX_ILP=ON` forwards this flag
only when compiling `musa_gguf_kernels.mu`. Its default is OFF. CUDA/ROCm builds
and other MUSA translation units do not receive it. Unsupported compilers should
leave the option off; enabling it may produce a compiler error. MP22 is the
runtime target of this investigation, not a claim about other architectures.

The initial isolated candidate recompiles only the GGUF device object and
relinks the accepted metadata/tile shim from its unchanged other object/archive
inputs. The original dependency file, library, kernel object and model files
are protected by hashes. It reuses the FFN-merged PTE/PTD and native runner,
with GDN merging and shared-prefix attention disabled. CMake integration and
isolated relink measurements must be distinguished when reporting validation.

Eleven real-matrix microbench geometries, including the complete 248320-row
predecoded Q6 output head, each pass three full byte-exact activations. Many
large Q4/Q5 geometries improve roughly 6–12% in alternating 100-sample medians;
the Q6 head improves about 3%, while the small Q6 attention projection regresses.
Unchanged-code controls expose several-percent timing variability, so these are
candidate-selection measurements, not full-model throughput claims.

Other compiler probes are not promoted: forced-minreg, occupancy bias and FOP/
burst options produce device fatbin bytes identical to the control; a temp-
register cap preserves values but regresses badly. Narrow row tiling and flat
accumulator probes also do not establish a broad speed gain. The initial flat
probe had an overbroad source transformation and did not compile; its corrected
variant is kept separately. No rejected kernel layout or compiler flag is added
to the deployment.

## Full-model acceptance and normal throughput

The model is Qwen3.5-4B Q4_K_M, with FP32 activations/accumulation and lossless
Q6 weight predecoding. This is not a BF16 model experiment. Both measured
variants use the same FFN-merged PTE/PTD bytes; neither GDN merging nor shared
attention prefixes are enabled. No TF32, requantization, MTP, vocabulary pruning
or relaxed acceptance gate is used.

Native tests against the isolated shim pass eleven GGUF tests and eleven
attention tests, with one second-device test skipped. All three original text,
single-image and multiple-image trajectories run twice with reset. All 2298
complete 248320-float logits and 1149 reset pairs match the accepted baseline
byte for byte. Frozen matched HF references pass the unchanged
`0.001 + 0.001 * abs(reference)` gate, with no outliers or argmax mismatches.
Maximum absolute errors remain 0.0000228882, 0.000739098 and 0.000905991.
Negative controls and an independent full-output/reference rehash audit pass.

ABBA and BAAB timing series each have four processes, with one complete warmup
and two measured generations per process. The original 28 prompt tokens,
88 decode steps, 89 generated IDs and EOS remain unchanged. There is no
profiler, launch blocking or logit dumping during timing; complete output
readback, finite checks and argmax remain enabled. Runtime maps verify the
chosen native shim, unchanged bundle extension, and no libtorch/libpython.
Compilation and heavy audits run outside the timed series.

| Variant | Decode minimum | Decode median | Decode maximum | Serial prefill median |
| --- | ---: | ---: | ---: | ---: |
| FFN merge, original scheduling | 6.463 | 6.518 | 6.618 | 6.520 |
| FFN merge, ILP scheduling | 6.968 | 7.030 | 7.063 | 7.006 |

Units are tokens/s, with eight measured runs per variant. Decode median
improves **7.86%**; separate series improve 7.97% and 7.57%. Prefill is the
existing serial-token path, not batched prefill. GPU access is shared and
clocks are not locked. Historical 8.473 tok/s results used different desktop
conditions and cannot be combined with this ratio to claim unmeasured
exclusive-GPU throughput. **The 10 tokens/s target is not achieved.**

The separate complete-generation nonblocking trace retains 812 kernel labels
and twelve stream synchronizations per step. Decode host-window median is
150.169 ms, compute-track union 126.841 ms and output-validation gap 1.307 ms.
These profiled times are not normal throughput. Async kernel labels share
intervals: their durations must not be summed, nor added to CPU wait time.
The profiler reports one exceptional Pixel event; the selected model compute
intervals satisfy the analyzer's monotonic-time checks.

An additional ILP-only launch probe tests 64/128/256/512-thread blocks on eight
real Q4/Q5 geometries with three complete byte-exact activations each. The
64-thread version regresses; larger blocks do not establish a broad gain over
the unchanged 128-thread control. No launch-geometry change is promoted.
Four Q4/Q5 cached/global-load combinations also pass all three complete
activations on eight geometries, without a consistent gain. The unchanged
launch and cache controls have device fatbins identical to the measured ILP
candidate; their several-percent variation is not an optimization. Neither
cache-policy nor launch-layout experiments change the deployment.

## CMake integration

For an existing configured MUSA build, opt in explicitly:

```sh
cmake -S . -B <build> -DEXECUTORCH_MUSA_GGUF_MAX_ILP=ON
cmake --build <build> --target aoti_cuda_shims test_musa_gguf test_musa_causal_attention -j8
```

A fresh current-source CMake build passes ON/OFF/ON toggling without manual
object deletion: both ON objects match, and OFF differs. The final device
fatbin matches the isolated measured kernel exactly. Eleven GGUF tests and
fourteen current attention tests pass, with one second-device test skipped.
The CMake shim also contains current host attention code, unlike the isolated
relink's historical `3aa4731265` host objects. These are distinct binaries;
the normal throughput table above measures the isolated relink only.

The first extraction audit failed because `llvm-objcopy --dump-section`
without a separate output rewrote its input ELF. The measured shared library
and protected deployment were unchanged. The kernel object was deterministically
recompiled and restored to its recorded SHA before rerunning the build in a
fresh directory. The corrected extraction uses a distinct output and passes
all protected-input checks. Failed logs and the rewritten object are retained.
Relinking the original kernel with original inputs also reproduces the original
shared library byte for byte, independently checking the experimental link path.

The current-source CMake shim separately passes all 2298 complete original
logits and 1149 reset pairs, byte-identical to the isolated candidate and
accepted baseline, with the same frozen-HF errors. This model check keeps the
protected runner and bundle extension; it is not a benchmark of an entirely
rebuilt runtime bundle. A separate audit rehashes all CMake-shim outputs and
frozen references, and checks native test logs, runtime maps, source hashes and
protected inputs. Neither experiment reruns HF or the raw-image frontend,
establishes full 4096-token/concurrency coverage, or resolves historical-export
bit-equality differences. The option remains OFF by default and the accepted
deployment is not overwritten.

Local evidence under ETSupport `artifacts/gguf-20261029`:
`compiler-gemv-probe-v1`, `build-decode-ilp-v1`, `test-ilp-gguf-v1.log`,
`test-ilp-attention-v1.log`, `narrow-tiled-metadata-probe-v1`, and
`flat-tiled-metadata-probe-v2`, `ilp-launch-probe-v1`,
`decode-ilp-validation-v1`, `decode-ilp-benchmark-v1`, `decode-ilp-benchmark-v2`,
`global-ilp-profile-v1`, `decode-ilp-acceptance-v2.json`,
`build-decode-ilp-cmake-v2`, `decode-ilp-cmake-validation-v1`, and
`ilp-objcopy-recovery-v1`. Supplemental launch/cache probes live in
`ilp-launch-probe-v1` and `ilp-cache-probe-v1`; the final combined audit is
`decode-ilp-acceptance-v3.json`.
