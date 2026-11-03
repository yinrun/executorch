# Native GGUF four-component arithmetic

`EXECUTORCH_MUSA_GGUF_QUAD_ARITHMETIC=ON` keeps four decoded weight components
together through dequantization and accumulation in the existing Q4/Q5 GEMV
and predecoded Q6 GEMV. It defaults to **OFF** and is MUSA-only. No operator,
weight format, runtime ABI or PTE/PTD change is required.

The tested configuration is MUSA SDK 5.1, MP22, ILP scheduling, and
Qwen3.5-4B **Q4_K_M**, with FP32 activations/accumulation. This is not a BF16
model experiment or qualification of other compilers/devices. For an already
configured MUSA build:

```sh
cmake -S . -B <build> \
  -DEXECUTORCH_MUSA_GGUF_MAX_ILP=ON \
  -DEXECUTORCH_MUSA_GGUF_QUAD_ARITHMETIC=ON
cmake --build <build> --target aoti_cuda_shims test_musa_gguf test_musa_causal_attention -j8
```

The previous `EXECUTORCH_MUSA_GGUF_BYTE_CONVERT` option may be ON or OFF when
quad arithmetic is enabled: both produce the same kernel object. Disabling
quad arithmetic restores the selected previous byte-conversion/scalar path.
CUDA/ROCm and other MUSA translation units are unchanged.

## Mechanism and experiment selection

The preceding byte-conversion optimization exposed unsigned bytes but
immediately extracted scalar components. Keeping their subsequent arithmetic
in four-component values lets this SDK select packed conversions. For the
wide Q4 K-block loop, post-register-allocation machine IR changes from 32
`MOV_U8888_TO_F32_BST1` operations to eight `MOV_U8888_TO_F32_BST4` operations,
and removes 24 combined bit-extraction operations. Excluding `IMPLICIT_DEF`
pseudo-operations, that loop goes from 215 to 167 static operations.

The predecoded Q6 head similarly replaces 32 scalar signed conversions with
eight `MOV_S8888_TO_F32_BST4` operations. Its whole-function static count falls
from 569 to 479, excluding `IMPLICIT_DEF`; this is not the Q4 loop's scope.
These are compiler machine-IR observations, not final binary disassembly or
dynamic cycle counts. Neither count predicts a proportional model speedup.

The production helper contains four ordinary scalar `fmaf` calls. Each
component retains its original first-input FMA, second-input FMA, partial
accumulator and final reduction order. Scale multiply/subtract remains
uncontracted under `-ffp-contract=off`. There is no lower-precision arithmetic,
TF32, new dequantization FMA, or regrouped dot-product summation.

The initial `__builtin_elementwise_fma` probe does not compile with this SDK.
An explicit LLVM vector-intrinsic alias is investigated separately but is not
selected. The scalar-helper version avoids that alias and its compiler
warning. Three successful probe groups cover **330 complete byte-exact
micro-output comparisons** over eleven real Q4/Q5/Q6 geometries. Explicit
dequantization FMA does not establish a compelling additional gain and is not
selected. Extra compiler FOP/burst switches produce identical device fatbins
on the selected source; they are not added to the build.

## Normal whole-model timing

The paired baseline already includes ILP scheduling, FFN merging, lossless Q6
predecode and the previous byte-conversion optimization. Both variants reuse
the same model bytes, native runner and bundle extension. GDN merging,
shared-prefix attention, MTP, vocabulary pruning and requantization remain
disabled.

ABBA and BAAB series each use four processes, one complete warmup and two
measured generations per process. Each variant has eight measured runs.
All original 28 prompt tokens, 88 decode steps, 89 generated IDs and EOS
match. Normal timing has no profiler, launch blocking or logit dump;
complete readback, finite checks and argmax remain enabled. Library maps
verify the chosen shim and no libtorch/libpython. Compilation, profiling,
heavy audits and other investigation GPU jobs run outside timed series.

| Variant | Decode minimum | Decode median | Decode maximum | Serial prefill median |
| --- | ---: | ---: | ---: | ---: |
| ILP + previous byte conversion | 6.9200 | 6.9593 | 7.5052 | 7.0160 |
| ILP + four-component arithmetic | 7.1544 | 7.1824 | 7.4406 | 7.1944 |

Units are tokens/s. The pooled decode median improves **3.21%**; independent
series improve **3.37%** and **3.03%**. All samples, including transient high
readings, are retained. The maximum is not sustained performance. The GPU is
shared with desktop applications and clocks are not locked. Historical
measurements from other desktop conditions must not be used as this pair's
baseline. Prefill is still serial-token execution, not batched prefill.
**The 10 tokens/s objective remains unmet.**

## Refreshed global profile

A separate nonblocking trace covers loading, serial prefill, a complete warmup
generation, a complete measured generation and output checks. The original
IDs/EOS and actual native library maps are verified. Measured decode retains
812 kernel labels and twelve stream synchronizations per step: this change
optimizes existing kernels rather than reducing their count.

Profiled decode host-window median is 145.402 ms; compute-track interval union
is 121.959 ms, copy-label union 0.984 ms and post-execute validation gap
1.315 ms. The profiler's generation rate is 6.822 tokens/s, **not** the normal
7.182 tokens/s timing result. Async labels can share intervals; their durations
must not be summed, nor added to CPU synchronization wait. These observations
keep GPU execution as the priority for further investigation; they do not
uniquely attribute every delay or prove an exclusive-device throughput ceiling.

## Precision and integration

The isolated runtime relinks only the GGUF kernel object into unchanged
historical host objects. Eleven native GGUF tests pass with its loader path
verified. Text, single-image and multiple-image trajectories each run twice
with reset. All **2298 complete 248320-float logits** and **1149 reset pairs**
match the accepted baseline byte for byte. The unchanged frozen-HF gate
`0.001 + 0.001 * abs(reference)` has no outliers or argmax mismatches.
Maximum absolute errors remain 0.0000228882, 0.000739098 and 0.000905991.
Corrupted-output and perturbed-reference negative controls fail as expected.

A fresh current-source CMake build passes ON/OFF/ON toggling automatically.
ON matches the measured candidate's device fatbin; OFF reproduces the accepted
byte-conversion object. With quad enabled, toggling the older byte option
OFF/ON leaves the object identical. Separate default and ILP-only compilations
also reproduce their respective accepted objects byte for byte. All 21
non-GGUF link-input entries match the previous current-source CMake build.
Eleven GGUF and fourteen attention tests pass, with one second-device test
skipped. Full three-case model acceptance and normal timing above belong to
the **isolated relink**, not a fresh whole-model run of the CMake shim or a
rebuilt runner.

Original model/deployment files are protected by hashes and are not replaced.
HF and the raw-image frontend are not rerun. This fixed-workload result does
not establish full 4096-token, concurrency, other-model or other-device
coverage. The new option remains OFF by default.

## Evidence

Under ETSupport `artifacts/gguf-20261029`:

- `quad-arithmetic-probe-v1` through `v4`, `quad-burst-probe-v1`, and
  `quad-arithmetic-probes-v1.json` retain the failed, rejected and selected probes.
- `build-decode-quad-arithmetic-v1`, `test-quad-arithmetic-v1.log`, and
  `decode-quad-arithmetic-validation-v1` retain isolated numerical acceptance.
- `decode-quad-arithmetic-benchmark-v1` and `v2` retain normal ABBA/BAAB timing.
- `build-decode-quad-arithmetic-cmake-v1` retains toggle, compatibility,
  binary-provenance and native-test checks.
- `global-quad-arithmetic-profile-v1` retains the separate complete-generation
  nonblocking trace, actual native library maps and interval-union analysis.
- `decode-quad-arithmetic-acceptance-v1.json` is the independent full-output,
  frozen-reference, timing, build and profile audit.
