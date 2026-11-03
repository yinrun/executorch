# Regrouped FP32 Q6 output head

`EXECUTORCH_MUSA_GGUF_GROUPED_Q6_HEAD=ON` is an experimental, default-OFF
specialization for the Qwen3.5-4B output head. Unlike the preceding byte-vector
and quad-arithmetic options, it **changes FP32 summation/rounding order**.
It does not promise byte-identical logits to the original implementation.

The tested model remains **Q4_K_M**, with losslessly predecoded Q6 head
weights and FP32 activations/accumulation. This is not a BF16-weight run.
There is no TF32, MTP, requantization, vocabulary pruning, reduced output
validation, or PTE/PTD modification. Qualification is limited to SDK 5.1,
MP22, and the recorded fixed text/image workloads.

For an already configured MUSA build:

```sh
cmake -S . -B <build> \
  -DEXECUTORCH_MUSA_GGUF_MAX_ILP=ON \
  -DEXECUTORCH_MUSA_GGUF_QUAD_ARITHMETIC=ON \
  -DEXECUTORCH_MUSA_GGUF_GROUPED_Q6_HEAD=ON
cmake --build <build> --target aoti_cuda_shims test_musa_gguf test_musa_causal_attention -j8
```

## Mechanism and dispatch

The specialization requires format 114, batch one, exactly 248320 output
rows and 2560 input columns, and 16-byte-aligned input/weight addresses.
Other shapes, batches, formats, or alignments retain their previous path.
This deliberately does not extend a head-specific win to smaller matrices:
the same row tiling slows several down-projection geometries.

Each lane forms a dot product over sixteen signed quantization levels and
inputs, using four FP32 partial accumulators. A fixed reduction tree combines
the partials before applying their common FP32 scale. Four row tiles share
the input loads. This replaces repeated per-weight scale multiplication with
one scale/FMA per group, but distribution of the scale across a floating-point
sum is not bitwise equivalent. The model-specific fixed bounds also allow
removing row-tail checks; the selected row count is divisible by 32.

Two alternating microbenchmark repeats measure the accepted quad baseline
at 12.3500--12.3528 ms and the new head at 9.4320--9.4475 ms: roughly **31%
head throughput improvement**, or **24% lower head latency**. These are
not whole-model gains. Each comparison reads every output for three input
patterns and retains the unchanged micro tolerance. A separate control
proves three complete outputs byte-identical to the selected grouped
prototype, not to the old ungrouped baseline.

The simplified fixed-shape kernel uses 110 compiler-reported temporary
registers versus 65 for the dynamic-shape grouped prototype, yet measures
faster. Both report zero local memory and spill bytes. Register counts alone
are therefore not used to accept or reject a performance candidate.

## Whole-model acceptance

Normal ABBA and BAAB series each run four processes, with one complete warmup
and two measured generations per process. Each variant has eight measured
runs over the original 28 prompt tokens, 88 decode steps and 89 generated IDs
including EOS. There is no profiler, launch blocking or logit dumping during
timing. Complete readback, finite checks and argmax remain enabled. Compilation,
heavy audits and other investigation GPU work run outside the timed series.

| Variant | Decode minimum | Decode median | Decode maximum | Serial prefill median |
| --- | ---: | ---: | ---: | ---: |
| Accepted quad arithmetic | 7.1609 | 7.1977 | 7.4898 | 7.1907 |
| Quad plus regrouped Q6 head | 7.3091 | 7.3423 | 7.6990 | 7.2919 |

Units are tokens/s. The pooled decode median improves **2.01%**; the separate
series improve **1.66%** and **2.20%**. All samples, including transient high
readings, remain in the report. The maximum is not sustained performance.
The desktop GPU is shared and clocks are not locked. Prefill remains serial
token execution, not batched prefill. **The 10 tokens/s goal remains unmet.**

The numerical gate is the original frozen-HF
`0.001 + 0.001 * abs(reference)` bound, all-finite values, exact argmax and
the original generated token IDs/EOS. The same bound is additionally checked
against the accepted native baseline. Reset reproducibility still requires
byte identity. The previous byte-exact validators are not changed or weakened;
this rounding-changing option has a separate validator and explicit contract.

Full text, single-image and multi-image acceptance compares all 248320 floats
at every step, twice after reset: **2298 full logits and 1149 reset pairs**
pass. Generated IDs/EOS and all argmax results match the original trajectory.
The maximum additional absolute difference from the accepted native baseline
is **0.000003814697265625** in every case. No complete logit vector is
byte-identical to that baseline; all are byte-identical to the grouped
prototype. This is a tolerance-qualified optimization, not a byte-exact one.

| Case | Maximum error versus frozen HF | Maximum normalized gate error |
| --- | ---: | ---: |
| Text | 0.0000226498 | 0.017321 |
| Single image | 0.000739098 | 0.368700 |
| Multiple images | 0.000905991 | 0.646272 |

The normalized error must be at most one; there are no outliers. Neither HF
nor the raw-image frontend is
rerun. Reference hashes are checked against the frozen accepted artifacts.
NaN, positive/negative infinity, a large output perturbation, perturbed HF
references and truncated outputs must fail the gate. A one-bit perturbation
is not claimed to fail a tolerance-based contract.

## Refreshed global profile

A separate nonblocking trace verifies the actual native shim/extension maps
and matches the captured runner PID to the trace. Every measured decode step
has 812 kernel labels, twelve stream synchronizations and exactly one grouped
Q6 head launch. The optimization changes kernel execution rather than launch
or synchronization counts.

This trace measures a 133.445 ms median host execution window, 112.219 ms
compute-track interval union, 0.918 ms copy-label union and 0.855 ms
post-execute validation gap. Its profiled rate is 7.478 tokens/s, **not** the
normal paired 7.342 tokens/s median. It is not paired with the preceding quad
trace under identical desktop conditions; the difference between those two
traces must not be attributed entirely to this head optimization. Async
labels can share intervals and cannot be summed; CPU waits also overlap GPU
work. The measured model speedup remains the ABBA/BAAB result above.

## Build and fallback coverage

The measured isolated library replaces only the GGUF kernel object in the
unchanged historical native host. Original runners, model bytes, extensions
and earlier libraries remain protected by hashes. Native library mappings
are checked; model runs do not load libtorch or libpython.

Current-source CMake ON/OFF/ON builds reproduce ON exactly; OFF reproduces
the previously accepted quad-arithmetic object. ON's device fatbin matches
the measured isolated runtime. Toggling the older byte-conversion option
with quad enabled does not change the object. All 21 non-GGUF link-input
entries match the prior current-source quad build.

Twelve GGUF and fourteen attention tests pass against the current CMake
runtime, with one second-device test skipped. The new full-head test checks
all output rows against double-precision reference sums, output sentinels,
the calling stream and subsequent input updates. Existing byte-exact
batch/offset/tail tests remain unchanged. Full-model precision and timings
belong to the isolated relink, not a rerun with the current CMake host shim.

## Other investigated paths

Nine grouped Q4/Q5 and Q6 probe families plus the static Q4 reduction probe
retain 915 complete micro-output checks. Their tolerance checks are not
substitutes for model acceptance. Most grouped Q4/Q5 variants regress;
fusing the input-sum pass recovers some shapes but is not promoted globally.
The separate byte-exact static Q4 reduction probe does not establish a
convincing broad improvement and is not included in this change.

Nested unroll directives in the early grouped reduction generate dynamic
register indexing. Explicit fixed-index reductions eliminate the pathological
Q4/Q5 32-component version's 128-byte local-memory allocation, but fixing
that regression alone does not make the grouped Q4/Q5 path a general winner.

## Evidence

Under ETSupport `artifacts/gguf-20261029`:

- `grouped-probes-audit-v1.json` and the `grouped-quad-*-probe-*` directories
  retain the broader microbenchmark search, including negative results.
- `grouped-head-probe-v1` retains paired head measurements and the exact
  grouped-prototype control.
- `build-grouped-head-v1` and `grouped-head-validation-v1` retain the dynamic
  grouped prototype; `build-grouped-head-v2` and `grouped-head-validation-v2`
  retain the repository implementation and its independent model runs.
- `build-grouped-head-cmake-v2` retains successful integration and toggles.
  The earlier CMake-v1 attempt stopped before configuration because its
  prerequisite isolated build summary was not yet available; it is not a
  successful build. An initial microbenchmark link attempt used the driver
  library instead of `libmusart` and did not produce a runnable binary.
- `grouped-head-benchmark-v1` and `v2` retain normal paired timing;
  `global-grouped-head-profile-v2` retains the separate nonblocking trace.
  Profile-v1 completed the model but failed its runtime-map capture, so it
  is not the accepted profile. The retry searches the exact runner command
  and verifies its PID against the trace, instead of assuming the runner
  appears in the profiling client's main-thread child list.
- `grouped-head-acceptance-v2.json` independently rechecks complete outputs,
  original reference hashes, reset equality, builds, tests and timings.
