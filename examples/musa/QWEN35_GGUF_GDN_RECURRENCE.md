# Experimental native GDN recurrence

`load_gguf_text(..., native_gdn_recurrence=True)` replaces the single-token
FP32 recurrent update with an opaque native MUSA operator. The option is
**OFF by default**. It is independent of packed GDN projection merging;
that earlier projection-only optimization is not enabled here.

## Scope and implementation

The operator accepts one batch, one token, 16 or 32 heads and 128-wide keys
and values. Query/key normalization, query scaling, sigmoid beta and
exponentiated decay are performed before the call. It computes the decayed
state, key-memory product, delta update, updated state and query output in
one kernel. Activations, products, state and accumulators remain FP32.
There is no TF32, requantization or lower-precision state.

The functional output packs the updated state followed by the output vector
into one contiguous allocation. The exporter retains explicit recurrent
state ownership and its existing reset/copy behavior. Inputs are not mutated.
The native shim validates exact shapes, dtype, device, contiguity, backing
storage spans and output/input nonoverlap; it uses the calling MUSA stream.
Other shapes are rejected rather than silently approximated.

This is a new native operator and ABI entry, not just a change to the GEMV
kernel. It requires a new export and a runtime with
`aoti_torch_musa_gdn_recurrence_out`. CUDA/ROCm and the default GGUF loading
path do not use it.

## Initial validation: not qualified for deployment

The first serial-partial kernel passes two native GDN tests, twelve existing
GGUF tests, and sixteen Python/eager/export/native integration regression
tests. The native integration test runs a stateful exported model three times
and checks the entire state/output against eager execution. A failed earlier
test command named a nonexistent FFN test module; the corrected command and
its passing log are retained separately.

The Qwen3.5-4B Q4_K_M candidate export contains 24 recurrence calls, 32 merged
FFNs, eight valid-prefix attention calls, one delegate, five user inputs and
the same 64 mutated buffers. GGUF source hash, tensor records, packed sizes,
lossless Q6 predecode and tied output head match the accepted FFN-only export.
The preceding grouped-Q6-head optimization remains enabled in the runtime.

Text and single-image trajectories pass the unchanged frozen-HF full-logit
gate. **Multiple-image acceptance fails an exact argmax check at prefill
position 135**, despite every step remaining within the original numeric
bound. At that position the frozen HF top two logits are 17.6537303925 and
17.6537208557 (IDs 12391 and 440); the candidate reverses their order. Its
maximum absolute logit error at that step is 0.0000271797. Across all 605
multi-image steps, maximum HF error is 0.000745773 and maximum normalized
tolerance error is 0.528809. There is one argmax mismatch, not a numerical
tolerance outlier. Generated IDs/EOS and repeat-after-reset bytes match,
but these facts **do not override the failed per-step argmax requirement**.

The original gate is not relaxed and no position-specific fallback is added.
This initial kernel must not be presented as an accepted optimization or a
10 tokens/s result. The subsequent balanced-tree variant also fails the
same prefill argmax check: all 605 multi-image steps remain within numerical
tolerance, with maximum HF error 0.000711441 and normalized error 0.484177.
Microbenchmark timing through Python dispatch is not a native whole-model
throughput claim.

## Layout-informed correction

The current kernel follows the distinct reduction orders found in the cached
generated Triton recurrence, rather than assuming that `sum` means one global
balanced tree. Both products first accumulate indices `r`, `r+32`, `r+64`,
`r+96` sequentially for each of 32 partials. The key-memory sum combines
four-partial groups, then eight groups. The query-output sum combines
eight-partial groups, then four groups. Those two trees are not interchangeable.
All arithmetic remains FP32 with contraction disabled. This is a fixed-order
implementation for the tested geometry, not a guarantee that arbitrary future
Triton autotuning configurations use the same order.

A new cancellation-sensitive native regression checks both reductions,
complete updated state/output and input preservation for 16 and 32 heads.
It deliberately fails against both the original serial and balanced-tree
libraries. All three native tests and four Python/eager/export/native tests
pass with the correction. A fresh current-source CMake build has exactly the
same device fatbin as the isolated library used for full-model validation;
both libraries pass the native tests. Fatbin comparison is read-only.

Text, single-image and multi-image validation now passes **2298 complete
248320-float logits**, 1149 byte-exact reset pairs and all 18 negative controls.
An independent audit rehashes all candidate and accepted grouped-head outputs:
**all 2298 are byte-identical to the accepted grouped-head runtime**. This is
stronger than tolerance agreement with that runtime. They remain nonidentical
to the older predecoded baseline, because the already accepted grouped Q6
head changes rounding. Maximum HF errors remain 0.0000226498, 0.000739098 and
0.000905991, respectively. Original IDs/EOS, per-step argmax and the unchanged
HF gate all pass. HF and the raw-image frontend are not rerun.

The model still uses the new GDN PTE/PTD export. Full-model acceptance belongs
to the isolated historical-host relink, not a full-model rerun with the fresh
CMake host. Numerical correctness alone does not enable this option by default
or establish a speedup.

## Global profiling and performance decision

The original serial candidate has a fresh ordinary ABBA comparison against
the accepted grouped-head runtime: one complete warmup and two measured
generations per process, four measured runs per variant. The original
28-token prompt, 88 decode steps and 89 generated IDs including EOS are kept.
No profiler, launch blocking, logit dumping, compilation, heavy audit or other
investigation GPU job runs during timing. Full readback/finite/argmax checks
remain enabled, and runtime maps exclude libtorch/libpython.

| Serial-kernel experiment | Decode minimum | Decode median | Serial prefill median |
| --- | ---: | ---: | ---: |
| Accepted grouped head, fresh control | 7.2669 | 7.3031 | 7.2765 |
| Initial native GDN recurrence | 7.1626 | 7.1679 | 7.1436 |

Units are tokens/s. The median ratio is 0.981488, a **1.85% regression** in
this series, not a deployment gain. That serial kernel also fails numerical
acceptance. Its separate profiled speed is 6.6864 tokens/s and must not be
used as ordinary benchmark throughput.

The global trace explains why a faster isolated recurrence need not help:
every measured decode step has **932 kernel labels versus 812** in the accepted
export, a net increase of five per GDN layer (120 per step). The opaque call
forces normalized queries/keys, sigmoid values and reset state to materialize;
separate state-copy and output-square kernels also appear. Three previously
fused recurrence stages becoming one native launch does not offset these new
fusion boundaries. The candidate still has 12 stream synchronizations per
step. Simply reducing the recurrence's own launch count is insufficient.

The serial trace's host execution window median is 149.496 ms, compute-track
interval union 120.916 ms, copy-label union 0.957 ms and output-validation gap
1.314 ms. It is not paired with the earlier accepted profile under identical
desktop conditions. Async labels share intervals; neither their duration sum
nor CPU waits plus GPU durations is valid per-kernel attribution. The paired
ordinary benchmark, not subtracting these two traces, determines performance.

The corrected kernel has its own ordinary ABBA comparison, with the same
warmup, trajectory, validation and isolation rules and four measured runs per
variant:

| Corrected-kernel experiment | Decode minimum | Decode median | Decode maximum | Serial prefill median |
| --- | ---: | ---: | ---: | ---: |
| Accepted grouped head, fresh control | 7.3273 | 7.4318 | 8.0102 | 7.6583 |
| Layout-informed native GDN | 7.1323 | 7.1501 | 7.1806 | 7.1408 |

The median ratio is 0.962101, a **3.79% regression in this series**. The high
baseline sample is retained, not removed; a shared desktop and unlocked clocks
prevent treating this percentage as a universal cost of GDN. Neither serial
nor corrected timing establishes a gain, and the two candidate variants are
not a direct paired comparison with one another. **The corrected operator
remains OFF by default and does not replace the accepted deployment.** The
recorded accepted grouped-head result remains 7.3423 median tokens/s in its
own earlier paired series, not the baseline of this new experiment.

The next optimization should broaden the
boundary to include query/key preparation and output normalization/state
handling, then check actual launch counts before a whole-model performance
claim. Removing checks, relaxing argmax or a position-specific fallback is
not part of the plan. The 10 tokens/s goal remains unmet.

The subsequent [wider Triton GDN experiment](QWEN35_GGUF_FUSED_GDN.md)
passes full accuracy and reduces launches to 692, but its ordinary ABBA/BAAB
timing does not establish a reliable gain. It also remains unpromoted.

## Reproduction and evidence

For an already configured MUSA build, `aoti_cuda_shims` includes the new
native kernel, and `test_musa_gdn` runs its shape/storage/alias/stream tests.
Run the Python tests with:

```sh
python -m unittest executorch.backends.musa.tests.test_musa_gdn_recurrence -v
```

Setting `EXECUTORCH_MUSA_GDN_RUNNER` and `LD_LIBRARY_PATH` to a matching native
runner/runtime also enables the stateful AOTI integration test. Enabling the
loader option is experimental and still requires the complete model gate.

Local ETSupport evidence under `artifacts/gguf-20261029`:

- `gdn-recurrence-probe-v1`: nine tile/partition variants, complete GPU output
  and state checks for 16 and 32 heads. An initial missing global `libmusart`
  load failed before execution; the corrected probe is recorded separately.
- `build-gdn-recurrence-v1`, `build-gdn-recurrence-isolated-v1`: current CMake
  build and isolated addition of only the new GDN host/kernel objects to the
  accepted historical host. The GGUF object is unchanged.
- `test-gdn-recurrence-native-v1.log`, `test-gdn-recurrence-gguf-v1.log`,
  `test-gdn-recurrence-regression-v2.log`: passing tests.
- `decode-gdn-recurrence-export-v1`: fresh candidate PTE/PTD and six eager
  text/image probes. Old deployment files are not replaced.
- `gdn-recurrence-validation-v1`: failed strict full-model acceptance;
  `gdn-precision-diagnostic-v1.json`: independent full multi-image diagnosis.
- `gdn-recurrence-balanced-v1`, `build-gdn-balanced-v1`, and
  `gdn-balanced-validation-v1`: the subsequent tree-reduction experiment.
- `gdn-balanced-precision-diagnostic-v1.json`: all 605 multi-image steps from
  the rejected balanced-tree experiment, with reset-byte checks.
- `gdn-recurrence-benchmark-v1`, `global-gdn-recurrence-profile-v1`, and
  `gdn-followup-audit-v1.json`: diagnostic serial-kernel timing, complete
  global trace and verified launch/trajectory/precision summaries.
- `gdn-recurrence-layout-v1`, `build-gdn-layout-v1`, and
  `build-gdn-layout-cmake-v1`: layout-informed prototype, isolated runtime and
  matching current-source device build.
- `gdn-layout-controls-v1`, `test-gdn-layout-native-v2.log`, and
  `test-gdn-layout-python-native-v1.log`: native positive/negative controls and
  passing stateful AOTI integration.
- `gdn-layout-validation-v1`, `gdn-layout-acceptance-v1.json`, and
  `gdn-layout-benchmark-v1`: complete corrected-kernel precision, independent
  byte-identity audit and separate ordinary timing.
- `gdn-layout-decision-v1.json`: final numerical-pass/performance-regression
  decision and source/test provenance. No accepted model or library is replaced.
