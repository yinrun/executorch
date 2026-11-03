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
10 tokens/s result. A tree-reduction variant is investigated separately.
Microbenchmark timing through Python dispatch is not a native whole-model
throughput claim.

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
