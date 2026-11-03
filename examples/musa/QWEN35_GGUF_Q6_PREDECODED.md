# Opt-in lossless Q6 preprocessing

This candidate keeps Qwen3.5-4B Q4_K_M's quantized levels and FP32
activations/accumulation. It does not requantize to INT8, store dense floating-point
weights, prune the vocabulary, enable TF32, or use speculative decoding.

## Layout and ownership

`musa_q6_predecoded_v1` is a private MUSA layout, **not a GGUF tensor format**.
Its operator format ID is 114. Each original 256-element Q6_K block occupies
320 bytes instead of 210:

- Bytes 0–255 contain the original signed six-bit levels in INT8 containers.
- Bytes 256–319 contain sixteen FP32 products of the original FP16 block scale
  and signed group scale, in native little-endian order.
- FP32 weight reconstruction remains `prepared_scale * float(level)`.
  Per-lane FMA sequencing and the original 128-lane reduction tree are retained.

`predecode_q6()` explicitly creates an independently owned GPU buffer before
export. There is no pointer-keyed runtime cache or hidden assumption that input
storage never changes. Changing the source after preprocessing does not change
the snapshot. Callers must preprocess again when they want a new snapshot.
The native operator reads its supplied prepared buffer on the calling stream;
same-stream input and prepared-weight updates are tested.

The loader replaces only Q6 buffers when explicitly requested and prepares the
shared embedding/output buffer once. Q4 and Q5 remain in their original formats.
The default loader behavior is unchanged. The re-exported PTD contains the
prepared representation, not both representations or preprocessing operations.
The original GGUF file and existing deployment artifacts remain untouched.

For the tested 4B file, 22 unique tensors change layout. Packed storage grows
from 2,692,700,160 to 3,133,685,760 bytes: **440,985,600 extra bytes (420.56 MiB)**.
PTD size grows from 2,732,396,288 to 3,173,381,888 bytes. These are stored tensor
and file sizes, not a measurement of peak GPU allocation. Preprocessing uses
4096-row chunks, but the source and destination of the current tensor coexist
temporarily. Runtime loading transfers the larger PTD; it does not repack weights.

## Usage

```python
from examples.musa.aoti_qwen35_gguf import load_gguf_text

model, loading = load_gguf_text(
    gguf_path,
    context_length=4096,
    valid_length_attention=True,
    predecode_q6_weights=True,
)
```

Export this model using the existing MUSA export pipeline, then run with a shim
built from this revision. The old shim does not understand format 114. Aligned
single-token inference uses a two-row-per-thread Q6 kernel; unaligned and
batched cases retain a checked, order-preserving generic implementation.

Run focused Python and native tests in the configured MUSA environment:

```sh
python -m unittest executorch.backends.musa.tests.test_musa_q6_predecoded -v
cmake --build "$BUILD_DIR" --target test_musa_gguf -j8
LD_LIBRARY_PATH="$BUILD_DIR/backends/cuda:$BUILD_DIR/extension/cuda:$MUSA_HOME/lib" \
  "$BUILD_DIR/backends/cuda/test_musa_gguf"
```

## Validation scope

The initial original-package byte-exact gate failed at text position 1 and its
failure artifacts are retained. A four-way isolation test over 28 prompt steps
found:

- Old export, old versus new shim: byte exact.
- Fresh export without preprocessing versus historical export: maximum absolute
  logit difference 0.00001728535; unchanged argmax and no tolerance outliers.
- Fresh unpreprocessed control versus preprocessed candidate: byte exact.

All 28 embedded MUSA kernel binaries in the two fresh exports have identical
hashes. Thus the historical-package difference also occurs without Q6
preprocessing; the precise older-versus-newer code-generation cause is not yet
isolated. Do not claim byte equality to the historical deployment.

The acceptance gate uses a fresh unpreprocessed control for byte equality and
independently rechecks the frozen matched-HF-step full logits using
`abs(actual-reference) <= 0.001 + 0.001*abs(reference)`. Token IDs, EOS, complete
vocabulary output and reset reproducibility must also match. This does not
relax the HF tolerance or relabel old HF batch-reference failures as passing.

On MP22/M1000, the original text, single-image and multi-image requests passed
the full gate, with 232, 856 and 1210 full-logit comparisons respectively.
All 2298 outputs are byte identical to the fresh unpreprocessed control, including
1149 reset pairs. Autonomous token IDs, EOS and step counts match the historical
requests. Maximum absolute errors against the frozen matched-HF-step references
are 0.00002288818, 0.00073909760 and 0.00090599060, with zero tolerance outliers
and zero argmax mismatches. HF inference and vision encoding were not rerun;
the image requests reuse the original prepared vision features.

Eleven native GGUF tests pass, including 240 new offset/batch/row-tail comparisons,
60 embedding comparisons and a prepared-weight stream-update test. Eleven native
attention tests pass; the second-device test skips on this single-device machine.
Three focused Python tests pass. Three actual Q6 matrix geometries, with three
activation vectors each, are byte exact against the original native kernels.
MP21 compilation succeeds; hardware execution is only checked on MP22.

Continuous three-way throughput results are pending. Microbenchmarks are not
model throughput. An experimental Q4 metadata-predecoded layout was slower and
was not integrated. Full 4096-step trajectory, migration and batched prefill
acceptance are not added by this change.
