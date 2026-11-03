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
Three focused Python tests pass; nine pass when including adjacent packed-op
and model-metadata regressions. Three actual Q6 matrix geometries, with three
activation vectors each, are byte exact against the original native kernels.
MP21 compilation succeeds; hardware execution is only checked on MP22.

## Continuous three-way benchmark

On 2026-11-03, a continuous A/B/C/C/B/A series compared the previous deployed
export/shim, a freshly exported unpreprocessed control, and the Q6-preprocessed
candidate. The control and candidate use the same new shim. All variants use
the original runner, extension and 28-token text prompt, generating 89 tokens to
EOS (88 decode steps). Each process has one warmup and two measured generations;
all 18 trajectories match. Profiling, launch blocking and logits dumps are off;
complete output readback, finite checks and argmax remain enabled.

| Variant | Minimum decode tok/s | Median decode tok/s | Maximum decode tok/s |
| --- | ---: | ---: | ---: |
| Previous deployed export/shim | 6.491739 | 6.935159 | 7.605488 |
| Fresh unpreprocessed control | 7.457675 | 7.471915 | 7.582305 |
| Q6-preprocessed candidate | 7.559308 | 8.259904 | 8.478574 |

Q6 preprocessing improves the median by **10.5% relative to its fresh control**.
The combined candidate is 19.1% faster than the previous deployment in this
series; that entire gain must not be attributed to the Q6 kernel. The serial
token-prefill medians are 6.390, 7.476 and 8.366 tok/s, respectively. This is not
batched prefill. The **10 tok/s decode target is not met**, including by the
candidate's fastest measured run.

The series lasts 379.365 seconds, including final artifact checks, with
realtime/monotonic/boottime continuity checks. This is a shared, unlocked desktop
M1000 rather than an exclusive, fixed-clock benchmark. Slow samples are retained.
Live process maps verify the requested shim and original extension without
libtorch/libpython. The frozen deployment and baseline binaries remain unchanged.

## Evidence and remaining work

Local artifacts under the ETSupport workspace's `artifacts/gguf-20261029` include
`repack-probe-v1`, `build-decode-predecoded-v1`, `decode-predecoded-export-v1`,
`decode-predecoded-control-export-v1`, `predecoded-isolation-v1`,
`decode-predecoded-controlled-validation-v1`, and
`decode-predecoded-fulltext-benchmark-v1`. Their adjacent scripts preserve build,
export, validation and benchmark commands. `decode-predecoded-validation-v1`
retains the initial failed historical-package byte-exact gate, not a passing gate.
The final audit is `decode-predecoded-acceptance-v1.json`.

An experimental Q4 metadata-predecoded layout was slower and was not integrated.
Further Q4 access-layout work should be measured before changing the deployment
format again. Full 4096-step trajectory, migration, raw-image performance and
batched prefill acceptance are not added by this change.
