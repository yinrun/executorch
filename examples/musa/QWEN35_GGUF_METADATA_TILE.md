# GGUF metadata loads and prepared Q6 row tiling

Measured on 2026-11-03 with M1000/MP22, MUSA 5.1, Qwen3.5-4B
Q4_K_M and the accepted opt-in Q6-predecoded export. This is not the BF16
model. Activations, dequantized weights and accumulation remain FP32.

## Changes

The aligned 16-component Q4/Q5 decode path reads the original 16-byte block
header with one `uint4` load, then extracts the two FP16 scales and packed
six-bit group factors. The four-component fallback retains its original byte
loads. The existing dispatcher guarantees 16-byte alignment for the wide path;
both original block sizes and row strides preserve this alignment.

Prepared Q6 decode now supports two- and four-row-per-thread tiles. Matrices
with at least 4096 rows and exactly 2560 columns select the four-row tile,
including the complete 248320-row output head. Other aligned single-token
shapes keep the two-row tile, avoiding the small-matrix regression observed
with unconditional larger tiles. Batched and unaligned inputs keep the generic
implementation. Per-lane FMA order and the original reduction tree are retained.

No new packed format, weight allocation, vocabulary filtering, TF32 or
speculative decoding is introduced. Both benchmark sides use **the same PTE
and PTD**; only the native shim changes. Re-export is not needed. The original
GGUF, accepted baseline shim, PTE/PTD and relocated deployment remain unchanged.

## Performance

Two separate uninterrupted series use A/B/B/A and B/A/A/B orders. Each process
runs one warmup and two measured generations. The original 28-token text prompt
generates 89 tokens to EOS, with 88 timed decode steps. All 24 trajectories
match. Normal timing retains output readback, finite checks and full-vocabulary
argmax, without profiling, launch blocking or logits dumping.

| Decode runtime | Minimum tok/s | Median tok/s | Maximum tok/s |
| --- | ---: | ---: | ---: |
| Accepted Q6-predecoded baseline | 7.970056 | 8.093073 | 8.183758 |
| Metadata/tile candidate | 7.899005 | 8.472541 | 8.595187 |

These are eight measured generations per side. The combined median improvement
is **4.69%**. Independent series improve by 3.93% (8.005054 to 8.319702) and
5.26% (8.110851 to 8.537318), taking 220.290 and 214.075 seconds respectively.
They are not one uninterrupted eight-process run: Python regression tests and
profiling ran between the series. Realtime, monotonic and boottime boundaries
are recorded per process. The desktop GPU is shared and clocks are unlocked.
The slower candidate sample is retained; minimum throughput did not improve.
The **10 tok/s target is not achieved**.

Serial token-by-token prefill medians are 8.168414 versus 8.506308 tok/s. These
are not batched-prefill throughput or raw-image preprocessing measurements.

Sixteen isolated variants were explored. Narrower vectors, larger generic
thread blocks/tiles, float-bit integer conversion and expanded Q4 metadata
layouts were not adopted. The final native microbenchmark shows 1.071x for
the representative Q4 FFN gate and 1.017x for the prepared Q6 head. Several
other geometries are flat or slower, including Q5 at 0.965x. An earlier isolated
Q6-head result was 1.114x; it is not claimed as the final production gain.
There is no separate full-model ablation attributing the combined gain to
either change individually.

## Accuracy and regression gates

- Text, single-image and multi-image cases compare 232, 856 and 1210 complete
  248320-float logits respectively: **2298 byte-exact outputs** against the
  previously accepted Q6-predecoded baseline and **1149 reset pairs**.
- Generated IDs, EOS, prompt/decode lengths and all original trajectories match.
- Frozen matched HF step references retain the original elementwise
  `abs(actual - reference) <= .001 + .001 * abs(reference)` gate, with zero
  outliers and argmax mismatches. Maximum errors are 0.00002288818359375,
  0.0007390975952148438 and 0.0009059906005859375 for the three cases.
- Byte-corruption and HF-tolerance negative controls fail as intended. The
  final audit re-reads full output/reference files and verifies their hashes.
- Eleven native GGUF tests pass, including new 15/16/17 and 4095/4096/4097-row
  prepared-Q6 boundaries, offsets, output canaries, batches and stream updates.
  Eleven native attention tests pass; the second-device test is skipped.
  Nine adjacent Python tests pass. MP21 compilation passes; runtime is MP22 only.
- Live process maps verify the intended shim and original native extension,
  without libtorch/libpython. Build, model and protected-package hashes pass.

HF inference and the raw-image/vision frontend were not rerun. Prepared original
image features and frozen step references are reused. The earlier historical
export byte-equality failure remains documented in
[the Q6 preprocessing report](QWEN35_GGUF_Q6_PREDECODED.md); equality here is to
the accepted fresh export, not to that historical deployment. Full 4096-step
context trajectories, concurrency and deployment migration are not newly accepted.

## Next bottleneck and evidence

A separate two-decode-step **blocking** profile attributes 127.770 of 147.182 ms
kernel time to packed Linear (86.81%). Q4 wide kernels account for 74.140 ms,
Q4 narrow kernels 2.749 ms, Q5 17.200 ms, ordinary prepared Q6 15.788 ms and the
full head 17.893 ms. Blocking-profile durations and overlapping CPU/API waits
must not be used as normal decode latency or added together. Q4 weight-read
layout and unpack scheduling remain the next priorities; lossless row/block
interleaving should be evaluated before adding more metadata storage.

Local evidence lives under `artifacts/gguf-20261029` in the ETSupport workspace:

- `q4-unpack-probe-v1`: all variant sources, resource reports and paired tests.
- `build-decode-metadata-tile-v1`: isolated native build and protected hashes.
- `decode-metadata-tile-validation-v1`: complete three-case logits and gates.
- `decode-metadata-tile-benchmark-v1` and `-v2`: both complete timing series.
- `decode-metadata-tile-profile-blocking-v1`: trace and interval-union analysis.
- `decode-metadata-tile-acceptance-v1.json`: final source/runtime/evidence audit.

Adjacent `build_decode_metadata_tile_v1.py`, `validate_decode_metadata_tile_v1.py`,
`benchmark_decode_metadata_tile_v{1,2}.py` and `audit_decode_metadata_tile_v1.py`
retain commands and environment. Native tests use the candidate build's
`LD_LIBRARY_PATH`; the old bundle's `run.sh` overrides it and must not be used
for candidate validation.
