# Packed lane width and integer-dot feasibility

These isolated probes follow the [packed counter investigation](QWEN35_GGUF_PACKED_COUNTER_FOLLOWUP.md).
The qualified Qwen3.5-4B Q4_K_M runtime, FP32 activations/state, model export
and full-output numerical contract remain unchanged. Neither a new lane
mapping nor an integer-dot model path is promoted. Stable 10 decode tokens/s
is not established.

## Current packed-kernel lane mapping

The older vector-width sweep predates quad arithmetic and local-first
reduction. This experiment changes the current packed-quad kernel instead.
Each lane handles one, two, four or eight packed words per half-block, using
32-, 64-, 128- or paired 128-bit payload loads. Every word still converts four
quantized levels together and uses explicit FP32 FMA. Original packed bytes,
block order and Split-K partition boundaries remain intact.

The number of components per lane changes the local/subgroup summation tree,
so noncontrol variants require tolerance checks, not a byte-equality promise.
The four-word control preserves the qualified local-first order. Its device
fatbin is byte-identical to the previous standalone dequantization control
in both new families; this is not whole-library identity to the native shim.

| Words/lane | FP32 accumulators/lane | Lanes/output row | Rows/block | Q4 single-part temp registers | Q5 temp registers |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 32 | 4 | 38 | 42 |
| 2 | 8 | 16 | 8 | 65 | 71 |
| 4, control | 16 | 8 | 16 | 109 | 111 |
| 8 | 32 | 4 | 32 | 195 | 193 |

All use 128 threads per block and report zero local-memory/spill bytes.
Register counts are compiler reports, not measured occupancy. Lower register
usage does not predict better throughput in this case.

Both forward and reverse sweeps cover eight real Q4/Q5 matrices and three
changing inputs. The 1/2/4-word family performs 48 runs; the 4/8-word family
performs 32. All **240 complete output comparisons** pass the unchanged
`0.001 + 0.001 * abs(reference)` micro tolerance with finite values and exact
argmax. The **96 control outputs** are byte-identical. Maximum absolute errors
are 0.00000190735 and 0.000000953674 for the respective families; maximum
normalized errors are 0.000857189 and 0.000645597, both below one. This does
not substitute for the full text/image HF trajectory gate.

Each process uses at least three seconds of alternating active warmup, then
100 recorded reference/candidate event-timing pairs after two initial pairs.
All 8000 pairs and post-pair frequency samples are retained. Sampling reads
1.4 GHz 6424 times and 730 MHz 1576 times. No frequency is requested or locked;
active warmup does not guarantee a particular clock. There is no profiler,
preload or launch blocking during timing.

Representative reference/candidate median-time ratios, first/reverse order:

| Matrix | 1 word | 2 words | 8 words |
| --- | ---: | ---: | ---: |
| Q5 QKV | 0.6766 / 0.6827 | 0.8376 / 0.8325 | 1.0322 / 1.0371 |
| Q4 SSM output | 0.6960 / 0.6824 | 0.8535 / 0.8844 | 0.8914 / 0.9035 |
| Q4 attention Q | 0.6341 / 0.6427 | 0.8343 / 0.8391 | 0.9322 / 0.9629 |
| Q4 FFN down | 0.6583 / 0.6685 | 0.8268 / 0.8331 | 0.9317 / 0.9225 |
| Q4 merged gate/up | 0.5566 / 0.5522 | 0.7766 / 0.7828 | 0.9212 / 0.9662 |

A ratio above one favors the candidate. Every ratio is against its own
alternating qualified-runtime reference; the two families are not directly
paired against each other. The eight-word Q5 result is not a general gain:
its corresponding unchanged standalone controls already read 1.0131/1.0198,
and every large Q4 geometry regresses. The small attention-K control varies
substantially in both families. All eight shapes and controls are retained
in the audit rather than selecting favorable cases. Long-K timings include
the separate partial-reduction launch on both sides.

## Integer primitive prerequisite

Before implementing a different dot-product algorithm, a separate MP22 probe
checks the SDK integer primitives. It does not quantize an activation, change
a model kernel, prune logits or benchmark a model. Independent host integer
arithmetic is a primitive-test oracle, not CPU model inference.

The SDK 5.1 signed `__dp4a(int, int, int)` wrapper extracts bytes through plain
`char` in `mp_ext_61_intrinsics.hpp`. This AArch64 build uses unsigned plain
`char`, confirmed by a compile-time assertion. Consequently the unmodified
wrapper fails a negative-byte fixture. For both packed inputs `0xffffffff`
and initial accumulator `21432 - 32768`, the correct signed result is
**-11332**, while the observed result is **248764**: exactly the result of
interpreting all eight bytes as unsigned 255. The unsigned overload gives
the correct result in that fixture. The failed build/run report is retained.

Two independent corrections are verified without changing the installed SDK:

- Explicit `signed char` vector conversion before `__musa_dot8_s32` fixes
  sign interpretation under the original compiler defaults.
- Recompiling the original device source with only `-fsigned-char` also
  passes the original 262156 signed/unsigned checks and four output canaries.
  This is an isolated diagnosis, not a production compiler-flag change.

The second prototype additionally tests a packed unsigned-dot formulation
for Q4/Q5's nonnegative levels and signed activation bytes. For four lanes,

```text
sum(level * signed_byte)
  = unsigned_dot(level, byte_bits XOR 0x80)
    - 128 * unsigned_dot(level, 1)
```

The XOR and one are applied to each byte. Levels are in 0..31, so every
integer partial fits safely in int32. The complete corrected probe covers
explicit signed, unsigned and this u5-by-s8 formulation over two changing
65539-element input sets, including byte extrema and a non-block-aligned
tail: **393234 scalar results and six canaries pass exactly**.

Post-register-allocation target machine IR selects three `DOT8_DST0`
operations for the ordinary unsigned result and the two terms of the
corrected u5-by-s8 result. Explicit signed arithmetic still uses scalar
`IMADD_S32` operations. This is compiler machine IR, not final binary
disassembly, measured instruction throughput or a GEMV speedup.

The result only establishes a usable integer primitive and exposes a local
SDK wrapper signedness hazard. A lossless FP32 input decomposition, residual
and extreme-value handling, actual packed GEMV, end-to-end preparation cost,
and full-model precision/performance validation are **not implemented or
accepted here**. The original FP32 model path remains in use; no lossy input
quantization or relaxed gate is silently substituted.

## Evidence

Under ETSupport `artifacts/gguf-20261029`:

- `packed-lane-width-probe-v1/v2`: sources, builds, resources, complete
  comparison summaries, both orders and all unfiltered timing pairs.
- `audit_packed_lane_width_v1.py`, `packed-lane-width-decision-v1.json`:
  protected hashes, control-device identity, resources and all 240 checks.
- `packed-dot-feasibility-v1`: retained default-wrapper failure, diagnostic
  fixture, machine IR and isolated signed-char control.
- `packed-dot-feasibility-v2`: explicit signed and corrected unsigned-dot
  source, integer fixtures, canaries and machine IR.
- `audit_packed_dot_feasibility_v1.py`, `packed-dot-feasibility-decision-v1.json`:
  independent result, source, flag-difference and target-instruction audit.
