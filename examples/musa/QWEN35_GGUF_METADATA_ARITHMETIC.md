# GGUF metadata arithmetic and narrower-thread probes

These follow the [local-first reduction investigation](QWEN35_GGUF_LOCAL_REDUCTION.md).
They keep Qwen3.5-4B Q4_K_M weights, FP32 arithmetic and the original packed
layout. They do not change production dispatch, re-export the model, introduce
TF32/MTP/requantization, or establish 10 tokens/s. The local-reduction source
option is independent of these unpromoted experiments.

## Four-component metadata arithmetic

The baseline converts four six-bit scale/minimum factors separately. One
variant packs them into four unsigned bytes before conversion; another keeps
four unsigned integers and converts that vector directly. Both multiply the
result by the original FP16-derived FP32 bases, preserving exact weight
dequantization and the original accumulation/reduction order.

Forward/reverse sweeps over eight real matrices and three changing inputs
pass **144 complete byte-exact outputs**, including the unchanged control.
The byte-packing variant regresses on several Q4 shapes: attention gate ratios
are 0.9350/0.9330 and merged gate/up 0.9793/0.9777. It is not selected.

The unsigned-integer vector has small gains: Q5 QKV 1.0064/1.0077, FFN down
1.0110/1.0097 and merged gate/up 1.0051/1.0047, while attention gate is
0.9927/0.9957. These alternating native-event ratios do not establish a general
model win. Neither metadata variant receives full-model promotion or is
included in the local-reduction source option.

A post-register-allocation compiler dump confirms that both expressions
actually change conversion instructions, rather than relying on source syntax:

| Metadata form | Q4 one-part loop | Q4 four-part loop | Q5 loop |
| --- | ---: | ---: | ---: |
| Four scalar conversions | 169 | 169 | 183 |
| Packed unsigned bytes | 167 | 168 | 182 |
| Unsigned-integer vector | 165 | 166 | 180 |

Counts cover the largest self-loop, excluding IMPLICIT_DEF; they are static
compiler machine-IR operations, not final ISA or dynamic cycles. The byte form
replaces four scalar U32-to-F32 conversions with one U8888-to-F32 BST4 operation,
but adds packing/extraction operations. The integer vector instead uses one
U32-to-F32 BST4 operation without that extra packing. Fewer operations alone
do not establish a speedup: the byte form is slower despite its lower count.
All variants report zero local/spill bytes. Q4/Q5 temporary-register ranges
are 109--111 for the control, 116--118 for byte packing and 110--112 for the
integer vector; these are not measured occupancy or stall diagnoses.

## Narrower work assignment with local reduction

A separate six-variant family tests 4/8/16 components per lane with both
component-first and local-first reductions. It explicitly uses 4/8/16-byte
payload loads for the corresponding widths; narrower variants do not overread
the last packed block. Thread blocks remain 128 threads, using subgroup widths
32/16/8 respectively. No launch uses a subgroup wider than 32.

All **288 complete outputs** pass the unchanged native micro tolerance; the
144 component-first outputs also match byte for byte. Maximum absolute error
across rounding-changing variants is 0.00000190735 and maximum normalized
micro error 0.000738012. These are micro checks, not new model HF acceptance.

Reducing register use does not help here. Compiler-reported temporary registers
fall from 109--111 at width 16 to 65--71 at width 8 and 38--42 at width 4,
without local memory or spills. Nevertheless, the narrower variants are
substantially slower, including with local-first reduction. In the first
sweep, merged gate/up ratios are 0.7512 for width 8 and 0.6239 for width 4,
versus 1.0309 for width-16 local reduction. This rejects narrowing as a route
to a general gain; it does not claim a measured occupancy explanation.

All controls and slower observations are retained. No whole-model speedup is
inferred by combining ratios from these sweeps with earlier model timings.

## Evidence

ETSupport `artifacts/gguf-20261029` contains:

- `metadata-vector-arithmetic-probe-v1`: three variants, builds, resource logs,
  native harness, both sweeps and per-variant machine-IR dumps.
- `analyze_metadata_vector_v1.py`, `metadata-vector-decision-v1.json`:
  opcode counts, full-output checks and independent source/library rehashes.
- `narrow-local-reduction-probe-v1`: six variants, both native sweeps and resource logs.
- `audit_narrow_local_reduction_v1.py`, `narrow-local-reduction-decision-v1.json`:
  independent output-count, tolerance, exact-control and unchanged-binary checks.
