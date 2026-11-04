# Packed dequantization and Q6 reduction follow-up

These isolated experiments follow the qualified
[local-first Q4/Q5 reduction](QWEN35_GGUF_LOCAL_REDUCTION.md). They keep the
Qwen3.5-4B Q4_K_M weights, FP32 activations/state/accumulation and caller-owned
Split-K export. Neither experiment replaces the deployment, adds a production
option, changes the model export or establishes stable 10 decode tokens/s.

## Packed Q4/Q5 dequantization

Four variants compare the current local-first control, vectorized metadata
arithmetic, fused dequantization, and their combination. The previous metadata
probe used component-first reduction; this experiment tests the combination
with the now-qualified local-first path. Both implementations preserve the
FMA accumulation, subgroup reduction and partition-combination order.

The initial vector-FMA builtin does not exist in the installed SDK compiler.
Its failed compile is retained and has no benchmark or model qualification.
The first working implementation uses explicit scalar `fmaf` calls for each
component. The second uses a narrowly scoped `clang fp contract(fast)` helper
for the vector multiply/subtract expression. Other expressions retain
`-ffp-contract=off`; this is not global fast-math or reduced-precision math.
For finite GGUF scales, the half-scale times integer-scale times quantized
level product is exactly representable in FP32, as established by the earlier
[dequantization proof](QWEN35_GGUF_COUNTERS.md). Fusing its subtraction need not
change the rounded dequantized value.

Both working implementations pass forward and reverse sweeps on eight real
Q4/Q5 matrices, including the merged gate/up projection: **384 complete outputs
are byte-identical** to the qualified local-first runtime. Each microbenchmark
alternates reference/candidate calls, retains 100 samples after two warmups,
and compares three complete input/output patterns outside timing. Long-K
timings include separate partial reduction on both sides; the production
caller-owned-parts path does not. These are not full-model HF tests.

The compiler's post-register-allocation machine IR reveals a substantial
difference between the two FMA expressions:

| Expression | Q4 single-part loop | Q4 four-part loop | Q5 loop |
| --- | ---: | ---: | ---: |
| Current local-first control | 169 | 169 | 183 |
| Vector metadata only | 165 | 166 | 180 |
| Scalar explicit dequantization FMA | 211 | 212 | 218 |
| Scalar FMA plus vector metadata | 181 | 182 | 229 |
| Vector contraction helper | 107 | 107 | 121 |
| Vector helper plus metadata | 103 | 104 | 118 |

Counts exclude `IMPLICIT_DEF` pseudo-operations. These are static instructions
in the largest self-loop, not final binary disassembly, dynamic instruction
counts or cycles. Fewer instructions do not establish lower execution time.

Scalar FMA regresses substantially: merged gate/up throughput ratios are
0.8423/0.8547, and FFN down 0.8570/0.8006 in the two orders. The vector helper
avoids that regression but does not establish a general win: merged gate/up
ratios are 0.9998/0.9966 and FFN down 0.9860/0.9963. Adding metadata helps some
shapes, with merged gate/up 1.0129/1.0017 and attention gate 1.0438/1.0219,
but attention Q is 0.9740/0.9988 and Q5 QKV changes direction. The identical
control also varies. No variant is promoted by selecting only favorable shapes
or rounds; all samples and unchanged-code controls remain available.

## Prepared Q6 projection reduction

The existing prepared-Q6 projection reduces four component accumulators across
a 32-lane subgroup before combining them locally. A separate candidate first
combines the four components locally and then performs one subgroup reduction.
This reduces source-level shuffle expressions from 20 to five per output row,
but changes FP32 summation order. It does not alter weight scaling, accumulation
within each component, Split-K partitions or their final combination order.
The already-regrouped output head is unchanged.

Two variants and both orders test the complete 248320-row head, 2560-by-9216
FFN down projection and 1024-by-2560 attention V projection. All **36 complete
outputs** pass the unchanged local tolerance. The 24 unchanged-control/head
outputs are byte-identical. Maximum absolute error is 0.000000953674 and maximum
normalized error is 0.000496941. This does not substitute for the three-case
HF trajectory gate, and no full-model promotion is attempted.

FFN down ratios are 1.0356/1.0082, but attention V changes from 1.0254 to
0.9431. The unchanged FFN control itself ranges from 0.9663 to 1.0569.
Read-only ELF section comparison proves that control's complete device fatbin
is byte-identical to the qualified runtime. Thus several-percent fluctuations
cannot be interpreted as code improvements. The candidate is left unpromoted.
Both variants report zero local-memory and spill bytes; register counts do not
by themselves establish occupancy or a bottleneck.

## Evidence

Under ETSupport `artifacts/gguf-20261029`:

- `packed-dequant-fma-probe-v1`: unsupported-builtin compile attempt, retained.
- `packed-dequant-fma-probe-v2/v3`: complete sources, native harnesses, builds,
  both microbenchmark orders, compiler resource logs and machine-IR dumps.
- `audit_packed_dequant_fma_v1.py`, `packed-dequant-fma-decision-v1.json`:
  independent hash, complete-output and instruction-count audit.
- `q6-local-reduction-probe-v1`: separate control/candidate source, builds,
  real-matrix complete-output comparisons and both timing orders.
- `audit_q6_local_reduction_v1.py`, `q6-local-reduction-decision-v1.json`:
  unchanged-device proof, resource and micro-output audit.
