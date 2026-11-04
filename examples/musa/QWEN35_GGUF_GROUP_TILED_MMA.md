# Group-tiled integer MMA follow-up

These isolated probes follow the [integer matrix investigation](QWEN35_GGUF_INTEGER_MMA.md).
They retain Qwen3.5-4B Q4_K_M weights, exact five-plane FP32 input encoding,
FP32 accumulation, original-FP32 fallback and the unchanged numerical gate.
No kernel here is integrated into the qualified runtime. Stable 10 decode
tokens/s remains unproved.

## Eight-group tile

The first layout maps four output rows times eight quantization groups onto
the 32 matrix rows. Matrix columns encode group and input byte plane, using
either five adjacent columns per group or eight padded columns per group.
Both need two matrix instructions per original 256-element weight block.
This reduces loop iterations per CTA, but doubles matrix instructions per
output row: two per four rows, versus eight per 32 rows in the previous probe.
It is not a reduction in total matrix work.

Both layouts complete the forward/reverse eight-matrix sweep: 48 processes,
144 full-output comparisons and byte-exact native controls. Input encoding,
GEMV and any separate Split-K reduction are included in timing. The merged
gate/up native/candidate ratios are 0.09541/0.09570 for compact planes and
0.09525/0.09531 for padded planes, about ten times slower than the reference.
The padded layout reports 122 temporary registers and 28/36/48 spill bytes
for Q4 single-part, Q4 four-part and Q5 respectively. These are compiler
resource reports, not a causal explanation of runtime or numerical behavior.

## Two- and four-group tiles

The next layout uses two or four quantization groups, with eight columns per
group. It preserves the previous total of one quarter matrix instruction
per output row per original weight block. Four groups fail the unchanged
precision gate after earlier forward checks pass. The whole sweep is failed;
successful partial timings are retained, not presented as qualification.

Separating fallback into its own GPU kernel reduces the fast kernel to
67--69 temporary registers and zero spills, but does not fix the failure.
Explicit CTA barriers immediately before and after each matrix instruction
also do not fix it. All stages remain on the supplied stream. No tolerance,
finite check, argmax check or input encoding condition is weakened.

| Probe | Failure | Maximum absolute error |
| --- | --- | ---: |
| Inline fallback, four groups | Reverse merged gate/up | 0.281058 |
| Separate fallback, four groups | Forward attention gate | 0.298724 |
| Separate fallback plus barriers | Forward attention gate | 0.140014 |

The failure's root cause is not established. In particular, neither spill
removal nor added synchronization proves that spilling or reconvergence caused
it. The prior compiler-predication observation is not sufficient to diagnose
this different mapping. These failed kernels must not be used by the model.

Across all four probes, 192 complete output comparisons pass before the three
retained failures; this aggregate is not a passing test suite. Codec tests
include fallback inputs, but timed GEMV inputs all use eligible encoding
groups. Thus they do not qualify fallback GEMV arithmetic, arbitrary row
tails or production stream behavior. The very poor performance also argues
against further model integration of this family.

## Evidence

Under ETSupport `artifacts/gguf-20261029`:

- `integer-mma-group-tile-v1`: complete eight-group compact/padded sweep.
- `integer-mma-group-tile-v2`: smaller tiles and retained reverse-order failure.
- `integer-mma-group-tile-v3`: separated fallback and retained failure.
- `integer-mma-group-tile-v4`: added barriers and retained failure.
- `audit_group_tile_and_fused_attention_v1.py` and
  `group-tile-and-fused-attention-decision-v1.json`: protected hashes, complete
  results, partial sweeps, raw timing samples and failure audit.
