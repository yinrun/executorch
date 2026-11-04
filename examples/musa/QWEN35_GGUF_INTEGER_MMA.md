# Integer matrix instructions for lossless input planes

This follows the [integer-dot](QWEN35_GGUF_LOSSLESS_DOT.md) and
[exact-residual](QWEN35_GGUF_RESIDUAL_DOT.md) investigations. It tests MP22
signed integer matrix multiplication rather than scalar DOT8 instructions.
The original Qwen3.5-4B Q4_K_M weights and FP32 input/state/output contract
remain unchanged. The qualified model runtime is not modified.

The resulting GEMV passes the recorded numerical checks only with an isolated
source workaround. All measured variants are substantially slower than the
existing packed FP32 kernel. There is no new model speedup or evidence of
stable 10 decode tokens/s.

## Primitive and representation

SDK 5.1 and the local MUTLASS MP22 traits expose a 128-thread
`__musa_imma_m32n32k32_mma` with signed byte operands and int32 accumulation.
A standalone test compares every output with independent host int64 primitive
arithmetic, not CPU model inference. Fixtures cover full signed byte ranges,
-128/127 boundaries, Q5-sized positive weights, sparse plane columns, identity
matrices, nonzero accumulator seeds, changing inputs, default/nonblocking
streams and output-tail canaries.

The base test passes 526336 integer outputs, two canaries and two invalid
calls. Four partial-output/sparse-operand configurations and two smaller
consumer-predicate controls each pass the same count. The seven configurations
total 3684352 comparisons on repeated fixtures, not that many unique inputs.
The smaller controls do **not** reproduce the full-GEMV failure below.

GEMV reuses the checked five-plane lossless input codec. Biased bytes become
signed packed digits through XOR 0x80808080, avoiding the SDK's problematic
signed-char DOT8 wrapper. Original Q4/Q5 levels are exact positive int8 values.
One matrix instruction computes 32 rows by 32 columns over 32 input elements;
five output columns hold the five plane dots. Other columns are unused.
Each individual dot is bounded by 32*31*128, preventing int32 overflow.

There are eight input groups per original 256-value weight block. One variant
reconstructs each FP32 group dot before applying weight scale/minimum; the
other accumulates plane contributions separately before final reduction.
Both change FP32 summation order. Lossless input encoding does not establish
byte-exact GEMV relative to the qualified native implementation.

## Observed code-generation failure

The initial real Q4 test fails on all 4096 outputs, with maximum absolute
error about 8.92324e6. Disabling MAX_ILP does not fix it. Both reduction modes
also fail a 128-row diagnostic. Instrumentation checking every integer product
makes both pass; duplicate plane columns, including averaging their identical
integer outputs, do not fix the uninstrumented implementation.

Compiler evidence narrows the failure without establishing a general SDK fix:

- Optimized LLVM IR places the collective before the output-consumer predicate.
  Its declaration has `nofree nosync nounwind readnone`, but no `convergent`.
  The installed intrinsic definition likewise specifies `IntrNoMem` without
  that property.
- In the original Q4 four-part kernel's post-register-allocation machine IR,
  `WMMA_SMMA_323232_MatC_0` appears in consumer block `bb.6`, after predecessor
  `bb.14` narrows the execution predicate. The instruction requires the full
  128-thread operand distribution, not just selected output consumers.
- One per-thread volatile shared-memory write of the first matrix result
  keeps the instruction in `bb.14`, before that narrowing. This changes its
  observable use, not its numerical formula. The modified GEMV passes the
  recorded checks.

This is evidence of unsafe collective movement in this compiled GEMV, not
proof that every partial-output primitive fails. The workaround uses 128
separate int slots, 512 bytes per CTA; no thread reads another thread's slot.
It remains isolated and is not a production or compiler-wide fix.

The SDK advertises `-disable-machine-sink` and
`-mtgpu-enable-machine-sink=false`. Neither individually nor together fixes
three tested Q4/Q5/Split-K geometries: all twelve default/flag-control runs
fail the unchanged numerical gate. No such flag is added to production.
An earlier empty-inline-assembly liveness attempt does not compile and is
retained as a failed attempt, not an accepted workaround.

## Micro results, not model acceptance

The workaround family tests eight real matrices, both reduction modes, a
native control and opposite traversal orders: 48 processes and 144 complete
output comparisons. All pass finite checks, the unchanged
`0.001 + 0.001*abs(reference)` tolerance and exact argmax. The 48 native-control
outputs are byte-exact. Maximum absolute candidate error is 4.768372e-6;
maximum normalized error is 0.002252410.

Its two libraries also pass 4194688 input-codec bit comparisons, 20 canaries
and six invalid-length calls. Adversarial representation tests cover codec
fallback. All timed GEMV inputs encode successfully: GEMV fallback/row-tail
qualification, complete text/image trajectories, HF gates and production
stream/workspace integration remain untested here.

Each process performs at least three seconds of alternating warmup and
retains 100 event pairs after the first two. Timing includes fresh encoding,
GEMV and Split-K reduction where applicable, but not workspace allocation.
No profiler, launch blocking, concurrent investigation build/audit or clock
changes are used. The desktop GPU remains shared. All 4800 frequency samples
are retained: 3257 at 730 MHz and 1543 at 1.4 GHz. Samples are not guarantees
of every event's clock.

Ratios are native time divided by candidate time; above one would be faster.
Each cell contains forward/reverse traversal measurements.

| Matrix | Combine each input group | Combine planes after accumulation |
| --- | ---: | ---: |
| Q4 attention gate | 0.1412 / 0.1405 | 0.1670 / 0.1674 |
| Q5 QKV | 0.1642 / 0.1819 | 0.1915 / 0.1630 |
| Q4 FFN gate | 0.1533 / 0.1532 | 0.1826 / 0.1523 |
| Q4 FFN down | 0.1505 / 0.1510 | 0.1850 / 0.1529 |
| Q4 merged gate/up | 0.1664 / 0.1647 | 0.1368 / 0.1638 |

All candidate ratios lie between 0.1368 and 0.2012; native controls span
0.9964-1.0066. No sample supports model integration or a new decode-rate claim.

## Instruction evidence and decision

Machine IR contains `WMMA_SMMA_323232` for the primitive and its `_MatC_0`
variant for zero-initialized GEMV. This confirms selection of integer matrix
operations, not merely an MMA-labelled filename. It is compiler evidence,
not final binary disassembly or measured tensor-core utilization.

The initial opcode parser handles uppercase names only, misses the mixed-case
`MatC` suffix, and fails its audit. The corrected parser reuses the same hashed
raw logs and confirms every expected matrix instruction. The initial failed
parser/audit is retained; no numerical gate is weakened.

The mapping uses only five of 32 output columns and repeats matrix work,
unpacking and metadata processing for each 32-element group. These motivate
a different tile layout, but are not measured attribution of the slowdown.
Volatile-store overhead is not independently isolated. Simply replacing DOT8
with these matrix instructions is rejected as an optimization.

The [group-tiled follow-up](QWEN35_GGUF_GROUP_TILED_MMA.md) tests a different
layout, separates fallback and adds synchronization controls. It retains
substantial slowdowns and new precision failures rather than promoting them.

Under ETSupport `artifacts/gguf-20261029`:

- `integer-mma-capability-v1`, `integer-mma-liveness-v1/v2`,
  `integer-mma-convergence-v1`: primitive controls and retained compile failure.
- `integer-mma-gemv-v1/v2/v3`: failing originals, MAX_ILP control and complete
  workaround codec/micro evidence.
- `integer-mma-diagnostic-v1`, `integer-mma-duplicate-v1`,
  `integer-mma-sinking-v1`: instrumentation, duplicate-column and flag controls.
- `integer-mma-ir-v1`, `integer-mma-machine-ir-v1`,
  `integer-mma-machine-ir-v2.json`: raw IR and corrected opcode extraction.
- `audit_integer_mma_v2.py`, `integer-mma-decision-v1.json`,
  `audit_integer_mma_followup_v1.py`, `integer-mma-followup-decision-v1.json`:
  independent numerical, timing, protected-hash and code-generation audits.
