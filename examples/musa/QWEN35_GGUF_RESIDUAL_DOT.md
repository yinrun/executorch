# Integer component plus exact FP32 residual

This follows the [four/five-plane lossless-input experiment](QWEN35_GGUF_LOSSLESS_DOT.md).
It tests a different Q4/Q5 GEMV representation, not a precision downgrade:
one or two integer byte planes plus a separately preserved FP32 residual.
The original Qwen3.5-4B Q4_K_M weights, qualified runtime and model export
remain unchanged. All measured variants are slower than the native packed
FP32 reference. None is promoted, and stable 10 decode tokens/s is not proved.

## Representation and validation

For each 32-value input group, let `E` be its maximum biased FP32 exponent
and `P` the integer-plane count, either one or two. The encoding is

```text
scale = 2^(E - 124 - 8*P)
integer = trunc(x / scale)
coarse = integer * scale
residual = x - coarse
x = coarse + residual
```

The device obtains the truncated integer from the mantissa and exponent bits,
without division or rounding-mode assumptions. Maximum integer magnitudes are
63 and 16383 for one and two planes. Balanced signed byte digits are stored
with unsigned bias 128. Coarse multiplication is exact. For nonzero coarse,
truncation leaves the operands within a factor of two, making the subtraction
exact under the eligible normal-value conditions; zero coarse preserves `x`
as the residual. This reasoning is supplemented by GPU bitwise reconstruction
checks before marking any group eligible.

Eligibility retains `44 <= E <= 220`, excludes subnormals, negative zero and
nonfinite inputs, and rejects a group if any reconstructed input differs.
Ineligible groups use the original FP32 inputs directly in GEMV. Decoder
fallback copies original bits, preserving signed zero and NaN payloads rather
than asserting bitwise stability of arithmetic NaNs. No residual is dropped.

An independent host oracle computes truncation and residuals through double
arithmetic, checks the stored integer and residual bits, reconstructs the
input, and independently checks every eligibility flag. It is a representation
test, not CPU model inference. Fixtures retain mixed exponents, random bit
patterns, zero signs, finite extrema, subnormals, infinities, NaNs and eligibility
boundaries. The 32771-group input exercises a partial final thread block.

Across four compiled libraries, two changing input sets each yield 8389376
bitwise reconstruction checks, 40 output-buffer tail canaries and 12 invalid
length calls. Each library encodes 36944 groups and falls back on 28598 groups
across those two adversarial sets. These are repeated fixture comparisons,
not that many unique values or a model-activation fallback measurement.

## GEMV and measured cost

The new kernel computes unsigned DOT8 integer contributions with the checked
bias correction, and four FP32 partial accumulators compute the residual dot
product. It then applies original group scales/minima. The reduced number of
integer planes does not eliminate residual loads, conversions or FMAs.
Regrouping changes FP32 operation order: lossless input encoding is not
byte-exact GEMV relative to the qualified native reference.

Workspace contains four output-part arrays, `P*K` digit bytes, `4*K` residual
bytes, and group scales, sums and flags. Allocation is outside the micro timing;
every timed candidate invocation includes fresh input encoding, residual
generation, GEMV and, for long Q4 inputs, the four-part final reduction.
There is no claimed production workspace/stream integration.

The first family tests eight real matrices, three variants including a native
control, and both traversal orders: 48 processes and 144 complete output
comparisons. The second tests four representative matrices with word-loop
unroll factors eight, one and two: 24 processes and 72 complete outputs.
All 216 comparisons pass finite checks, `0.001 + 0.001*abs(reference)` and
exact argmax. The 48 native-control outputs are byte-exact. The second family
also matches the original one-plane residual implementation byte for byte
on all 72 outputs; this is not exactness to the native FP32 reference.
Maximum absolute micro error is 0.00000190735; maximum normalized error is
0.00161124. No full-model HF/logit or text/image trajectory gate is run.

Each process performs at least three seconds of alternating active warmup,
then retains 100 alternating event pairs after discarding the first two.
There is no profiler, launch blocking, clock change or competing workload
from this investigation during ordinary timings. Existing desktop activity
remains. All 7200 frequency samples are retained: 6471 at 1.4 GHz, 720 at
730 MHz and nine at 1.11 GHz. Frequency samples are not interval-wide clock
guarantees, and control ratios also vary on this shared device.

Ratios below are native time divided by candidate time; greater than one
would be faster. Each cell gives forward/reverse traversal measurements.

| Matrix | One integer plane + FP32 residual | Two integer planes + FP32 residual |
| --- | ---: | ---: |
| Q4 attention gate | 0.8788 / 0.8435 | 0.7243 / 0.6791 |
| Q5 QKV | 0.8787 / 0.8516 | 0.6702 / 0.6448 |
| Q4 FFN gate | 0.8624 / 0.8038 | 0.6358 / 0.6417 |
| Q4 FFN down | 0.8449 / 0.8451 | 0.6424 / 0.5968 |
| Q4 merged gate/up | 0.8256 / 0.8282 | 0.6103 / 0.6008 |

Reducing word-loop unrolling does not recover a win. In the second family,
one-plane ratios span 0.7220-0.9022 with full unrolling, 0.3963-0.4719 with
unroll one, and 0.5781-0.7067 with unroll two. These families are separate
measurement series, not paired evidence for cross-family speed differences.

## Stage and compiler diagnostics

Separate merged-gate/up diagnostics retain three complete output comparisons
and input reconstruction per run, in addition to the 216 comparisons above.
They time encoding alone after the complete-bundle timing phase:

| Integer planes | Sampled clock in both phases | Native median ms | Full candidate median ms | Encoding-only median ms |
| --- | ---: | ---: | ---: | ---: |
| One | 730 MHz | 1.10200 | 1.48334 | 0.09254 |
| Two | 1.4 GHz | 0.77948 | 1.30592 | 0.06462 |

These separate phase medians are not additive, and the different clocks
preclude comparing one-plane and two-plane absolute times as a code effect.
They do not justify treating encoding elimination as a measured model gain.

Post-register-allocation machine IR confirms 16 DOT8 instructions in each
fully unrolled one-plane GEMV function, versus 24 for two planes and 40/48
in the previous four/five-plane design. The one-plane implementation still
contains 493-537 whole-function instructions after excluding `IMPLICIT_DEF`,
including fallback; these are neither final ISA nor dynamic cycle counts.

For Q4 single-part GEMV, reducing word-loop unrolling from eight to two to
one reduces temporary registers from 113 to 63 to 53. All variants report
zero local memory/spill bytes. Smaller static loop bodies contain four/two
DOT8 instructions but execute more iterations: they do not perform fewer
integer operations per group. Their worse measured timings reject register
count or static instruction count alone as an optimization criterion.

## Decision and evidence

The independently audited result rejects these implementations for deployment.
This narrows the integer-dot approach without changing numerical tolerances,
removing output checks, replacing weights, enabling TF32, or claiming a new
end-to-end speed record. The qualified runtime remains untouched.

The subsequent [integer matrix-instruction experiment](QWEN35_GGUF_INTEGER_MMA.md)
checks collective signed-int8 execution, a specific code-generation hazard,
and the cost of computing lossless input planes with matrix instructions.

Under ETSupport `artifacts/gguf-20261029`:

- `residual-dot-probe-v1/v2`: codecs, independent tests, Q4/Q5 kernels,
  complete warm micro timing, build commands and protected hashes.
- `residual-dot-stage-probe-v1`: separate encoding-only diagnostic results.
- `dump_residual_dot_ir_v1.py`, `residual-dot-machine-ir-v1.json`: compiler
  entry-point-scoped machine IR and resource evidence for all four libraries.
- `audit_residual_dot_v1.py`, `residual-dot-decision-v1.json`: independent
  protected-hash, test coverage, timing-median and compiler-evidence audit.
