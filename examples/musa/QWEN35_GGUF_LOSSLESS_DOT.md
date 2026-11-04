# Lossless input encoding and integer GEMV prototype

This follows the [integer primitive feasibility check](QWEN35_GGUF_LANE_WIDTH_AND_DOT.md).
It implements a new isolated Q4/Q5 GEMV, but does **not** replace the qualified
Qwen3.5-4B Q4_K_M runtime. Input encoding passes bitwise reconstruction tests;
the regrouped GEMV passes micro numerical checks but is slower than the
existing packed FP32 path. No full-model qualification or new decode rate is
claimed. Stable 10 decode tokens/s remains unproved.

## Lossless representation, not dropped precision

Each group of 32 FP32 input values shares a power-of-two scale. A normal
FP32 value has integer mantissa `m`, biased exponent `e` and sign. For `P`
byte planes, the prototype uses `padding = 8 * P - 26` and the group's maximum
biased exponent `E`. It constructs a signed integer

```text
I = sign * m * 2^(e - E + padding)
x = I * 2^(E - 150 - padding)
I = sum(digit[p] * 256^p), with -128 <= digit[p] <= 127
```

Both four- and five-plane variants are tested. Left shifts preserve every
mantissa bit. A right shift is allowed only when every discarded bit is zero.
The signed integer is decomposed exactly into balanced base-256 digits, stored
as unsigned bytes with bias 128. Integer magnitudes stay below `2^(8*P-2)`;
decomposition uses int64 without overflowing or relying on signed right shift.

The fast encoding is restricted to `44 <= E <= 220`. Nonfinite values,
subnormals, negative zero, exponent spreads that cannot be represented exactly,
or failed reconstruction checks invalidate the entire 32-value group. Original
FP32 inputs remain available. The GEMV then computes that group's contribution
directly from the original FP32 values; it does not discard a residual or
silently substitute approximate quantization.

Every potentially encoded input is reconstructed on GPU and compared by its
FP32 bit pattern before the group is marked valid. The independent test also
rebuilds encoded integers from the stored bytes and reconstructs through host
double arithmetic. Host arithmetic is an encoding-test oracle, not CPU model
inference. Bitwise fallback tests preserve signed zero and NaN payloads by
copying the original bits, not by claiming arithmetic NaNs are byte-stable.

Across four compiled libraries, two changing 1048672-value input sets each
produce **8389376 bit-pattern comparisons**, 40 output-tail canary checks and
12 invalid-length checks. Fixtures include ordinary normals, mixed exponents,
zero signs, normal/subnormal boundaries, finite extrema, infinities and NaNs.
The 32771-group length exercises a partial final thread block. Every group
flag is checked against the independent host eligibility calculation, so an
implementation that always falls back cannot pass as an encoded fast path.

## Integer projection

Weights retain their original Q4/Q5 packed bytes and scale/minimum metadata.
The new kernel uses only the verified **unsigned** `__dp4a` overload. For each
plane, the signed contribution is obtained exactly as

```text
dot(level, signed_digit) = unsigned_dot(level, biased_digit) - 128 * sum(level)
```

Each unsigned dot sum is at most `32 * 31 * 255 = 252960`; the integer
accumulators and bias correction fit safely in int32. Plane contributions are
converted and combined in FP32, then the weight scale/minimum is applied to
the group dot and the original-input group sum. This changes the floating-point
operation order relative to dequantizing each weight and accumulating it.
**Lossless inputs do not imply byte-identical GEMV outputs.**

There is no use of the SDK's problematic plain-char signed wrapper, no global
`-fsigned-char` change, TF32, lower-precision weight conversion, vocabulary
screening or relaxed tolerance. All input preparation and inference arithmetic
run on MUSA. Noneligible groups take a GPU FP32 branch in the same kernel.

The caller supplies workspace for partial outputs and encoded inputs. Every
timed candidate invocation includes input encoding, GEMV, and the separate
partial reduction for long-K Q4. Workspace allocation is outside micro timing;
this is not a measurement of integration into the existing model allocator
or caller-owned-parts export.

## Numerical and timing results

The first family compares the existing standalone FP32 control, four-plane
integer projection and five-plane projection. The second compares the original
five-plane kernel with explicit packet loads, with and without `__ldg` caching.
Both families test eight real matrices in forward/reverse orders, three
changing FP32 inputs per process, at least three seconds of active warmup,
and 100 recorded alternating timing pairs after two initial pairs.

All **288 complete output checks** satisfy the unchanged
`0.001 + 0.001 * abs(reference)` micro gate, with finite values and exact argmax.
Maximum absolute differences are 0.00000286102 and 0.00000238419; maximum
normalized errors are 0.00230176 and 0.00195650, respectively, below the limit
of one. The 48 original-control outputs are byte-identical to the qualified
native reference. The packet family additionally checks all 144 outputs
byte-for-byte against the first five-plane integer prototype, not the native
FP32 baseline. Input reconstruction is checked again on every micro input.

Representative reference/candidate median-time ratios, first/reverse order:

| Matrix | Four planes | Five planes | Five planes, cached packets |
| --- | ---: | ---: | ---: |
| Q5 QKV | 0.6144 / 0.6029 | 0.7811 / 0.7770 | 0.7913 / 0.7376 |
| Q4 FFN gate | 0.6134 / 0.5934 | 0.6986 / 0.7057 | 0.7253 / 0.6897 |
| Q4 SSM output | 0.6662 / 0.6616 | 0.7650 / 0.7574 | 0.8033 / 0.7924 |
| Q4 FFN down | 0.6115 / 0.6326 | 0.7081 / 0.7134 | 0.7265 / 0.7420 |
| Q4 merged gate/up | 0.5836 / 0.5720 | 0.6911 / 0.6812 | 0.6915 / 0.6842 |

A ratio above one would favor the candidate; no new integer configuration
establishes a win. Each ratio has its own alternating native control, and
separate families must not be treated as directly paired measurements. All
eight matrices, unchanged controls and low samples remain in the reports.
The device is shared and unlocked. Across 9600 timing pairs, 8676 frequency
samples read 1.4 GHz and 924 read 730 MHz; none is discarded.

For the three 2560-column sinusoidal inputs, four planes encode 72/72/66 of
80 groups, while five planes encode all 80 groups. Thus the five-plane
regression is present even without fallback on these inputs. This is a micro
input distribution, not measured full-model activation coverage. The codec's
deliberately adversarial fixture fallback rate must not be used as a model
fallback estimate either.

## Stage and compiler diagnostics

A separate same-process diagnostic measures encoding alone after the complete
candidate timing phase, using the merged 18432-by-2560 matrix:

| Diagnostic | Native reference, ms | Candidate including encoding, ms | Encoding-only median, ms |
| --- | ---: | ---: | ---: |
| Original five-plane | 0.909 | 1.295 | 0.078 |
| Cached-packet five-plane | 0.906 | 1.330 | 0.124 |

All sampled frequencies in both phases are 1.4 GHz. These are sequential
diagnostic phases on a shared device, not additive component timings or
normal model throughput. They do not establish the cause of the difference
between the two encoding-only measurements. They provide no evidence that
simply removing the preparation launch would make this design a broad winner.

Compiler machine IR selects 40 `DOT8_DST0` operations in each four-plane
integer kernel and 48 in each five-plane kernel. The original scalar-looking
load expressions already produce aggregate `LD_B1024`/`LD_B256` instructions.
Explicit uncached packets do not reduce the whole-function static count;
cached packets select more smaller loads. Counts include alternate branches
and are not dynamic instruction counts, final disassembly or stall analysis.
Every candidate reports zero local memory and spill bytes.

The first IR parser mistakenly counted three generated `_ConstCalcProg`
helpers as additional integer kernels. Its completed compile and failed parser
report are retained. The corrected parser keeps those helpers separately and
requires the three analyzed kernel names to match compiler entry-point metadata.
No numerical/performance gate is weakened to address that analysis error.

## Decision and evidence

Neither family is integrated into production or reexported into a model.
Full text/image trajectories, HF/logit qualification, native stream/lifetime
coverage and ordinary end-to-end timing would still be required for promotion.
Micro correctness alone is not the optimization objective.

A smaller integer component with an exactly preserved FP32 residual is a
different possible design, not an accepted result here. Its reconstruction,
fallbacks, residual arithmetic and complete preparation cost would need fresh
validation. This result does not authorize dropping low planes or loosening
the original model gate to obtain speed.

Under ETSupport `artifacts/gguf-20261029`:

- `lossless-dot-probe-v1/v2`: codec, independent fixtures, kernels, harnesses,
  builds, protected hashes and both complete timing orders.
- `lossless-dot-stage-probe-v1`: separate encoding-only stage diagnostics.
- `lossless-dot-machine-ir-v1.json`: retained initial parser failure.
- `dump_lossless_dot_ir_v2.py`, `lossless-dot-machine-ir-v2.json`: corrected
  entry-point-scoped compiler analysis, including raw IR logs.
- `audit_lossless_dot_v1.py`, `lossless-dot-decision-v1.json`: independent
  hash, coverage, resource, numerical and unfiltered-sample audit.
