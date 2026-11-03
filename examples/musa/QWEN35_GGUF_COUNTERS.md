# Native GGUF instruction and counter investigation

This follows the optional ILP scheduling experiment in
[QWEN35_GGUF_ILP.md](QWEN35_GGUF_ILP.md). The baseline is Qwen3.5-4B
Q4_K_M with FP32 activations/accumulation, losslessly predecoded Q6 weights,
FFN merging, and SDK 5.1 MP22 ILP scheduling. It is not a BF16 model.
The accepted deployment, PTE/PTD and numerical gate remain unchanged.

## Hardware counters

Moore Perf Compute 1.3.0 captures one selected real-matrix GEMV launch using
`SpeedOfLight,LaunchStats,MemoryWorkloadAnalysis`, with three application
replay passes. These are independent profiling runs, not whole-model timing
or an exclusive-device experiment. MP22 does not expose all MP31 DDR metrics.

| Selected ILP kernel | Compute(MP) throughput | L1 throughput | Memory pipes busy | L1 hit rate | Registers/thread |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4 FFN gate | 81.45% | 43.66% | 27.01% | 49.44% | 104 |
| Q4 FFN down | 73.63% | 38.97% | 14.68% | 50.07% | 104 |
| Q5 QKV | 92.17% | 51.43% | 30.11% | 62.34% | 108 |
| Q6 output head | 69.93% | 38.33% | 30.49% | 40.31% | 66 |

Compute(MP) is the tool's metric, not measured FLOP utilization. These
observations motivate investigating unpack/conversion instructions, but do
not prove that the entire model is compute-bound. Counters from different
replay captures cannot establish causal speedups. A narrower Q4 kernel has
63 registers/thread and 82.05% L1 hit rate yet regresses in ordinary alternating
microbenchmarks: lower register usage or a higher hit rate is not acceptance.

After stopping this investigation's GPU workloads and waiting 60 seconds,
five GPU-utilization samples read 0%, 70%, 70%, 0%, 70%. Existing desktop
processes hold device descriptors, but no usable per-process engine counters
were found. Open descriptors do not identify the source of utilization. No
user application was terminated and no clocks were changed. These samples
cannot justify extrapolating exclusive-device throughput.

## Rejected or unpromoted experiments

Q6 row tiles, Q4/Q5 vector widths, Q4/Q5 metadata predecoding, K-block unrolling
and explicit dequantization FMA collectively pass **516 complete byte-exact
micro-output comparisons**. None establishes a sufficiently consistent new
whole-model gain to replace the accepted runtime. Wider metadata increases
weight traffic; vectorizing its loads improves the prototype but does not
establish a broad win. Large unroll factors and larger Q6 tiles regress.

Explicit dequantization FMA receives additional checks: all 130,023,424
finite-half/integer scale products are exact in FP32, and 761,868 complete
synthetic output values match byte for byte across finite half encodings,
offsets and row tails. Eleven native GGUF tests pass. Whole-text ABBA/BAAB
benchmarks retain all original IDs/EOS, but independent median speed ratios
are 0.9991 and 1.0071. Combined medians of 7.0156 and 7.0589 tokens/s are not
treated as a stable gain. This candidate has **not** passed the full three-case
HF/logit trajectory gate and is not promoted.

Two isolated grouped-dot prototypes test four/eight/sixteen partial
accumulators. They share input sums and move scaling outside the per-element
dot product, changing FP32 rounding order. All **144 complete micro-output
comparisons** pass the local tolerance, but are not byte-identical and are
not full-model HF acceptance. Both implementations are slower, including
the input-sum launch cost; runtime workspace integration is not measured.
Neither modifies the deployment or relaxes its acceptance gate.

## Compiler machine IR

The SDK compiler can dump the post-register-allocation pass
`mtgpu-linear-scan-fence-setting`. This provides target machine IR even when
the installed objdump cannot disassemble the device object. It is not final
binary ISA disassembly, and static opcode counts are not dynamic cycles.

For `linear_vector_decode_kernel<12,16>`, the baseline K-block loop contains
32 scalar `MOV_S32_TO_F32_RTE_BST1` level conversions. An isolated unsigned
byte-vector conversion expression selects 32 `MOV_U8888_TO_F32_BST1`
operations instead. It replaces 24 shift/AND pairs with 24 combined
`BIT_AND_LSR_DST0_FFFF` operations. Excluding `IMPLICIT_DEF` pseudo-operations,
the loop has 239 versus 215 machine-IR operations. This is **not** evidence
of one instruction converting four bytes simultaneously: the conversion
count remains 32.

Four conversion variants, including an unchanged control, pass **132 complete
byte-exact micro-output comparisons** across eleven real Q4/Q5/Q6 matrices.
The unchanged control's device fatbin matches the accepted ILP runtime.
Q4/Q5-only conversion subsequently passes isolated full-model evaluation,
with a small reproducible gain described in
[QWEN35_GGUF_BYTE_CONVERT.md](QWEN35_GGUF_BYTE_CONVERT.md). Q6 conversion is not
selected. Microbenchmark ratios alone are not promotion.

## Local evidence

The local evidence root is ETSupport `artifacts/gguf-20261029`:

- `ilp-followup-experiments-v1.json` audits the 516 exact comparisons,
  dequantization proof, FMA native/text tests and six hardware-counter reports.
- `mcu-gemv-v1`, `mcu-gemv-v2`, `idle-gpu-observation-v1` retain raw profiles,
  launch arguments, replay metadata and shared-device observations.
- `grouped-q45-probe-v1`, `grouped-q45-probe-v2` retain the separate
  non-bit-exact, unpromoted grouped-dot experiments.
- `ilp-machine-ir-v1/comparison.json` retains per-basic-block opcode counts;
  `ilp-vector-convert-probe-v1` retains source, compilation and microbenchmarks.

The previously accepted ILP result is 7.030 median tokens/s in its recorded
normal shared-desktop series. The separately paired byte-conversion comparison
measures 7.014 versus 7.093 median tokens/s. **The 10 tokens/s target is not met.**
