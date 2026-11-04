# Packed GEMV input reuse

This follows the [packed-loop and Q5-cache investigation](QWEN35_GGUF_PACKED_LOOP_AND_PLANES.md).
The global blocking profile identifies packed Linear as the dominant remaining
cost. The current experiment targets repeated activation loads within the
packet/quad GEMV, rather than changing the quantized weight layout or reducing
precision. The baseline is the qualified source-linked fused-prefix-attention
runtime. The separate Q5 cached-load and Q6 head-unroll candidates are not added.

The model remains Qwen3.5-4B Q4_K_M, with FP32 activations, state and
accumulation. No TF32, requantization, MTP, vocabulary pruning, tolerance change,
model re-export or output-check removal is used. These are isolated prototypes;
there is not yet an integrated input-reuse CMake option or deployment change.

## Mechanism and controls

The existing 128-thread block assigns eight lanes to each of sixteen output
rows. Each lane repeatedly loads its activation vectors while reading different
packed weights. Four variants retain the same weights, dequantization, explicit
FMA order, local-first reduction and Split-K partition boundaries:

| Mode | Activation access |
| --- | --- |
| 0 | Original global/read-only loads |
| 1 | Stage the current K partition once per block in shared memory |
| 2 | Stage that partition with its four-float vectors transposed |
| 3 | Load from the first eight lanes of each warp and broadcast to the other rows |

Modes 1 and 2 cooperatively stage the inputs and execute one block barrier
before the row-tail guard. No staging kernel or persistent global scratch is
introduced. Each four-part CTA stages only its own partition, including an
uneven last partition. For mode 2, the vector index within each 256-element
block changes from `quad` to
`(quad / 32) * 32 + (quad % 4) * 8 + (quad % 32) / 4`.
The global weight bytes do not change. The largest activation staging request
among the eight real microbench matrices is 10240 bytes per block.

This reduces repeated source-level input loads, not a claim of a proportional
DRAM-traffic reduction: the original read-only path can already cache or combine
loads. Shared-memory layout, register allocation and synchronization can also
hurt performance. The experiment keeps those costs inside the timed kernel.

## Microbenchmarks

Eight real Q4/Q5 matrices, four variants and forward/reverse orders produce
64 processes. Each retains at least three seconds of active warmup, two initial
warmup pairs and 100 alternating baseline/candidate timing pairs. Three changing
inputs per process produce **192 complete byte-identical outputs**; all **1536
output canaries** pass. The mode-0 device fatbin reproduces the qualified control
exactly. No clock, priority, affinity or desktop-application change is made.

| Matrix | Rows x columns | Mode-2 first / reverse speed ratio |
| --- | --- | --- |
| Attention gate, Q4 | 4096 x 2560 | 1.0180 / 1.0129 |
| QKV, Q5 | 8192 x 2560 | 1.2183 / 1.2150 |
| FFN gate, Q4 | 9216 x 2560 | 1.2170 / 1.2195 |
| SSM output, Q4 | 2560 x 4096 | 1.1767 / 1.1593 |
| Attention K, Q4 | 1024 x 2560 | 0.8830 / 0.8795 |
| Attention Q, Q4 | 8192 x 2560 | 1.1943 / 1.1976 |
| FFN down, Q4 | 2560 x 9216 | 1.0655 / 1.0667 |
| Actual merged gate/up, Q4 | 18432 x 2560 | 1.2283 / 1.2214 |

Ratios above one favor the candidate. The small attention-K regression is kept,
not hidden by excluding its shape. Long-K microbenchmarks include a separate
partial reduction on both sides; the exported caller-owned-parts model does not.
Mode 2 is selected uniformly for the subsequent model experiment, without a
shape-specific exception. This is not a claim that it wins at every shape.

The plain shared layout is inconsistent: merged gate/up ratios are 1.1144 and
0.8962, with all sampled frequencies at 1.4 GHz and 730 MHz respectively. The
warp-broadcast variant regresses on all eight matrices in both orders. Across
the complete sweep, all 6400 timing pairs remain: 4014 frequency observations
read 730 MHz and 2386 read 1.4 GHz. These observations are outside the events
and do not exclude intra-pair changes or establish a frequency-based cause.

| Mode | Q4 single-part temp registers | Q4 four-part | Q5 |
| --- | ---: | ---: | ---: |
| Original | 109 | 111 | 111 |
| Plain shared | 100 | 103 | 109 |
| Transposed shared | 103 | 106 | 109 |
| Warp broadcast | 109 | 111 | 119 |

All report zero local-memory and spill bytes. These compiler counts are not
measured occupancy. Native timings, not the register reduction alone, select
the transposed candidate.

## Isolated model qualification

Only the GGUF device object is replaced in the qualified source-attention link.
Its complete device image matches the measured mode-2 image exactly. Other
link inputs, runner, extension, PTE/PTD and protected runtimes remain unchanged.
The isolated library SHA-256 is
`8e14a4bb5e24a4b07ba629b86bbc0b8dc58cd9a342fffd259554baf65bb372ea`.

The existing native executables are run against that library with verified
link resolution: 17 GGUF and 15 attention tests pass; one second-device test
skips. This includes offsets, tails, uneven partitions, calling streams and
in-place input/weight updates. It is not a new four-state CMake build test.

Text, single-image and multi-image trajectories each run twice with reset.
All **2298 complete 248320-float outputs** are byte-identical to the qualified
source-attention runtime. The **1149 reset pairs**, original IDs/EOS, finite
checks, exact argmax and unchanged `0.001 + 0.001 * abs(reference)` frozen-HF
bound pass. Eighteen negative controls are rejected. Maximum absolute HF errors
remain 0.0000262260, 0.0006752014 and 0.0008490086 respectively.

Prepared image features and frozen HF references are reused; neither the raw
image frontend nor HF is rerun. Independent auditing checks every raw buffer,
reference hash, runtime mapping, native test, compile command and the exact
single-object link substitution. This does not establish arbitrary-shape,
long-context or concurrent-service acceptance.

## Ordinary full-generation timing

The original runner/export uses 28 prompt tokens, 89 generated IDs including
EOS and 88 decode steps. Eight processes run ABBA then BAAB, each with one
complete warmup and two measured generations: eight samples per variant.
There is no profiler, launch blocking or logit dumping during timing. The
agent only waits; the desktop GPU remains shared. All original trajectories,
warmups and slow observations are retained and independently audited.

| Variant | Minimum | Median | Maximum decode tokens/s | Serial-prefill median |
| --- | ---: | ---: | ---: | ---: |
| Source-attention baseline | 8.6564 | 9.9668 | 10.6293 | 9.9796 |
| Transposed-input candidate | 8.8133 | 11.0429 | 12.1618 | 10.5444 |

The pooled median ratio is 1.10796, but separate ABBA and BAAB ratios are
1.00592 and 1.09880. One candidate process measures 8.8133 followed by 12.1618;
both are kept. The BAAB candidate samples all exceed 11, but the complete
series does not meet a minimum-based ten-token/s criterion. The pooled ratio
is not claimed as a causal 10.8% speedup on this shared GPU. The 12.1618 peak
is an ordinary full-generation observation, not a guaranteed deployment rate.
Serial prefill is not batched prefill. Same-process full-generation pairing
remains a useful next check before attributing a robust model-level gain.

## Refreshed global profiling

A separate nonblocking `msys --trace=musa,osrt` capture covers loading, serial
prefill, one complete warmup and one complete measured generation. Original
IDs/EOS and runtime maps pass. All 232 execution windows retain 788 kernel
labels, including 75 single-part Q4, 48 four-part Q4, 24 Q5 and eight fused
attention launches. Stream synchronizations remain twelve per step; async
allocations/frees remain zero. The input staging does not add a kernel.

Across 88 measured decode steps, median host execution is 130.911 ms, compute
interval union is 116.775 ms and the post-execution validation gap is 0.850 ms.
These are profiled intervals, not ordinary throughput. Asynchronous labels
can overlap and must not be summed to infer individual kernel time.

A separate blocking capture covers 28 prefill and eight decode steps, preserving
the first nine original IDs. Its 788 kernel intervals per window are distinct
and nonoverlapping, allowing bounded per-kernel attribution. It has no complete
generation warmup and is not an end-to-end performance benchmark.

| Blocking-profile category | Calls per step | Median aggregate ms |
| --- | ---: | ---: |
| Packet Q4, single part | 75 | 41.6702 |
| Packet Q4, four parts | 48 | 15.8601 |
| Grouped Q6 output head | 1 | 15.3247 |
| Prepared Q6, four parts | 16 | 12.8548 |
| Packet Q5 | 24 | 11.5461 |
| GDN pointwise kernel `_9` | 24 | 3.7476 |
| GDN reduction kernel `_10` | 24 | 2.4445 |
| GDN reduction kernel `_7` | 24 | 2.3642 |

Median total kernel time is 118.518 ms, with packed Linear at 99.697 ms and
other kernels at 18.832 ms. Category medians need not sum exactly. Packed
Linear still accounts for roughly 84% of kernel time. Host-window median is
223.583 ms and compute union is 118.612 ms; CPU waits overlap GPU work and
must not be added. The older source-attention blocking capture reports
125.259 ms total and 106.431 ms packed Linear, but these separate captures
do not establish a causal cross-trace speedup.

An independent audit reopens both SQLite traces, verifies the selected PID,
hashes, maps, original trajectories and all 268 execution windows, recomputes
compute unions, checks launch/synchronization counts, and rechecks the blocking
per-kernel sums and nonoverlap. No ordinary rate is inferred from the traces.

## Decision and evidence

The transposed-input prototype merits same-process pairing and a guarded,
default-off source implementation with fresh build-toggle, native and model
qualification. The plain-layout and warp-broadcast variants are not selected.
Further work must retain the small-shape regression and low ordinary samples,
and distinguish a source build from this isolated prototype. **Stable minimum
decode throughput of ten tokens/s remains unproven.**

Under ETSupport `artifacts/gguf-20261029`:

- `packed-input-reuse-probe-v1`, `audit_packed_input_reuse_v1.py`,
  `packed-input-reuse-probe-v1-decision.json`: four source/build variants,
  resources, all 64 processes, complete outputs/canaries and unfiltered pairs.
- `build_packed_input_reuse_v1.py`, `build-packed-input-reuse-v1`,
  `validate_packed_input_reuse_v1.py`, `packed-input-reuse-validation-v1`,
  `audit_packed_input_reuse_runtime_v1.py`,
  `packed-input-reuse-runtime-decision-v1.json`: isolated link, native tests,
  complete model outputs and independent precision/provenance audit.
- `benchmark_packed_input_reuse_v1.py`, `packed-input-reuse-benchmark-v1`,
  `audit_packed_input_reuse_ordinary_v1.py`,
  `packed-input-reuse-ordinary-decision-v1.json`: ordinary ABBA/BAAB timing,
  all lows, original trajectories and protected-file hashes.
- `global-packed-input-reuse-profile-v1`,
  `packed-input-reuse-blocking-profile-v1`,
  `audit_packed_input_reuse_profiles_v1.py`,
  `packed-input-reuse-profile-decision-v1.json`: full nonblocking and short
  blocking traces, interval/count analyses and independent raw-trace audit.
