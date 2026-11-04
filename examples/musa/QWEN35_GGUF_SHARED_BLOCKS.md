# Shared-input block grouping

This follows [vector dequantization](QWEN35_GGUF_SHARED_DEQUANT.md).
The combined runtime's blocking profile still attributes about 83% of kernel
time to packed Linear. This experiment changes the number of independent
eight-lane output-row groups per shared-input CTA. It does not change the
per-row FMA sequence, local-first reduction, Split-K boundaries, weight format,
FP32 precision or shared-memory capacity guard.

The workload remains Qwen3.5-4B Q4_K_M on M1000 MP22 with FP32 activation,
state and accumulation. No TF32, requantization, MTP, pruning, tolerance
relaxation or output-check removal is used. These are isolated prototypes;
the source's default block size and deployment runtime are unchanged.

## Mechanism

| Mode | Threads per shared-input block | Output rows per block |
| --- | ---: | ---: |
| 0 | 128 | 16 |
| 1 | 64 | 8 |
| 2 | 256 | 32 |
| 3 | 512 | 64 |

Only the shared-input row index and its launch grid/block dimensions change.
Staging still loads each K partition once per CTA and synchronizes all threads
before checking the row bound. Dynamic shared bytes per CTA do not change.
The original non-shared capacity fallback remains at 128 threads, and Q6 and
other kernels are unchanged. Kernel-launch count is not reduced by reducing
the number of CTAs within a launch.

For example, the 18432-by-2560 merged Q4 projection stages 10 KiB per CTA.
The 128/256/512-thread layouts use 1152/576/288 CTAs, implying 11.25/5.625/2.8125
MiB of logical activation staging reads. These are source-derived logical
loads, not measured DRAM traffic or proof of the reason for a speedup: caches,
occupancy, shared-memory access and scheduling also matter.

The earlier [global-input block-grouping probe](QWEN35_GGUF_PACKED_COUNTER_FOLLOWUP.md)
did not establish a broad improvement. The present experiment instead tests
the shared-input plus vector-dequantization path; it does not reclassify that
older negative result.

## Paired microbenchmark

Eight real matrices, four modes and both traversal orders produce 64 processes.
Each process retains at least three seconds of active warmup, 100 alternating
timing pairs, three complete output comparisons and 24 guard checks. Independent
auditing verifies all 192 complete finite outputs byte for byte, all 1536 guards
and all 6400 timing pairs. The 128-thread control's complete device image
matches the qualified vector-dequantization source ON image.

Ratios are baseline median divided by candidate median, forward / reverse.
The baseline is the qualified 128-thread combined source runtime:

| Matrix | 128-thread control | 64 threads | 256 threads | 512 threads |
| --- | ---: | ---: | ---: | ---: |
| blk.0.attn_gate.weight | 0.9918 / 1.1047 | 0.7209 / 0.6931 | 1.0183 / 1.1332 | 1.2098 / 1.2502 |
| blk.0.attn_qkv.weight | 0.9989 / 0.9991 | 0.6295 / 0.6312 | 1.0339 / 1.0350 | 1.0387 / 1.0386 |
| blk.0.ffn_gate.weight | 0.9999 / 0.9991 | 0.5995 / 0.6176 | 1.0890 / 1.0872 | 1.1106 / 1.0956 |
| blk.0.ssm_out.weight | 0.9893 / 1.0024 | 0.7318 / 0.6808 | 1.0162 / 1.0043 | 0.9989 / 0.9758 |
| blk.3.attn_k.weight | 0.9991 / 0.9965 | 0.8311 / 0.8535 | 1.2037 / 1.2074 | 1.0133 / 1.0044 |
| blk.3.attn_q.weight | 0.9994 / 0.9985 | 0.6281 / 0.6089 | 1.0936 / 1.0931 | 1.1107 / 1.1106 |
| blk.4.ffn_down.weight | 1.0004 / 1.0023 | 0.6288 / 0.6287 | 1.0791 / 1.0788 | 1.0929 / 1.0947 |
| blk.0.ffn_gate_up.merged | 1.0004 / 0.9986 | 0.5760 / 0.5770 | 1.1423 / 1.1431 | 1.1608 / 1.1614 |

The unchanged attention-gate control itself varies up to 1.1047, and that
variation is retained. All eight 256-thread shapes improve in both orders,
although the SSM-output gain is small. The 512-thread variant improves the
large FFN/Q projections more, but SSM output regresses and the attention-K gain
is much smaller than with 256 threads. The 64-thread variant regresses on all
eight shapes in both orders and is rejected. Both 256 and 512 are selected
for full-model qualification rather than selecting a shape-specific winner.

All variants retain Q4 single-part / Q4 four-part / Q5 temporary-register counts
104 / 107 / 110 with zero local-memory and spill bytes. All 6400 frequency
endpoint samples are retained: 1944 at 1400 MHz and 4456 at 730 MHz. These
endpoints do not establish constant frequency during a pair. No clocks,
priorities, governors or other desktop applications are changed. Long-K
microtimings include the same separate part-reduction launch on both sides;
the exported model retains its existing caller-owned parts path.

## Two isolated runtime qualifications

Each larger-block object is linked into a separate runtime by changing only
the GGUF device object in the qualified link. Its device image matches its
measured microbenchmark variant. The guarded-source native test binaries load
the selected runtime explicitly: both 256 and 512 pass all 18 GGUF and 15
attention tests, including partial-row, capacity, finite-half-scale and
calling-stream checks. One second-device restoration test skips per runtime
on this single-device system.

Each candidate independently passes the three complete model cases:

| Case | Complete output vectors per candidate | Reset pairs per candidate | Maximum absolute error vs frozen HF |
| --- | ---: | ---: | ---: |
| Text | 232 | 116 | 0.0000262260 |
| Single image | 856 | 428 | 0.0006752014 |
| Multiple images | 1210 | 605 | 0.0008490086 |

Both preserve every original generated ID and EOS, all finite logits and the
unchanged elementwise HF bound. Across the two candidates, 4596 complete
248320-element output vectors match the qualified baseline byte for byte,
with 2298 reset pairs and 36 negative controls. Independent audits reread all
raw outputs/references and verify loaded libraries, compile/link provenance,
native tests and protected artifacts. Prepared image features and frozen HF
outputs are reused; raw image frontend, HF inference, long-context and
concurrent-service qualification are not rerun or implied.

## Whole-model attribution

Each candidate has a separate blocking capture of 28 prefill and eight decode
steps, preserving the original nine-token prefix. Independent SQL audits cover
all 72 windows. Every step still has 788 distinct nonoverlapping kernels,
147 shared GEMVs (75 Q4 single-part, 48 Q4 four-part and 24 Q5), eight fused
attention kernels, twelve stream synchronizations and zero async allocation
or free. No original packed-GEMV fallback executes in these model traces.

| Blocking decode median (ms) | Prior combined 128-thread trace | 256 threads | 512 threads |
| --- | ---: | ---: | ---: |
| Total kernel time | 113.355 | 107.037 | 106.129 |
| Packed Linear aggregate | 94.370 | 88.070 | 87.137 |
| Other kernels aggregate | 18.965 | 18.948 | 18.992 |
| Shared Q4 single-part aggregate | 38.030 | 32.879 | 32.001 |
| Shared Q4 four-part aggregate | 15.134 | 14.390 | 14.369 |
| Shared Q5 aggregate | 10.671 | 10.239 | 10.197 |
| Q6 head | 15.326 | 15.326 | 15.325 |
| Q6 four-part aggregate | 12.767 | 12.767 | 12.763 |

These separate instrumented captures are attribution evidence, not a causal
ordinary-runner speedup. Independent medians need not sum exactly. The 256/512
host execution-window medians are 208.523/205.949 ms, with compute-track unions
107.122/106.222 ms; neither is ordinary throughput. Packed Linear remains the
largest target. The unchanged Q6 head and four-part projection kernels alone
account for about 28.1 ms in the 512-thread trace.

## Ordinary three-mode timing

The original uninstrumented runner compares the qualified 128-thread combined
source runtime with both candidates. Two symmetric orders, A-B-C-C-B-A and
C-B-A-A-B-C, use twelve processes. Every process retains one full warmup and
two measured generations, giving eight measured generations per mode. Every
generation keeps 28 prompt tokens, 89 generated IDs including EOS and 88 decode
steps. No profiler, launch blocking, frequency instrumentation or logit dumping
is enabled; the agent performs no competing work during timing.

| Runtime | Minimum | Median | Maximum decode tokens/s | Serial-prefill median |
| --- | ---: | ---: | ---: | ---: |
| 128-thread combined source | 9.0405 | 11.7315 | 12.3757 | 11.6738 |
| 256-thread candidate | 9.4551 | 10.0804 | 13.0320 | 9.8308 |
| 512-thread candidate | 9.7423 | 11.2605 | 13.0393 | 11.4645 |

All low samples and all original checks remain. The pooled median ratios
against baseline are only 0.85926 and 0.95985; the favorable peaks are not used
to replace these mixed ordinary results. In the first series, the 512-thread
median is 10.2257 versus baseline 10.5370; in the second it is 12.6859 versus
11.7459. Separate-process results on this shared GPU do not establish a causal
gain or regression. Serial prefill is not batched prefill.

The 13.0393 observation is a new ordinary-runner peak, not a stable deployment
guarantee. Both candidates still have minima below ten. Even the first-series
256-thread sample 9.997516979 is below ten and is not rounded into a pass.
The 512 candidate is selected for same-process pairing because it combines the
lower blocking kernel time, stronger large-matrix microbenchmarks and higher
observed ordinary minimum, not because only its peak is retained.

## Same-process 128 versus 512 pairing

A generation-switching probe keeps both shared kernels in one runtime.
Independent device auditing verifies all 50 function bodies from each qualified
variant, including the unchanged fallbacks, against the corresponding probe
functions. All 52 compared kernel metadata records also match after entrypoint
name normalization. The combined image has 56 functions; whole-image and
host-library identity are not claimed.

Before timing, both modes pass 464 complete text-output comparisons and 232
reset pairs against their independently HF-qualified ordinary runtimes. Every
step counts exactly 147 calls to the requested shared-input mode. Two opposite
generation sequences retain one complete warmup per mode per process, eight
measured generations per mode and all eight neighboring pairs:

| Instrumented mode | Minimum | Median | Maximum decode tokens/s |
| --- | ---: | ---: | ---: |
| 128 threads | 11.4439 | 11.6015 | 12.2094 |
| 512 threads | 12.0984 | 12.1865 | 12.3486 |

All eight pairs improve. The paired median ratio is 1.05002 and geometric mean
1.04668; the weakest pair is only 1.00041 and is not omitted. Both modes' 704
measured decode steps have frequency endpoints at 1400 MHz. This does not prove
constant frequency within each step. Frequency reads occur outside step timers
and cost a median 0.1786% of loop time; cadence effects are not excluded.

This pairing supports an approximately 5% benefit in the observed instrumented
setting, but it does not replace the mixed ordinary three-mode series, establish
low-frequency behavior or turn the 9.7423 ordinary minimum into a pass. The
512-thread source integration below still needs its own ordinary acceptance.

## Complete nonblocking global trace

The 512-thread ordinary candidate also has a fresh nonblocking trace with one
complete warmup and one complete measured generation, including all original
IDs and EOS. Independent SQLite auditing covers all 232 step windows: 28 prompt
and 88 decode steps in each run. Every window contains 788 kernel records,
147 shared GEMVs, eight fused-attention kernels, twelve stream synchronizations
and no async allocation/free. Runtime mappings and protected artifacts are
verified. Both full generations retain the original output checks.

Across the 88 measured decode steps, median host execution-window time is
118.6875 ms, compute-track union 104.5838 ms, copy-label union 0.6801 ms and
post-execution validation gap 0.8498 ms. Measured prefill medians for the same
metrics are 118.7504, 104.4250, 0.6831 and 0.8485 ms. These are profiled intervals,
not ordinary throughput or independent additive costs. Asynchronous kernel
labels can share intervals and are not summed for per-kernel attribution.
The short blocking traces above remain the bounded attribution evidence.

## Selectable source block size

`EXECUTORCH_MUSA_GGUF_SHARED_THREADS` is a CMake string setting accepting 128,
256 or 512, with default 128. It affects only the shared-input packed GEMV
launches. `SHARED_INPUT` and `VECTOR_DEQUANT` remain independently default OFF;
the block-size setting does not enable either feature. Source compilation
without an explicit thread definition also defaults to 128, and a static
assertion rejects unsupported values when the shared path is compiled.

The row index and launch grid now use the selected row groups. Staging capacity,
barrier placement, original 128-thread non-shared fallbacks, per-row arithmetic,
Q6 kernels and all type/alignment/stream checks are unchanged. Existing
byte-exact offset/tail tests additionally cover rows 159/160/161 and 191/192/193,
including boundaries of both the 32-row and 64-row blocks.

A fresh source build tests eight configurations: omitted/default 128, 256, 512,
reset to 128, repeated 512, 512 with shared input disabled, 512 with vector
dequantization disabled, and restored 512. Each passes 18 GGUF and 15 attention
tests, with the unavailable second-device test skipped. The first five and
restored configurations use the existing qualified shared/vector settings;
this does not mean the project's default enables these features.

Complete 128/256/512 device images match the previously measured corresponding
prototypes. Repeated 128 and 512 settings reproduce object, runtime and image
hashes. The disabled-shared image matches the previous qualified non-shared
image. The disabled-vector configuration has native correctness coverage, not
a new performance claim. Five separate configure attempts reject 0, 64, 1024,
a semicolon-separated list and a nonnumeric string with the intended diagnostic.

Only the 512 GGUF object is substituted into an isolated source runtime for
full-model testing. A fresh independent audit verifies all three cases again:
2298 byte-exact complete outputs, 1149 reset pairs, 18 negative controls and all
original IDs/EOS with unchanged frozen-HF bounds. The fresh CMake build and
isolated model link are distinct evidence; production artifacts are not replaced.

## Source-runtime ordinary retest

A fresh ordinary-runner comparison uses the qualified 128-thread source,
the new 512-thread source and an isolated Q6-head shared-input candidate.
Two opposite three-mode orders retain twelve process warmups and all 24
measured generations, eight per mode. No profiler, frequency sampling or
blocking environment is added; all original token IDs and EOS are retained.

| Runtime | Decode minimum | Decode median | Decode maximum | Serial prefill median |
| --- | ---: | ---: | ---: | ---: |
| 128-thread combined source | 9.5891 | 11.8339 | 12.4482 | 11.0075 |
| 512-thread source | 10.1224 | 12.3788 | 13.0030 | 11.1898 |
| 512 threads plus Q6 head staging | 9.9777 | 11.2519 | 12.8348 | 11.2436 |

All rates are tokens/s. The 512-thread source exceeds ten in every measured
generation of this batch, with a 1.04605 unpaired median ratio to its control.
This is not a causal speedup estimate or proof of stable deployment throughput:
the earlier byte-identical 512-thread device prototype still has a 9.7423 low,
and the historical ordinary peak remains 13.0393. Do not discard that batch.
The added head staging does not establish a model benefit; its microbenchmark,
precision qualification and blocking trace are recorded in
[Q6 output-head input reuse](QWEN35_GGUF_Q6_HEAD_INPUT.md).

## Decision

Retain 128 as the source/default block size. The 64-thread variant is rejected;
both larger variants are numerically qualified, and 512 now has a selectable
source implementation backed by the microbenchmark, blocking-profile and
same-process evidence. It is not yet a stable-ten deployment result. Remaining packed
Linear work, including the roughly 28.1 ms Q6 head/projection contribution in
the blocking trace, remains a target. No precision, token checks, low samples
or desktop GPU conditions are changed to satisfy the throughput objective.

## Evidence

Under ETSupport `artifacts/gguf-20261029`:

- `shared-block-size-probe-v1/{kernels.mu,bench.cpp,build.py,run.py}`:
  all four source/build variants and complete raw outputs and timing pairs.
- `audit_shared_block_size_v1.py`,
  `shared-block-size-probe-v1-decision.json`: independent image, resource,
  precision, guard and timing audit.
- `build_shared_block_{256,512}_v1.py`, `build-shared-block-{256,512}-v1`,
  `validate_shared_block_{256,512}_v1.py`,
  `shared-block-{256,512}-validation-v1`,
  `audit_shared_block_{256,512}_runtime_v1.py`,
  `shared-block-{256,512}-runtime-decision-v1.json`: both isolated runtimes,
  native tests and independent three-case precision/provenance qualification.
- `shared-block-{256,512}-blocking-profile-v1`,
  `audit_shared_block_{256,512}_profile_v1.py`,
  `shared-block-{256,512}-profile-decision-v1.json`: both blocking traces,
  full-window counts, interval sums and independent SQL audits.
- `benchmark_shared_block_sizes_v1.py`, `shared-block-sizes-benchmark-v1`,
  `audit_shared_block_sizes_ordinary_v1.py`,
  `shared-block-sizes-ordinary-decision-v1.json`: ordinary three-mode series,
  all 24 measured generations, twelve warmups and original trajectories.
- `alternating-shared-512-probe-v1`,
  `alternating-shared-512-device-audit-v1`,
  `audit_alternating_shared_512_devices_v1.py`,
  `alternating-shared-512-validation-v1`,
  `alternating-shared-512-benchmark-v1`,
  `audit_alternating_shared_512_v1.py`,
  `alternating-shared-512-decision-v1.json`: function/metadata identity,
  full text comparisons, mode counters, all pairs and frequency endpoints.
- `profile_global_shared_block_512_v1.py`,
  `global-shared-block-512-profile-v1`,
  `analyze_global_shared_block_512_v1.py`,
  `audit_shared_block_512_global_v1.py`,
  `global-shared-block-512-profile-decision-v1.json`: complete nonblocking
  trace, all 232 windows and independently verified interval/phase summaries.
- `build_shared_block_source_v1.py`, `build-shared-block-source-cmake-v1`,
  `build-shared-block-source-v1`, `validate_shared_block_source_v1.py`,
  `shared-block-source-validation-v1`, `audit_shared_block_source_v1.py`,
  `shared-block-source-decision-v1.json`: eight source configurations, five
  invalid values, exact device images, expanded row boundaries and fresh
  independent three-case source-runtime qualification.
- `benchmark_source_head_staging_v1.py`, `source-head-staging-benchmark-v1`,
  `audit_source_head_staging_ordinary_v1.py`,
  `source-head-staging-ordinary-decision-v1.json`: source-runtime three-mode
  retest, all 24 measured generations, twelve warmups and original trajectories.

These microbenchmarks are not model throughput. Stable minimum decode
throughput of ten tokens/s remains unproven.
