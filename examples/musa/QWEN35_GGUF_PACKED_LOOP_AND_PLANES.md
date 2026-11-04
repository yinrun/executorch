# Packed loop, packet-plane layout and Q5 caching

This follows the [output-head investigation](QWEN35_GGUF_HEAD_UNROLL.md).
The latest source-runtime blocking trace attributes about 85% of kernel time
to packed Linear; Q4/Q5 packet kernels are the largest remaining group.
The Qwen3.5-4B Q4_K_M weights, FP32 activations/state/accumulation and numerical
gate remain unchanged. No TF32, requantization, MTP or vocabulary pruning is
introduced. These are isolated experiments, not enabled production options.

## Current packet-kernel outer-loop unrolling

The old `ilp-block-unroll-probe-v1` predates packed-quad arithmetic and the
local-first reduction. This new probe changes only the compiler unroll
directive on the current packet kernel's outer quantization-block loop.
It retains per-weight dequantization, FMA order, local reduction and Split-K
partition boundaries. Compiler-default, explicit one, two and four run across
eight real Q4/Q5 matrices in forward/reverse order, including the actual merged
gate/up matrix. Long-K microbenchmarks include a separate partial reduction on
both sides; the exported caller-owned-parts path does not.

All 64 processes pass three complete byte-identical output comparisons and
eight output canaries per input: **192 complete outputs and 1536 canaries**.
Each process warms both sides for at least three seconds, then retains 100
alternating event pairs after two initial pairs. All 6400 samples remain:
2330 frequency observations read 1.4 GHz and 4070 read 730 MHz. No frequency
is requested or locked, and sampling cannot exclude intra-pair changes.

The default and explicit-one complete device images reproduce each other and
the qualified original object. Both actual unroll candidates regress in both
orders on every matrix:

| Matrix | Two-block ratios, forward / reverse | Four-block ratios |
| --- | ---: | ---: |
| Q4 attention gate | 0.9379 / 0.9242 | 0.7226 / 0.7783 |
| Q5 QKV | 0.9370 / 0.9402 | 0.4348 / 0.4390 |
| Q4 FFN gate | 0.9838 / 0.9831 | 0.8106 / 0.7153 |
| Q4 SSM output | 0.9904 / 0.9909 | 0.7584 / 0.6865 |
| Q4 attention K | 0.8468 / 0.8742 | 0.7471 / 0.7276 |
| Q4 attention Q | 0.9746 / 0.9757 | 0.7878 / 0.7813 |
| Q4 FFN down | 0.9595 / 0.9567 | 0.6925 / 0.7865 |
| Q4 merged gate/up | 0.9941 / 0.9854 | 0.7972 / 0.7957 |

Ratios are reference median latency divided by candidate median latency.
Q4 single-part temporary registers increase from 109 to 119/185; Q4 four-part
from 111 to 119/183; Q5 from 111 to 179/256. Four-block Q5 reports 72 local/spill
bytes; the others report zero. These compiler resources do not prove occupancy
or stall causes. The unchanged small attention-K control also varies, so no
favorable small sample is promoted. No unroll variant warrants model integration.

## Lossless packet-plane layout

The next experiment changes memory layout, not arithmetic. Unlike the earlier
whole-block row/K interleaving, it groups corresponding 16-byte packets across
4, 16 or 32 rows. Metadata and payload packets form separate planes within
each row-group/K-block tile. Storage remains exactly 144 bytes per Q4 block
or 176 per Q5 block; no metadata expansion or activation quantization occurs.

For row tile `R`, quantization blocks per row `B`, and packets per block `P`
(`9` for Q4, `11` for Q5), the destination packet index is:

```text
(((row // R) * B + block) * P + packet) * R + row % R
```

A GPU preparation kernel performs the permutation. Before each process's
numerical/timing work, an independent host inverse-index comparison checks
every prepared byte against the original packed matrix. The model arithmetic
remains entirely on MUSA. The original-layout control performs a D2D copy into
an equally sized separate buffer, covering relocation without transposition.
Preparation time is recorded separately and is not included in GEMV timing.
The prototype requires row counts divisible by its tile; arbitrary-tail
support, production metadata and cache lifetime management are not implemented.

Four layouts across the same eight matrices and both orders again pass
**192 complete byte-identical outputs and 1536 canaries**. All 6400 timing
pairs remain, with 2389 observations at 1.4 GHz and 4011 at 730 MHz. The original
layout reproduces the qualified original complete device image.

| Matrix | Tile 4 ratios | Tile 16 ratios | Tile 32 ratios |
| --- | ---: | ---: | ---: |
| Q4 attention gate | 0.9606 / 0.9651 | 0.9735 / 0.9576 | 0.9698 / 0.9543 |
| Q5 QKV | 1.0795 / 1.0805 | 1.0834 / 1.0873 | 1.0650 / 1.0654 |
| Q4 FFN gate | 0.9734 / 0.9737 | 0.9613 / 0.9643 | 0.9533 / 0.9528 |
| Q4 SSM output | 0.9759 / 0.9798 | 0.9685 / 1.0062 | 0.9526 / 0.9656 |
| Q4 attention K | 0.9776 / 0.9756 | 0.9875 / 0.9948 | 0.9782 / 0.9833 |
| Q4 attention Q | 0.9696 / 0.9573 | 0.9616 / 0.9640 | 0.9504 / 0.9552 |
| Q4 FFN down | 0.9772 / 0.9832 | 0.9973 / 1.0323 | 0.9965 / 1.0085 |
| Q4 merged gate/up | 0.9583 / 0.9574 | 0.9688 / 0.9714 | 0.9437 / 0.9530 |

All layouts report zero spills. Tile 4/16 Q5 uses 104 temporary registers and
tile 32 uses 103, versus 111 originally; Q4 packet variants use 106. Register
reduction alone does not predict performance. Q5 consistently improves, but
most Q4 matrices regress; some smaller gains change direction. Even the
unchanged attention-Q control reaches 1.0398 in one run. This does not justify
a global layout replacement or extrapolating Q5's ratio to all Linear work.

## Q5-only cache-policy follow-up

Q4 packet loads already use `__ldg`; Q5 uses ordinary loads. A separate
six-configuration experiment tests original/tile-8/tile-16 layouts with either
ordinary Q5 loads or `__ldg` for both packed data and activations. It targets the
real 8192-by-2560 QKV matrix, not a claim that the other recompiled kernels are
qualified. All twelve forward/reverse processes pass **36 complete byte-exact
outputs**, **288 canaries** and complete prepared-byte checks. All 1200 pairs
remain: 670 sampled frequencies are 730 MHz and 530 are 1.4 GHz.

| Layout / Q5 load policy | Forward ratio | Reverse ratio |
| --- | ---: | ---: |
| Original / ordinary, control | 1.002821 | 0.999795 |
| Original / read-only | 1.096838 | 1.056934 |
| Tile 8 / ordinary | 1.003043 | 1.072860 |
| Tile 8 / read-only | 1.018339 | 1.018610 |
| Tile 16 / ordinary | 1.052152 | 1.046546 |
| Tile 16 / read-only | 1.077637 | 1.077922 |

The original-layout read-only variant measures 0.62166 ms against 0.68186 ms
in the first order (all frequency samples 730 MHz), and 0.44648 against
0.47190 ms in reverse (all 1.4 GHz). The mixed-frequency tile-8 forward process
is retained, not removed. Ratios from separately sampled frequency conditions
must not be directly ranked as if every configuration were paired together.

The simpler original-layout read-only path is selected for full-model
qualification: it needs no new packed format, preparation, persistent cache,
weight mutation or model re-export. This is not a demonstrated model speedup
or a justification for enabling it by default. The packet-plane experiments
remain isolated, including all their regressions.

## Original-layout Q5 runtime qualification

An isolated runtime replaces only the GGUF device object in the qualified
source-attention link. It uses `PLANE_TILE=0` and `Q5_CACHE=1`; no packet-plane
preparation is used by the model. Before linking, the unchanged micro control
reproduces the original device image and the new object reproduces the measured
cache-only image exactly. All other link inputs, runner, extension, PTE/PTD and
protected runtimes remain unchanged. The Q6 head unroll candidate is not added.

Text, single-image and multi-image cases pass **2298 complete 248320-float
outputs**, **1149 byte-identical reset pairs**, original generated IDs/EOS,
finite checks, exact argmax and the frozen HF bound
`0.001 + 0.001 * abs(reference)`. Every output is byte-identical to the
qualified source-attention runtime, and eighteen negative controls are rejected.
Independent auditing rechecks all raw buffers and HF references, runtime maps,
original trajectories, protected hashes and the cache-only source/flag change.

| Case | Complete outputs | Maximum absolute error vs frozen HF |
| --- | ---: | ---: |
| Text | 232 | 0.0000262260 |
| Single image | 856 | 0.0006752014 |
| Multiple images | 1210 | 0.0008490086 |

Prepared image features and frozen HF are reused; the raw-image frontend and
HF model are not rerun. This is an isolated-link qualification, not new CMake
option coverage, arbitrary-shape/concurrent-service acceptance or deployment.
Ordinary and same-process full-generation timing of this Q5 candidate remain
necessary before claiming a model-level performance improvement.

## Evidence

Under ETSupport `artifacts/gguf-20261029`:

- `packed-unroll-probe-v1`, `packed-unroll-probe-v1-decision.json`: four builds,
  all 64 processes, complete raw outputs/canaries, resources and unfiltered rates.
- `packed-plane-probe-v1`, `packed-plane-probe-v1-decision.json`: GPU packing,
  complete host inverse checks, unchanged storage, original control and all
  64 processes. `audit_packed_probe_v1.py` independently checks both families'
  raw output bytes, input/source/library hashes, device-image identity,
  complete configuration coverage, resources and every recorded timing pair.
- `q5-plane-cache-probe-v1`, `audit_q5_plane_cache_v1.py`,
  `q5-plane-cache-probe-v1-decision.json`: Q5-specific cache/layout comparison,
  raw complete outputs, preparation checks and independent micro audit.
- `build_q5_cached_loads_v1.py`, `build-q5-cached-loads-v1`,
  `validate_q5_cached_loads_v1.py`, `q5-cached-loads-validation-v1`,
  `audit_q5_cached_loads_runtime_v1.py`, `q5-cached-loads-runtime-decision-v1.json`:
  original-layout cached-load runtime, protected inputs, three complete cases
  and independent full-buffer/frozen-HF/provenance audit.

No new ordinary full-model throughput is established here. Stable ten decode
tokens/s remains unproven; microbenchmarks are not substitutes for that gate.
