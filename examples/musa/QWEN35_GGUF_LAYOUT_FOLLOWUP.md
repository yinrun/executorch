# Lossless layout and graph replay follow-up

## Decision and scope

No candidate in this investigation replaces the accepted grouped-Q6-head
deployment. The recorded accepted result remains 7.3423 median decode tokens/s
in its earlier paired series. There is no new ordinary whole-model throughput
measurement here, and the **10 tokens/s target remains unmet**.

The workload remains Qwen3.5-4B Q4_K_M with FP32 activations, state and
accumulation. None of these probes introduces requantization, TF32, MTP,
vocabulary pruning or a relaxed numerical gate. All layout changes are
lossless and all inference arithmetic runs on MUSA. These are artifact-only
experiments; neither the repository loader nor a default runtime path changes.

The starting global evidence is the accepted 812-kernel graph and the
[wider GDN experiment](QWEN35_GGUF_FUSED_GDN.md), which reduced the count to
692 without establishing a reliable model speedup. That result motivates
examining projection data layout, arithmetic reuse and actual graph replay
rather than treating launch count as the performance objective.

## Metadata and memory layout

All comparisons use the currently accepted native shim, eight real Q4/Q5
matrices and three complete, finite, byte-exact output comparisons per matrix
and variant. Timing alternates baseline/candidate launches on the shared
desktop GPU. A per-matrix best ratio is not a deployment decision.

| Probe | Variants | Complete exact outputs | Outcome |
| --- | ---: | ---: | --- |
| FP32 metadata with quad arithmetic | 4 | 96 | No broad gain |
| Original packed blocks, row/block interleaving | 8 | 192 | Small, shape-dependent differences |
| Separate FP16-base plane, scalar metadata use | 2 | 48 | No broad gain |
| Separate FP16-base plane, vector metadata multiply | 2 | 48 | No broad gain |

The first format stores already multiplied FP32 scales/minima, keeping the
low-bit payload. It grows Q4 blocks from 144 to 192 bytes and Q5 from 176 to
224. Adding the newer quad conversion/FMA implementation and MP22 ILP
scheduling recovers much of the old scalar prototype's regression, but the
16-component versions remain mostly near parity. Four-component versions
regress. Caching Q5 loads helps some samples, not all shapes.

Interleaving keeps the original 144/176-byte blocks and exact arithmetic,
reordering row/block tiles of 1, 4, 16 or 32 with two load-cache policies.
Tile 1 without added caching is the unchanged-layout control. Small control
variation remains visible. Q5 QKV ratios range roughly from parity to 1.064;
large Q4 FFN gains are small or absent. These probes require row counts
divisible by the tile; they do not establish arbitrary-tail support. No PTE
is reexported or weight cache changed for this format.

The separate-plane format is more compact: each original-size block retains
its payload, while its old metadata region holds expanded byte-sized
scale/minimum levels. The original two FP16 bases move into a separate
four-byte-per-block plane. Total storage is **148 bytes for Q4 and 180 for
Q5**, only 2.78%/2.27% larger than the originals. Converting four metadata
bytes together avoids the old six-bit field reconstruction. Multiplication
by the original FP16 bases and subsequent FP32 arithmetic are unchanged.

The second version keeps metadata multiplication as a four-component vector
expression. The 1024-row Q4 key projection has ratios around 1.05–1.10 in
some variants, but Q5 QKV regresses and the large FFN projections are near
parity or slower. Small isolated wins do not justify global format conversion,
its extra storage, loader work and model reexport.

## Paired-row input reuse

A separate fixed-order kernel shares activation loads among one, two or four
output rows separated by a matrix-row partition. This tests the arithmetic
foundation for a possible gate/up epilogue, not an implemented fused FFN.
Both four- and sixteen-component lane layouts are tested.

The initial generated source accidentally includes an unrelated Q6 function
inside a scope: all six builds fail before GPU execution. The corrected
source then passes 144 complete exact outputs, but nested array reductions
create 64/128/256-byte local allocations in the sixteen-component variants.
Even the one-row control becomes substantially slower.

Explicit fixed-index reductions remove those local/stack allocations. The
corrected family passes another **144 complete exact outputs**, with zero
reported local memory in the sixteen-component variants. The one-row control
recovers large-matrix timing. Two/four-row reuse still regresses; for example,
the sixteen-component two-row large-matrix ratios are about 0.87–0.97.
The Q5 two-row kernel uses 159 temporary registers. Removing local storage
does not by itself establish better scheduling or occupancy. Do not build a
new fused FFN around this slower inner kernel on the strength of input reuse
alone.

## Graph capability and measured replay

The current MUSA backend rejects `enable_cuda_graph_for_method`. These tests
call the installed SDK directly; they do not bypass that backend restriction
in a deployed model.

The device reports memory-pool support. A static-buffer captured copy graph
passes 32 replays with changing inputs, **8192 exact floats**, and successful
cleanup. A second test allocates temporary memory during capture with
`musaMallocAsync`; the following captured D2D copy returns **invalid argument**,
before graph instantiation. This is the observed failing sequence, not proof
that every possible allocation-containing graph is unsupported. In particular,
the test never reaches validation of the AutoFreeOnLaunch instantiation flag.

Static-buffer graph timing uses the unchanged accepted GEMV kernels on four
real matrices, with either one or 32 repeated nodes. All eight cases pass
three changing-input, complete output comparisons: **24 exact outputs**.
One-node graph speed ratios are 0.576–0.775; 32-node ratios are 0.793–0.997.
None beats the equivalent directly enqueued sequence in this experiment.

The 32-node tests repeat one matrix with hot data; they are not a whole-model
graph and cannot predict model throughput. Event timing is normalized by node
count; one-node and batched absolute per-launch times include different amounts
of event/queue overhead. These results do not justify removing graph guards,
moving validation out of the model, or investing in graph-specific allocation
changes as an already proven route to 10 tokens/s.

## Evidence and next decision

An independent audit verifies **696 complete byte-exact micro-outputs** across
the successful layout/reuse families and graph timings, plus the static copy
graph's 8192 floats. It checks compilation results, source/report hashes and
rehashes the protected runner, runtime libraries and PTE/PTD files. No complete
model accuracy rerun or throughput promotion is claimed from these tests.

Local ETSupport evidence under `artifacts/gguf-20261029`:

- `compact-metadata-quad-probe-v1`, `interleaved-packed-quad-probe-v1`:
  source, builds, native timing and exact-output comparisons.
- `metadata-plane-quad-probe-v1`, `metadata-plane-quad-probe-v2`:
  low-growth metadata-plane prototypes and vector-expression follow-up.
- `paired-quad-q45-probe-v1`: compile-only failure; `v2` and `v3`: nested
  versus fixed-index reductions, resource logs and complete output tests.
- `graph-capability-probe-v1`, `graph-gemv-probe-v1`: SDK capability failure,
  static replay validation and alternating direct/replay timing.
- `audit_layout_followup_v1.py`, `layout-followup-decision-v1.json`:
  provenance, exact-output counts and unchanged protected artifacts.

The remaining large-matrix work should change useful parallel work per
launch, not merely shuffle metadata or merge more rows. A K-partitioned
projection is a possible next experiment, not a measured improvement: its
changed FP32 reduction order must pass the unchanged full HF/argmax/reset
gate before any performance promotion. The accepted deployment stays intact.
