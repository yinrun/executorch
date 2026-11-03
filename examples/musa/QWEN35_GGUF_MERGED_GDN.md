# Packed GDN gate/beta/decay merging

The 24 linear-attention layers in the original Qwen3.5-4B Q4_K_M GGUF each have
three same-input Q4_K projections: gate (4096 rows), beta (32 rows) and decay
(32 rows). The opt-in `merge_gdn_gate_projections=True` loader option concatenates
their unchanged packed rows into one 4160-row projection and splits its result.
It preserves QKV projection, gate reshaping, tiled head order, causal convolution,
recurrent updates, state reset, normalization and output projection.

No new kernel, native ABI, quantization format, TF32 or hidden pointer cache is
introduced. The existing metadata/tile runtime is reused. Default loading keeps
the original separate projections; the new option is independent of FFN merging
and shared-prefix attention. The candidate export combines this option with
`merge_ffn_gate_up=True`, but leaves shared-prefix attention disabled.

The merged projection rejects biased/non-packed layers, differing encodings,
input geometries or devices, and unequal beta/decay row counts. The GDN wrapper
also checks the configured head geometry, refuses a second merge, removes the
obsolete packed buffers and retains explicit recurrent-state ownership.

Twelve tests pass across `test_musa_qwen35_gdn_merge.py` and the existing FFN/GGUF
model tests. New tests cover opaque single-projection export, rejected inputs,
unchanged packed bytes, eager output checks, recurrent/conv-state updates and
reset, and a three-execution native AOTI comparison. The native probe includes
512-row gate and four-row controls, so it exercises the legacy narrow-kernel to
merged wide-kernel dispatch boundary; all output bytes match.

## Complete-model result: numerically valid, no incremental decode gain

The full export succeeds with 32 merged FFNs, 24 merged GDN projections, 169
packed-linear calls, one delegate, five user inputs and 64 mutated state buffers.
All six eager probe outputs are byte-identical to the preceding FFN-only export.
The original text, single-image and multiple-image native trajectories each run
twice with reset. All 2298 complete 248320-float logits and 1149 reset pairs are
byte-identical to the accepted predecoded baseline. Frozen matched Hugging Face
references pass the unchanged `0.001 + 0.001 * abs(reference)` gate with zero
outliers/argmax mismatches; maximum absolute errors remain 0.0000228882,
0.000739098 and 0.000905991. Negative controls and a separate full-output/reference
rehash audit pass. This does not rerun Hugging Face or the raw-image frontend,
establish full 4096-token/concurrency coverage, or resolve historical-export
bit-equality differences.

Two same-session three-variant series use baseline/FFN/candidate/candidate/FFN/
baseline and its reversed-role counterpart. Each process has one full warmup
and two measured original text generations: 28 serial-prefill tokens, 88 decode
steps, unchanged IDs and EOS. There is no profiler, launch blocking or logit
dumping during timing; readback, finite checks and argmax remain. Runtime maps
verify the same native shim and no libtorch/libpython. Eight measured runs per
variant give:

| Variant | Decode minimum | Decode median | Decode maximum | Serial prefill median |
| --- | ---: | ---: | ---: | ---: |
| Metadata/tile baseline | 6.394 | 6.437 | 6.488 | 6.445 |
| FFN only | 6.507 | 6.535 | 6.576 | 6.472 |
| FFN plus GDN | 6.498 | 6.526 | 6.975 | 6.562 |

Units are tokens/s. The incremental decode median ratio versus FFN-only is
0.99863 (-0.14%); per-series ratios are 0.99638 and 0.99981. Thus this option is
not promoted as an additional decode optimization. Its gain over the original
baseline must not be misattributed to GDN merging: FFN-only is already faster.
The single 6.975 maximum is not a sustained-throughput claim. The GPU remains
shared/unlocked, and these numbers cannot be compared directly with historical
8.473 tok/s measurements under different desktop load. The 10 tok/s goal is not
achieved.

The separate complete-generation nonblocking trace confirms 764 kernel labels
per step, down from 812 with FFN only and 844 originally, while twelve stream
synchronizations remain. Decode host-window median is 160.531 ms, compute-track
union 136.403 ms, output-validation gap 1.309 ms. Its profiled 6.199 tok/s is not
normal throughput. Fewer launches have not materially reduced the GPU critical
path; further work returns to the packed GEMV kernels rather than assuming
call-count reduction will meet the target.

Local ETSupport evidence under `artifacts/gguf-20261029`:
`test-merged-gdn-v1.log`, `decode-merged-gdn-export-v1`,
`decode-merged-gdn-validation-v1`, `decode-merged-gdn-benchmark-v1`,
`decode-merged-gdn-benchmark-v2`, `global-merged-gdn-profile-v1`, and
`decode-merged-gdn-acceptance-v2.json`. The option remains off by default;
previous accepted deployment artifacts are unchanged.
