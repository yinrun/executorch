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

Full-model export, complete original-trajectory numerical acceptance and
unprofiled same-session performance comparisons remain required. Reducing call
count is not proof of a throughput gain or achievement of the 10 tok/s goal.
Local ETSupport evidence starts at `artifacts/gguf-20261029/test-merged-gdn-v1.log`
and `decode-merged-gdn-export-v1`. Previous accepted artifacts remain unchanged.
