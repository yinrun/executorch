# Packed gate/up projection merging experiment

The Qwen3.5-4B GGUF has matching Q4_K gate/up projections in all 32 FFNs.
`load_gguf_text(..., merge_ffn_gate_up=True)` concatenates their unchanged packed
rows once during loading. `GgufMergedFeedForward` calls the existing native
packed linear once, splits the result, and preserves the original activation,
elementwise multiplication and down projection. This removes one packed-linear
invocation per layer without a new quantization format, kernel or native ABI.
The default remains false. Existing source-weight hashes and total packed byte
counts still describe the unchanged weights, not a requantized model.

The constructor rejects non-packed or biased projections and mismatched gate/up
encodings, shapes or devices. It preserves the down-projection module and the
activation function. Metadata, eager and small native-export tests pass: seven
tests in `test_musa_qwen35_gguf.py`, including planned combined memory and three
native executions comparing the merged and separate FFNs byte for byte.

Full-model eager output is not bit-identical to the previous eager model.
The first export probe deliberately stopped on this stricter check; artifacts
are retained. Merging changes the 4096-row eager GEMM chunk boundaries, unlike
the row-independent native decode kernel. Across six eager steps, maximum
absolute differences to previous eager outputs range from 5.54e-6 to 2.15e-5,
with identical selected tokens and no violations of the existing mixed-error
tolerance. This observation does not replace native numerical acceptance:
the complete native trajectories must still match all baseline logit bytes and
the frozen matched Hugging Face reference gates before promotion.

Full export, native trajectory validation and unprofiled same-session A/B are
pending. No model throughput gain or achievement of 10 tokens/s is claimed.
The accepted deployment and the off-by-default shared-prefix experiment are
unchanged. Local evidence under ETSupport `artifacts/gguf-20261029`:
`test-merged-ffn-v2.log`, failed `decode-merged-ffn-export-v1`, and the second
export `decode-merged-ffn-export-v2`.

Additional rejected kernel probes are retained rather than silently promoted:
six Q4/Q5 uint8-level expansion variants, eight current-metadata staged/unrolled
or shared-input variants, and three compact-Q6 scale-storage variants. Each
passes three byte-exact activations on every tested real geometry (48, 64 and
9 geometry comparisons respectively). None establishes a broad speed gain;
the compact-Q6 head also regresses. These probes add no deployed weight layouts.
