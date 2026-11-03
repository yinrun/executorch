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

## Full-model acceptance

The second export succeeds with one delegate, 64 mutated state buffers and five
user inputs. Packed-linear calls decrease from 249 to 217, with 32 merged
18432-row gate/up calls. The same accepted metadata/tile native shim and runner
are reused; shared-prefix attention is not enabled in this experiment.

All original text, single-image and multiple-image native trajectories pass,
each executed twice with reset: 2298 complete 248320-float outputs and 1149 reset
pairs are byte-identical to the accepted predecoded baseline. Frozen matched
Hugging Face references pass the unchanged `0.001 + 0.001 * abs(reference)` gate
with zero outliers and argmax mismatches. Maximum absolute errors are 0.0000228882,
0.000739098 and 0.000905991 respectively. Corruption and tolerance negative
controls pass. A separate audit rehashes every output and reference plus the
model, runtime and protected original artifacts. Native source hashes are
verified against their original metadata/tile commit, not the newer optional
shared-prefix shim source.

These checks do not rerun Hugging Face or the raw-image frontend, establish a
4096-token trajectory or concurrent execution, or resolve historical-export
bit-equality differences. The eager differences above do not occur in the
validated native trajectories.

## Performance

Two unprofiled, nonblocking, no-dump series use ABBA and BAAB order, each with
four processes, one full warmup and two measured original text generations per
process. Each generation has 28 serial-prefill tokens and 88 decode steps to
the original EOS. All token IDs, lengths and termination reasons match. Full
logit readback, finite checking and argmax remain enabled. Runtime maps verify
the intended native shim and absence of libtorch/libpython.

| Variant | Decode minimum | Decode median | Decode maximum | Serial prefill median |
| --- | ---: | ---: | ---: | ---: |
| Metadata/tile baseline | 6.345 | 6.416 | 6.480 | 6.409 |
| Merged FFN | 6.518 | 6.533 | 6.851 | 6.464 |

All units are tokens/s; eight measured runs per variant. Per-series median
improvements are 1.83% and 1.71%; the aggregate median improvement is 1.82%.
The desktop GPU is shared and unlocked. Its substantial background utilization
already existed before inference. Do not compare these fresh baseline/candidate
numbers directly to the earlier 8.473 tok/s as a code regression or extrapolate
an exclusive-GPU result. The goal of 10 tok/s is not reached.

A separate full-generation nonblocking profile confirms kernel labels decrease
from 844 to 812 per step, with the same twelve stream synchronizations.
Measured-decode host-window median is 161.079 ms, compute-track union median
136.774 ms and output-validation gap 1.314 ms. Its 6.170 profiled tok/s is not a
normal throughput measurement. Async label durations remain unsuitable for
per-kernel sums; GPU work and CPU waits overlap.

The first BAAB attempt (`benchmark-v2`) overlapped an accidentally concurrent
artifact audit during startup/warmup. That audit was stopped. The entire series
is retained but excluded from the final aggregate, and a clean BAAB (`v3`) was
rerun. The final numerical rehash audit runs only after timed work completes.

This is an optional, numerically validated small improvement, not a default
deployment replacement or an achievement of 10 tok/s. The next launch-reduction
target is the 24 GDN layers' same-input Q4 gate/beta/decay projections; those
have not been merged or benchmarked in this result.

Local evidence under ETSupport `artifacts/gguf-20261029`:
`test-merged-ffn-v2.log`, failed `decode-merged-ffn-export-v1`, successful
`decode-merged-ffn-export-v2`, `decode-merged-ffn-validation-v1`,
`decode-merged-ffn-benchmark-v1`, `decode-merged-ffn-benchmark-v3`,
`global-merged-ffn-profile-v1`, and `decode-merged-ffn-acceptance-v2.json`.

Additional rejected kernel probes are retained rather than silently promoted:
six Q4/Q5 uint8-level expansion variants, eight current-metadata staged/unrolled
or shared-input variants, and three compact-Q6 scale-storage variants. Each
passes three byte-exact activations on every tested real geometry (48, 64 and
9 geometry comparisons respectively). None establishes a broad speed gain;
the compact-Q6 head also regresses. These probes add no deployed weight layouts.
