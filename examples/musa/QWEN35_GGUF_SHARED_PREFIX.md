# Global profiling and shared attention prefix candidate

This work targets the original Qwen3.5-4B Q4_K_M workload on M1000/MP22.
The goal remains 10 decode tokens/s. The last accepted unprofiled result is
8.473 tok/s median; this candidate does not yet establish a new model result.

## Global baseline trace

A nonblocking `msys --trace=musa,osrt` capture covers loading, one full warmup
and one measured original text generation: 28 prompt tokens and 88 decode
steps per generation, to the original EOS. Both generated trajectories match.
The source/native shim is the accepted metadata/tile version. The desktop GPU
is shared and unlocked. Profiling overhead and concurrent desktop work make
the profiled 5.885 tok/s unsuitable as an unprofiled baseline comparison.

| Measured phase | Host execution-window median | Compute-track interval union median |
| --- | ---: | ---: |
| Serial prefill | 167.457 ms | 142.077 ms |
| Decode | 169.225 ms | 143.594 ms |

There are 844 kernel labels and 12 `musaStreamSynchronize` calls per step.
The eight valid-prefix attention calls each copy their position and mask back
to the host, allocate/free pinned control memory and synchronize the stream.
These repeated control dependencies are a target independent of GEMV tuning.
The 1185 `musaGetDevice` calls per step total only about 0.503 ms during measured
decode; removing correctness guards is not the priority. Host output validation
gaps have a median near 1.315 ms. Final-run teardown is excluded from these gaps.

Async kernel attribution is not trustworthy in this trace: 844 labels map to
only about 81 distinct intervals per step. Their apparent sum is about 335 ms,
while their interval union is only about 17 ms. Use the separate compute track
for global timeline coverage, not either label statistic as per-kernel cost.
CPU synchronization waits, copies and GPU intervals overlap and must not be
added together. The host/compute difference is not a pure CPU-computation time.
The previous blocking two-step trace remains a separate kernel-attribution
experiment, not normal token latency.

Fifteen lossless layout probes covered whole-block row interleaving and
16-byte-chunk interleaving, with and without lane permutation. Each passes
three complete-output comparisons on eight real Q4/Q5 geometries. Many cases
regress and the desktop measurements fluctuate substantially. None is promoted
to production based on isolated favorable samples.

## Explicit shared control

`causal_prefix(position, mask)` validates the complete GPU causal mask and
position on the calling stream, then returns a newly owned CPU int64 tensor
`[valid_length, cache_capacity]`. It is a snapshot: changing the source GPU
tensors does not change it. A new snapshot is required for each decoding step.

`prefix_attention(query, key, value, control)` accepts this explicit host prefix,
checks its geometry and bounds, and runs the same muDNN QK/scale/softmax/PV
sequence as the legacy checked operator. Attention arithmetic remains on GPU.
The existing `causal_attention` ABI and checked runtime remain available.
There is no global or pointer-keyed cache, no input-value guessing, no TF32 and
no relaxation of numerical tolerances.

Inductor did **not** automatically merge repeated custom prefix calls in the
small export probe. The opt-in multimodal step therefore creates exactly one
prefix tensor and passes it explicitly through the layer options to all eight
full-attention consumers. The expected change is eight attention-control
readback barriers to one; a complete-model runtime trace must verify it.

```python
model, report = load_gguf_text(
    gguf_path,
    context_length=4096,
    valid_length_attention=True,
    predecode_q6_weights=True,
    share_attention_prefix=True,
)
step = GgufMultimodalStep(model, share_attention_prefix=True).eval()
```

Both options are off by default. This requires a new export and a shim with the
two new entry points, but no weight requantization. Using the new attention
module outside the shared wrapper falls back to per-call prefix preparation,
not a hidden cache. The fixed Qwen path preserves projection, RoPE, Q/K norm,
KV updates, output gating and output projection; tests cover attention width
different from embedding width.

## Current validation boundary

Native validation passes fourteen attention tests (one additional second-device
test is skipped) and eleven GGUF tests. It covers byte equality to legacy BF16
and grouped FP32 attention, poisoned future KV entries, snapshots, control
updates on a nondefault stream, invalid masks and lengths, offsets and output
canaries. Small native AOTI exports exercise CPU control buffers with GPU math,
planned memory, repeated execution and runtime prefix changes, comparing all
new-attention output bytes against the legacy operator in the same export.

The first full-model eager probe caught an incorrect output reshape: attention
width is 4096 while embedding width is 2560. Its failed artifacts are retained;
the fix uses the head count and head dimension, with a projection/gating/cache
regression test. Full-model export and numerical/performance acceptance remain
required before claiming a deployment gain or achievement of 10 tok/s.

Local ETSupport evidence under `artifacts/gguf-20261029`:
`global-metadata-tile-profile-v1`, `row-interleave-probe-v1`,
`build-decode-shared-prefix-v1`, `test-shared-prefix-native-v1.log`,
`test-shared-prefix-python-v2.log`, `test-shared-prefix-python-v3.log`,
and the failed `decode-shared-prefix-export-v1` probe. The next export uses
`decode-shared-prefix-export-v2`; original accepted artifacts are unchanged.
