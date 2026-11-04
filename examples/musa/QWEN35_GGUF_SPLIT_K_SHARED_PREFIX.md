# Qwen3.5-4B GGUF combined decode configuration

This configuration combines the existing opt-in Qwen3.5-4B GGUF export
options that have passed native trajectory validation:

```python
model, loading = load_gguf_text(
    gguf_path,
    context_length=4096,
    valid_length_attention=True,
    predecode_q6_weights=True,
    merge_ffn_gate_up=True,
    split_k_projections=True,
    share_attention_prefix=True,
)
step = GgufMultimodalStep(model, share_attention_prefix=True).eval()
```

The MUSA runtime is built with the qualified shape-specialized configuration,
including `EXECUTORCH_MUSA_GGUF_SPLIT_K=ON`. The options do not change GGUF
weights or quantization. `share_attention_prefix=True` creates one host prefix
control snapshot for the eight full-attention layers instead of eight separate
GPU-to-host snapshots per decode step.

## Validation

The combined export has one delegate, 64 mutated state buffers, 32 merged FFN
calls, 64 split-K projection calls, eight prefix-attention consumers and one
prefix-control producer. Its six-step eager outputs pass the existing HF and
previous-export tolerances. A native text generation reaches the original EOS
with the same 89 generated token IDs as the qualified baseline.

## Performance

The controlled A/B series uses the Qwen3.5-4B Q4_K_M workload, 28 prompt
tokens, 88 decode steps, one warmup and two measured runs in eight interleaved
processes. Profiling, launch blocking and logit dumping are disabled.

| Variant | Decode minimum | Decode median | Decode maximum | Serial prefill median |
| --- | ---: | ---: | ---: | ---: |
| Split-K baseline | 10.0546 | 12.2392 | 13.5513 | 11.8293 |
| Combined candidate | 13.0189 | 13.2821 | 13.8257 | 13.6143 |

The candidate median improves **8.52%** and its minimum remains above 10
tokens/s. These numbers are for the shared-desktop M1000 MP22 setup and are not
an exclusive-device guarantee.

The corresponding nonblocking trace confirms the cause rather than relying on
timing alone:

| Per decode step | Split-K baseline | Combined candidate |
| --- | ---: | ---: |
| `musaStreamSynchronize` | 12 | 5 |
| `musaMemcpyAsync` | 21 | 7 |
| `musaLaunchKernel` | 386 | 386 |

Kernel launch count is unchanged; the gain comes from removing repeated
attention-control readbacks and their stream barriers. Profiled throughput is
not comparable to ordinary throughput because the profiler changes timing and
asynchronous labels overlap.

Evidence is retained under the local ETSupport artifact directory:
`split-k-shared-prefix-benchmark-v1/result.json`,
`global-split-k-shared-prefix-profile-v1/analysis-shared-prefix-v1.json`, and
`decode-split-k-shared-prefix-export-v1/result.json`.

The individual options remain opt-in. This document records the qualified
combined export recipe; it does not change generic model defaults.
