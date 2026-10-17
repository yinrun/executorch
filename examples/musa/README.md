# MUSA AOTI model acceptance

`aoti_model_acceptance.py` validates the complete MUSA path for fourteen models:

- `mlp`: Linear + ReLU + Linear + ReLU;
- `simple_cnn`: Conv2d + ReLU + MaxPool2d + Conv2d + Flatten + Linear;
- `simple_video_cnn`: Conv3d + ReLU + MaxPool3d + Conv3d + pooling + Linear;
- `simple_decoder`: ConvTranspose2d + ReLU + ConvTranspose2d with output padding;
- `simple_audio_decoder`: ConvTranspose1d + ReLU + ConvTranspose1d with output padding;
- `simple_video_decoder`: ConvTranspose3d + ReLU + ConvTranspose3d with output padding;
- `transformer_block`: torchvision-independent TransformerEncoderLayer;
- `causal_attention`: causal grouped-query attention with Q/K/V projections;
- `embedding_classifier`: int64 token IDs, Embedding, pooling, and Linear;
- `token_encoder`: int64 token IDs, Embedding, TransformerEncoderLayer, and Linear;
- `token_decoder`: int64 token IDs, Embedding, causal GQA, LayerNorm, and vocabulary projection;
- `narrow_resnet18`: ResNet-18 block topology with narrow channels and a 32x32 input;
- `resnet18`: unmodified torchvision ResNet18 with random weights and a 64x64 input;
- `mobilenet_v2`: torchvision MobileNetV2 with depthwise convolutions and a 64x64 input.

The script writes the PTE, external AOTI data, raw input, expected output, and
runner output to the selected directory. It compares the `executor_runner`
output with the MUSA eager result; no CPU execution is used as a correctness
baseline.

Floating inputs and weights default to FP32. `--dtype fp16` or `--dtype bf16` exercises
the same export and runner path in lower precision. The default absolute and
relative comparison tolerances are 1e-4 (FP32), 1e-2 (FP16), and 4e-2 (BF16);
token models always accept raw int64 input IDs and `--dtype` selects the
weights and output dtype. For `token_decoder`, the script also checks exact
argmax token IDs against MUSA eager.
override them with `--atol` and `--rtol` if needed. `--input-size 224` exports
ResNet or MobileNet with a 224x224 input instead of its default size. Fixed
spatial shapes and dtypes require separate exports.
`--batch-size 2` exports and checks a separate fixed-shape, two-input batch
(the default batch size is one).

`--batch-size 2 --dynamic-batch-max 4` instead exports one PTE with a bounded
dynamic batch dimension and checks batch sizes 2, 1, and 4 against MUSA eager.
The script pads raw input files to the bound and passes the actual shape via
`executor_runner --input_shapes`; that runner otherwise interprets the raw
buffer as a bound-sized tensor. Other dynamic dimensions are not covered.
For torchvision models, `--weights /path/to/state_dict.pth` loads a
local PyTorch state dictionary (for example, a cached pretrained checkpoint)
before exporting. Without it, the model uses random weights; the script never
downloads weights automatically.

The MUSA backend disables TorchInductor's layout optimization so Conv1d/2d/3d
fallback outputs and generated kernels agree on contiguous NCHW layout. The
eager reference disables muDNN TF32 to match the fallback's FP32 FMA math.
The native fallback supports non-transposed and transposed Conv1d/2d/3d with
output padding smaller than its stride. Transposed Conv1d uses muDNN's GEMM
backward-data algorithm; its implicit-GEMM algorithm rejects the 1D descriptor.
MUSA AOTI decomposes inference-only SDPA into matmul, softmax, and masking to
avoid an unavailable TorchMUSA AOTI fallback. Pattern matching is disabled
only while compiling graphs containing SDPA or attention-like softmax/matmul
patterns, as it otherwise fuses those ops back into the unavailable fallback.
Dropout is unsupported. The current
TorchMUSA export specializes attention batch dimensions, so bounded dynamic
batch acceptance is not yet available for models containing attention.
Boolean/additive masks, causal attention, and GQA are covered separately;
combining a causal flag with a custom mask is not supported.
All-masked rows are not covered: math SDPA can yield NaNs, while TorchMUSA's
low-precision flash attention can behave differently.

## Prerequisites

- MUSA and `torch_musa` are available to the selected Python environment;
- the MUSA-compatible Triton package required by TorchInductor is installed;
- `torchvision` is available for the standard `resnet18` model;
- `flatc` is available on `PATH` for PTE/PTD serialization;
- an ExecuTorch `executor_runner` built with `EXECUTORCH_BUILD_MUSA=ON`.

For example, from the repository root:

```bash
export PYTHON=/usr/bin/python3.12
export PYTHONPATH=/tmp/executorch-pydeps:$PWD/src:$PWD
export PATH=/tmp/executorch-flatc-build:$PATH
export RUNNER=/tmp/executorch-musa-cmake-1792075449/executor_runner

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model mlp \
  --output-dir /tmp/executorch-musa-acceptance/mlp \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model simple_cnn \
  --output-dir /tmp/executorch-musa-acceptance/simple_cnn \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model simple_audio_decoder --dtype bf16 \
  --output-dir /tmp/executorch-musa-acceptance/simple_audio_decoder_bf16 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model simple_video_decoder --dtype bf16 \
  --output-dir /tmp/executorch-musa-acceptance/simple_video_decoder_bf16 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model resnet18 \
  --output-dir /tmp/executorch-musa-acceptance/resnet18 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model resnet18 --input-size 224 \
  --output-dir /tmp/executorch-musa-acceptance/resnet18_224 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model mobilenet_v2 --batch-size 2 --input-size 64 \
  --output-dir /tmp/executorch-musa-acceptance/mobilenet_v2_batch2 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model simple_cnn --batch-size 2 --dynamic-batch-max 3 \
  --output-dir /tmp/executorch-musa-acceptance/simple_cnn_dynamic \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model causal_attention --dtype bf16 \
  --output-dir /tmp/executorch-musa-acceptance/causal_attention_bf16 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model token_decoder --dtype bf16 \
  --output-dir /tmp/executorch-musa-acceptance/token_decoder_bf16 \
  --runner "$RUNNER"
```

To validate official pretrained ResNet18 weights, download
`https://download.pytorch.org/models/resnet18-f37072fd.pth` separately and pass
the local checkpoint with `--weights`. Its SHA-256 is
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
This checks inference against MUSA eager using a random input, not ImageNet
accuracy or performance.
MobileNetV2 IMAGENET1K_V2 weights are available at
`https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth` (SHA-256
`7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f`).

On M1000 with torch-musa 2.9.0.post1+musa5.1.2mp22, the following exports
passed `executor_runner` versus MUSA eager (maximum absolute output error):

| Model | Input | Dtype | Weights | Max error |
| --- | --- | --- | --- | ---: |
| MLP | 1x16 | FP16 / BF16 | Random | 4.88e-4 / 3.91e-3 |
| Simple CNN | 1x3x32x32 | FP16 / BF16 | Random | 2.44e-4 / 1.95e-3 |
| Simple 3D CNN | 1x1x4x8x8 | FP32 | Random | 7.45e-9 |
| Simple decoder | 1x3x8x8 | FP32 | Random | 5.96e-8 |
| Simple audio decoder | 1x2x5 | FP32 / FP16 / BF16 | Random | 2.98e-8 / 2.44e-4 / 1.95e-3 |
| Simple video decoder | 1x1x2x3x4 | FP32 / FP16 / BF16 | Random | 0 / 0 / 0 |
| TransformerEncoderLayer | 1x4x16 | FP32 / FP16 / BF16 | Random | 4.77e-7 / 1.95e-3 / 7.81e-3 |
| Causal GQA | 1x4x16 | FP32 / FP16 / BF16 | Random | 1.19e-7 / 1.22e-4 / 3.91e-3 |
| Token encoder | 1x4 int64 | FP32 / FP16 / BF16 | Random | 5.96e-8 / 4.89e-4 / 1.96e-3 |
| Token decoder | 1x4 int64 | FP32 / FP16 / BF16 | Random | 1.79e-7 / 4.89e-4 / 3.91e-3 |
| ResNet18 | 1x3x224x224 | FP32 / FP16 | Official pretrained | 6.05e-6 / 3.91e-3 |
| ResNet18 | 1x3x64x64 | BF16 | Official pretrained | 0 |
| MobileNetV2 | 1x3x224x224 | FP32 | Official pretrained | 2.03e-6 |
| MobileNetV2 | 1x3x64x64 | FP16 | Official pretrained | 1.37e-2 (`--atol 0.02`) |
| ResNet18 | 2x3x64x64 | FP32 | Official pretrained | 3.34e-5 |
| MobileNetV2 | 2x3x64x64 | FP32 | Official pretrained | 1.76e-5 |

With a single dynamic-batch PTE for each model, batches 2/1/4 of FP32 MLP
passed (maximum error 4.47e-8). Batches 2/1/3 also passed for FP32 simple CNN
(3.58e-7), pretrained ResNet18 (3.53e-5), and pretrained MobileNetV2
(1.76e-5); both pretrained models used 64x64 inputs.
For the simple 3D CNN, batches 2/1/3 passed with a single PTE in both FP16
(maximum error 6.11e-5) and BF16 (4.89e-4).
For the simple decoder, batches 2/1/3 passed with a single PTE in both FP16
and BF16 (maximum error 0 for this test input).
The embedding classifier also passed batches 2/1/3 in FP32 with one PTE
(maximum error 1.20e-7), confirming dynamic int64 inputs without attention.
The token decoder matched eager token IDs `[4, 13, 56, 44]` in all three
precisions. It is a fixed-sequence toy model, not a pretrained LLM or a
KV-cache/incremental decoding validation.

`aoti_kv_cache_acceptance.py` separately exercises two consecutive single-token
decode steps with a rolling four-position KV cache. The same fixed-shape PTE
accepts int64 token IDs plus key/value tensors and returns logits plus updated
key/value tensors; the second `executor_runner` invocation reads the first
invocation's raw cache outputs. Every output is compared with MUSA eager and
the predicted token IDs must match exactly:

```bash
python examples/musa/aoti_kv_cache_acceptance.py \
  --output-dir /tmp/musa-kv-cache --runner /path/to/executor_runner \
  --dtype fp32
```

The default rolling mode tests raw multi-input/multi-output cache roundtrips,
not dynamic cache growth, batched decoding, a pretrained LLM, or throughput.
The oldest cache position is discarded at each step to keep all shapes static.
On M1000, FP32/FP16/BF16 all passed both steps with eager-matching predicted
token IDs `[3, 32]`; maximum logits differences were 2.38e-7, 9.77e-4, and
3.91e-3 respectively. Both returned caches also matched MUSA eager at each step.

Pass `--cache-mode growing` to test an append-only cache instead. One PTE with
bounded dynamic key/value sequence dimensions runs with past lengths 3 then 4
and returns updated cache lengths 4 then 5. The first input is padded to the
exported upper bound and resized with `--input_shapes`; the second runner call
reads the preceding runner output cache files directly. Pass
`--initial-cache-length 2` to exercise past lengths 2 then 3; its second input
is padded to the upper bound before the runner reads it. On M1000, FP32 passed
both starting lengths with maximum logits differences 2.38e-7 (3/4) and
1.91e-7 (2/3); FP16/BF16 passed the 3/4 case with zero output difference for
this input. All predicted token IDs matched MUSA eager. This does not establish
arbitrary sequence lengths beyond the exported input bound (2–4), variable
batches, or pretrained-model support.

`aoti_prefill_decode_acceptance.py` exports a separate three-token causal
prefill and bounded-dynamic decode PTE from the *same* toy decoder weights.
The prefill runner produces logits and key/value caches; its output files are
padded to the decode bound for the first call, whose cache outputs feed a
second decode call without padding. Runner outputs are checked against MUSA
eager at every step, including exact
argmax token IDs:

```bash
python examples/musa/aoti_prefill_decode_acceptance.py \
  --output-dir /tmp/musa-prefill-decode --runner /path/to/executor_runner \
  --dtype fp32
```

The prefill returns transposed key/value views. The shared CUDA-compatible
runtime now copies noncontiguous SlimTensor outputs with nonzero storage offsets
to planned ET tensors; the decode export uses contiguous cache inputs to match
the runner's raw file layout. On M1000, FP32 prefill and two decode calls passed
with maximum logits errors of 2.99e-7, 1.20e-7, and 1.50e-7 respectively;
FP16 and BF16 also passed with exact matching token IDs `[16, 40, 32]`.
This is a randomly initialized toy model, not a pretrained LLM or general
variable-length prefill acceptance.
Use `--dynamic-prefill` to export one bounded-dynamic prefill PTE and run
prompts of lengths 2, 3, and 4 against MUSA eager before decoding the
three-token prompt. Input buffers are padded to the export bound and their
actual shape is supplied through `--input_shapes`. On M1000, FP32, FP16, and
BF16 passed all three prompt lengths and both subsequent decode steps with
their respective pairs of PTEs.

`aoti_llama_acceptance.py` exercises the repository's actual five-layer Llama
architecture (RoPE, RMSNorm, SwiGLU, and attention) using its bundled
`demo_rand_params.pth` checkpoint and `demo_config.json`. It exports a PTE/PTD,
runs `executor_runner` on a three-token prompt, and compares all 512 logits
and the exact predicted token ID with MUSA eager. It also rejects unlowered
operators outside the single MUSA delegate:

```bash
python examples/musa/aoti_llama_acceptance.py \
  --output-dir /tmp/musa-llama-demo --runner /path/to/executor_runner
```

This checkpoint contains **random weights**, not a pretrained Llama. The
validation covers one fixed prompt with no KV cache; it establishes neither
useful language generation nor performance for a production model.
On M1000, FP32/FP16/BF16 all predicted token ID `364`; maximum absolute logit
errors against MUSA eager were 4.77e-7, 1.46e-3, and 1.17e-2, respectively.

`aoti_llama_kv_acceptance.py` exports the same five-layer Llama demo with its
upstream in-module KV cache enabled and runs five single-token steps through
one `executor_runner --server_mode` process and one PTE/PTD. Positions 0–2
prefill `[11, 7, 22]` token-by-token, and positions 3–4 decode `[5, 17]`.
Every 512-logit output and predicted token ID is checked against a separately
advanced MUSA eager model:

```bash
python examples/musa/aoti_llama_kv_acceptance.py \
  --output-dir /tmp/musa-llama-kv --runner /path/to/executor_runner
```

For this export the KV buffers live inside the AOTI delegate; restarting the
runner resets the cache, so all steps must execute in one process. The export
uses a fixed one-token input and an eight-position context bound, rather than
variable-length prompt prefill or externally serialized caches. On M1000,
FP32/FP16/BF16 passed all five steps with maximum logit errors 6.56e-7,
1.50e-3, and 1.56e-2, respectively, and predicted IDs
`[33, 351, 364, 76, 464]`. This still uses **random**, not pretrained, weights.
The upstream model passes its position tensor inside an `attn_options` dict;
the shared AOTI compiler now reconstructs that input tree. Its mutable cache
graph also requires MUSA's device assertion to use the actual indexed device
instead of the unindexed device string produced by PyTorch's device move pass.

Pass `--dynamic-prefill` to export one bounded-dynamic Llama KV PTE that accepts
one to seven tokens per call within its eight-position context. The runner
processes `[11, 7, 22]` together, then `[5]` and `[17]` separately in the
**same process**. A second, fresh runner process uses the **same PTE/PTD** to
prefill `[11, 7, 22, 5, 17, 13, 19]` in one call and decode `[3]` at the final
context position. Server requests carry
`input_shapes=1xL,L` to resize the token and position inputs on each step;
upper-bound-sized raw input files keep the storage valid. TorchMUSA currently
lacks a fake implementation for its dynamic SDPA math operator, so this export
uses an explicit masked matmul/softmax attention module. Each step is checked
both against that module's MUSA eager output and the original upstream SDPA
module's eager output. On M1000, FP32/FP16/BF16 passed all three calls with
predicted IDs `[364, 76, 464]` and maximum runner-vs-eager logit differences
5.97e-7, 1.47e-3, and 9.77e-3, respectively. The long prefill and final decode
also passed all three dtypes with predicted IDs `[103, 344]` and maximum logit
differences 4.77e-7, 9.77e-4, and 1.18e-2. This does not yet export the
unaltered upstream SDPA model with dynamic sequence lengths: registering a fake
kernel for TorchMUSA's math SDPA operator alone still causes PyTorch export to
specialize the sequence length to its example value.

These are random-input correctness checks, not arbitrary shape-generalization
or throughput measurements. Pretrained MobileNetV2 BF16 at 64x64 does not yet
meet the default tolerance: the maximum difference is 0.140625 and 121/1000
logits differ beyond `--atol 0.02 --rtol 0.04`. Do not count this configuration
as verified until the precision difference is understood. In isolated prefix
exports, the first two feature blocks match bit-for-bit. The next block's
pointwise Conv2d also matches, but its following BatchNorm differs at 4 of
98,304 BF16 values (maximum 0.03125); the discrepancy grows through later
blocks. Investigate BF16 BatchNorm rounding before relaxing the tolerance.

Without `--runner`, the script only exports the PTE/PTD and writes the eager
reference files. The runner command automatically adds the MUSA, Torch, and
Python shared-library directories used by the current environment.
