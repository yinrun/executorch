# MUSA AOTI model acceptance

## Native device metadata and compatibility

New `MusaPartitioner` exports default to `target_device=musa:0`. PTE tensor
metadata, memory-planning arenas, weight descriptors, and SlimTensor use a
distinct `MUSA=2` device type; CPU remains 0 and CUDA remains 1. MUSA weight
storage keys use `musa_fqn_weight:musa:`. The shared weight container magic
`ETCUDAFQN0` and the PTD flat-tensor layout are unchanged.

These serialized values are **not** PyTorch AOTI device IDs: TorchMUSA uses
`PrivateUse1=20`. The shim boundary translates MUSA metadata to/from 20;
passing serialized value 2 as an AOTI device ID is rejected. Native MUSA
SlimTensors report `is_musa()` and `is_gpu()`, but not `is_cuda()`.

A MUSA runtime also accepts existing `MusaBackend` artifacts whose tensor,
arena, and weight descriptors use the legacy CUDA tag 1. It registers both
native and legacy allocators and normalizes legacy SlimTensor device tags
to MUSA. This is compatibility with earlier **MUSA** exports, not execution
of CUDA binaries: a MUSA build registers `MusaBackend` only and rejects a
`CudaBackend` delegate. CUDA/ROCm builds retain their existing CUDA-tagged
path and do not allocate MUSA tensors. Older runtimes without the new enum
cannot run native-MUSA artifacts; rebuild the runtime together with the
exporter.

The implementation still shares the `backends/cuda` runtime, allocator,
and SlimTensor GPU storage code through MUSA API adapters. Native device
metadata does not imply that standalone packaging, full ATen-mode runtime
integration, or CUDA/ROCm hardware regression testing is complete.

## Model coverage

`aoti_model_acceptance.py` exercises the complete MUSA path for these models:

- `mlp`: Linear + ReLU + Linear + ReLU;
- `simple_cnn`: Conv2d + BatchNorm2d + ReLU + MaxPool2d + Conv2d + Flatten + Linear;
- `simple_audio_cnn`: Conv1d + BatchNorm1d + ReLU + strided Conv1d + AvgPool1d + Linear;
- `simple_video_cnn`: Conv3d + BatchNorm3d + ReLU + MaxPool3d + Conv3d + pooling + Linear;
- `simple_decoder`: ConvTranspose2d + ReLU + ConvTranspose2d with output padding;
- `simple_audio_decoder`: grouped ConvTranspose1d + ReLU + ConvTranspose1d with output padding;
- `simple_video_decoder`: ConvTranspose3d + ReLU + ConvTranspose3d with output padding;
- `dilated_audio_decoder`, `dilated_image_decoder`, `dilated_video_decoder`: ConvTranspose1d/2d/3d with output padding smaller than dilation but not stride;
- `tiny_unet`: Conv2d, GroupNorm, SiLU, down/up sampling, concatenated skip connection, and ConvTranspose2d;
- `bilinear_unet`: Conv2d, GroupNorm, SiLU, bilinear upsampling to the skip size, and concatenation;
- `transformer_block`: torchvision-independent TransformerEncoderLayer;
- `causal_attention`: causal grouped-query attention with Q/K/V projections;
- `embedding_classifier`: int64 token IDs, Embedding, pooling, and Linear;
- `token_encoder`: int64 token IDs, Embedding, TransformerEncoderLayer, and Linear;
- `token_decoder`: int64 token IDs, Embedding, causal GQA, LayerNorm, and vocabulary projection;
- `narrow_resnet18`: ResNet-18 block topology with narrow channels and a 32x32 input;
- `resnet18`: unmodified torchvision ResNet18 with random weights and a 64x64 input;
- `mobilenet_v2`: torchvision MobileNetV2 with depthwise convolutions and a 64x64 input.
- `efficientnet_b0`: torchvision EfficientNet-B0 with squeeze-and-excitation, SiLU, and depthwise convolutions;
- `densenet121`: torchvision DenseNet121 with dense concatenation blocks and transition pooling;
- `convnext_tiny`: torchvision ConvNeXt-Tiny with depthwise convolutions, LayerNorm, GELU, and residuals;
- `vit_b_16`: torchvision ViT-B/16 with twelve Transformer blocks, multihead attention, and LayerNorm;
- `deeplabv3_mobilenet_v3_large`: torchvision DeepLabV3 segmentation with a MobileNetV3 backbone and an output of 21-class pixel logits.

The script writes the PTE, external AOTI data, raw input, expected output,
runner output, and successful `result.json` report to the selected directory.
Without `--runner`, the report records an export only (`runner` is null).
It compares the `executor_runner`
output with the MUSA eager result; no CPU execution is used as a correctness
baseline. The GPU random input generator is synchronized before taking an
immutable host snapshot; eager execution and the runner read the same input.
Without that synchronization, an odd-sized batched input produced a raw file
with incomplete tail values even though the eager output was finite.

Floating inputs and weights default to FP32. `--dtype fp16` or `--dtype bf16` exercises
the same export and runner path in lower precision. The default absolute and
relative comparison tolerances are 1e-4 (FP32), 1e-2 (FP16), and 4e-2 (BF16);
token models always accept raw int64 input IDs and `--dtype` selects the
weights and output dtype. For `token_decoder`, the script also checks exact
argmax token IDs against MUSA eager. For EfficientNet-B0 and ViT-B/16 it checks
predicted class IDs, and for DeepLabV3 it checks exact per-pixel predicted labels.
Override the tolerances with `--atol` and
`--rtol` if needed. `--input-size 224` exports a supported torchvision
image model with a 224x224 input instead of its default size. Except
for the bounded dynamic-spatial `bilinear_unet` and DeepLabV3 cases below,
fixed spatial shapes and dtypes require separate exports.
`--batch-size 2` exports and checks a separate fixed-shape, two-input batch
(the default batch size is one).

`--batch-size 2 --dynamic-batch-max 4` instead exports one PTE with a bounded
dynamic batch dimension and checks batch sizes 2, 1, and 4 against MUSA eager.
The script pads raw input files to the bound and passes the actual shape via
`executor_runner --input_shapes`; that runner otherwise interprets the raw
buffer as a bound-sized tensor. For `bilinear_unet` only,
`--dynamic-spatial-max 48` exports one PTE with independent height and width
between 24 and 48. It checks 32x40, 24x30, 48x48, and 47x39 using that PTE.
For `resnet18`, `mobilenet_v2`, `deeplabv3_mobilenet_v3_large`, and `efficientnet_b0`,
`--dynamic-spatial-max 80` checks
64x64, 48x54, 80x80, and 79x71 using one PTE with independent height and
width between 48 and 80. Both include odd sizes. Dynamic batch and spatial
dimensions can also be combined for these supported models using
`--batch-size 2 --dynamic-batch-max 3 --dynamic-spatial-max 80` (use 48 for
the bilinear U-Net). This accepts a batch of 1–3 with independent bounded
height and width in one PTE.
ResNet18 and MobileNetV2 use `export_with_musa_pooling_shapes` for this path.
The installed PyTorch pooling meta helper tests the remainder of a symbolic
integer division even though Python `//` already rounds toward negative
infinity. This adds an unnecessary parity constraint to the ResNet stem's
pooling input. The helper temporarily uses `operator.floordiv` during export,
restores the original function on success or failure, and leaves the actual
pooling operators and model architecture unchanged. It preserves independent
integer height/width bounds rather than requiring multiples of four. As with
the SDPA export helper, export calls using this compatibility patch must be
serialized.
For `densenet121`, the tested export range uses `8 * Dim` for independent
height and width between 32 and 80, checking 64x64, 32x40, 80x80, and 72x64.
This constraint comes from the current exporter's pooling guards; it is not a
claim that eager DenseNet cannot process other sizes. The example size and
maximum must be multiples of 8, and the maximum must exceed 32.
`convnext_tiny` supports fixed spatial sizes and dynamic batch acceptance.
Other dynamic dimensions are not covered.
For torchvision models, `--weights /path/to/state_dict.pth` loads a
local PyTorch state dictionary (for example, a cached pretrained checkpoint)
before exporting. Without it, the model uses random weights; the script never
downloads weights automatically.

The MUSA backend disables TorchInductor's layout optimization so Conv1d/2d/3d
fallback outputs and generated kernels agree on contiguous NCHW layout. The
eager reference disables muDNN TF32 to match the fallback's FP32 FMA math.
The native fallback supports non-transposed and transposed Conv1d/2d/3d.
The MUSA BMM fallback supports contiguous output and contiguous or transposed
left- and right-hand matrices with safe batch padding. This covers the
non-contiguous batch view of ViT attention without copying the matrices;
arbitrary per-row strides are not yet supported.
Single-row BMM operands may have a zero stride for that singleton row axis,
as in the fixed-token upstream Llama KV-cache export; the leading batch
stride must not overlap distinct matrices unless the operand is a read-only
broadcast with zero batch stride. Broadcast batches use individual GEMM calls
instead of a strided-batched GEMM.
The `transposed_bmm` acceptance model exercises a transposed left-hand matrix
through PTE/PTD export and `executor_runner` output comparison, including
bounded dynamic batch sizes.
The `broadcast_bmm` model broadcasts a read-only right-hand weight across
batches using a zero inter-batch stride; output matrices remain disjoint.
Earlier standalone addmm tests exposed intermittent padded-row FP16/BF16
errors. The subsequent packing and GEMM-path fixes described below now pass
repeated native regressions, including the re-enabled no-bias padded case;
this does not establish support for arbitrary matrix strides.
For transposed convolution, output padding must be smaller than either its
stride or its dilation, as required by PyTorch. Transposed Conv1d uses muDNN's
GEMM backward-data algorithm; its implicit-GEMM algorithm rejects the 1D
descriptor.
MUSA AOTI decomposes inference-only SDPA into matmul, softmax, and masking to
avoid an unavailable TorchMUSA AOTI fallback. Pattern matching is disabled
only while compiling graphs containing SDPA or attention-like softmax/matmul
patterns, as it otherwise fuses those ops back into the unavailable fallback.
Dropout is unsupported. TorchMUSA's native attention export specializes batch
dimensions. For `transformer_block`, `causal_attention`, and `vit_b_16` with
bounded dynamic batch, the acceptance script instead expands calls through
`torch.nn.functional.scaled_dot_product_attention` into the backend's math
implementation before export. Other callers can explicitly use
`export_with_musa_sdpa_decomposition` from `executorch.backends.musa`. This
temporarily replaces a process-global Python function during export, so do not
run concurrent exports, and calls bound to a separate SDPA alias are not covered.
Boolean/additive masks, causal attention, and GQA are covered separately;
combining a causal flag with a custom mask is not supported.
All-masked rows are not covered: on M1000, FP32 SDPA and the backend math path
return NaNs, while low-precision TorchMUSA flash attention returned nonzero
values for both boolean and additive masks in a 3-token probe. Silently mapping
these cases to zeros or flash output would change masking semantics.

## Prerequisites

- MUSA and `torch_musa` are available to the selected Python environment;
- the MUSA-compatible Triton package required by TorchInductor is installed;
- `torchvision` is available for the standard ResNet18, MobileNetV2, EfficientNet-B0, ViT-B/16, and DeepLabV3 models;
- `flatc` is available on `PATH` for PTE/PTD serialization;
- an ExecuTorch `executor_runner` built with `EXECUTORCH_BUILD_MUSA=ON`.

For a reproducible build plus native/compiler/runner regression entry point,
see [MUSA regression instructions](../../backends/musa/tests/README.md).
It builds the runner and native test targets directly from this checkout,
without hand-linking test binaries or installing the C++ libraries first.

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
  --model transposed_bmm --batch-size 2 --dynamic-batch-max 3 \
  --output-dir /tmp/executorch-musa-acceptance/transposed_bmm \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model broadcast_bmm --batch-size 2 --dynamic-batch-max 3 \
  --output-dir /tmp/executorch-musa-acceptance/broadcast_bmm \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model transformer_block --dtype fp16 \
  --batch-size 2 --dynamic-batch-max 3 --runner-repeats 4 \
  --output-dir /tmp/executorch-musa-acceptance/transformer_block_dynamic \
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
  --model tiny_unet --dtype bf16 \
  --output-dir /tmp/executorch-musa-acceptance/tiny_unet_bf16 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model bilinear_unet --dtype bf16 --dynamic-spatial-max 48 \
  --output-dir /tmp/executorch-musa-acceptance/bilinear_unet_dynamic \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model bilinear_unet \
  --batch-size 2 --dynamic-batch-max 3 --dynamic-spatial-max 48 \
  --output-dir /tmp/executorch-musa-acceptance/bilinear_unet_batch_spatial \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model efficientnet_b0 --input-size 224 --dtype bf16 \
  --weights /path/to/efficientnet_b0_rwightman-7f5810bc.pth \
  --output-dir /tmp/executorch-musa-acceptance/efficientnet_b0_pretrained \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model efficientnet_b0 --batch-size 2 --dynamic-batch-max 3 \
  --weights /path/to/efficientnet_b0_rwightman-7f5810bc.pth \
  --output-dir /tmp/executorch-musa-acceptance/efficientnet_b0_dynamic \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model efficientnet_b0 --batch-size 2 --dynamic-batch-max 3 \
  --dynamic-spatial-max 80 \
  --weights /path/to/efficientnet_b0_rwightman-7f5810bc.pth \
  --output-dir /tmp/executorch-musa-acceptance/efficientnet_b0_batch_spatial \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model vit_b_16 --dtype fp16 \
  --weights /path/to/vit_b_16-c867db91.pth \
  --output-dir /tmp/executorch-musa-acceptance/vit_b_16_pretrained \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model vit_b_16 --batch-size 2 --dynamic-batch-max 3 \
  --weights /path/to/vit_b_16-c867db91.pth \
  --output-dir /tmp/executorch-musa-acceptance/vit_b_16_dynamic \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model vit_b_16 --dtype fp16 --batch-size 2 --dynamic-batch-max 3 \
  --vit-blocks 6 --weights /path/to/vit_b_16-c867db91.pth \
  --output-dir /tmp/executorch-musa-acceptance/vit_b_16_prefix6 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model vit_b_16 --dtype fp16 --batch-size 2 --dynamic-batch-max 3 \
  --vit-layer-tokens --vit-attention-stages --runner-repeats 5 \
  --weights /path/to/vit_b_16-c867db91.pth \
  --output-dir /tmp/executorch-musa-acceptance/vit_b_16_layer_tokens \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model deeplabv3_mobilenet_v3_large \
  --weights /path/to/deeplabv3_mobilenet_v3_large-fc3c493d.pth \
  --output-dir /tmp/executorch-musa-acceptance/deeplabv3_pretrained \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model deeplabv3_mobilenet_v3_large \
  --weights /path/to/deeplabv3_mobilenet_v3_large-fc3c493d.pth \
  --batch-size 2 --dynamic-batch-max 3 \
  --output-dir /tmp/executorch-musa-acceptance/deeplabv3_dynamic \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model deeplabv3_mobilenet_v3_large \
  --weights /path/to/deeplabv3_mobilenet_v3_large-fc3c493d.pth \
  --dynamic-spatial-max 80 \
  --output-dir /tmp/executorch-musa-acceptance/deeplabv3_dynamic_spatial \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model deeplabv3_mobilenet_v3_large \
  --weights /path/to/deeplabv3_mobilenet_v3_large-fc3c493d.pth \
  --batch-size 2 --dynamic-batch-max 3 --dynamic-spatial-max 80 \
  --output-dir /tmp/executorch-musa-acceptance/deeplabv3_batch_spatial \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model dilated_video_decoder --dtype bf16 \
  --output-dir /tmp/executorch-musa-acceptance/dilated_video_decoder_bf16 \
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
  --model simple_video_decoder --dtype bf16 \
  --batch-size 2 --dynamic-batch-max 3 \
  --output-dir /tmp/executorch-musa-acceptance/video_decoder_dynamic \
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
EfficientNet-B0 IMAGENET1K_V1 weights are available at
`https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth`
(SHA-256 `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934`).
ViT-B/16 IMAGENET1K_V1 weights are available at
`https://download.pytorch.org/models/vit_b_16-c867db91.pth`
(SHA-256 `c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9`).
ViT-B/16 requires 224x224 inputs. Its untrained classifier produces all-zero
logits, so only the pretrained checkpoint provides a meaningful output check.
DeepLabV3 MobileNetV3 COCO-with-VOC-labels weights are available at
`https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth`
(SHA-256 `fc3c493d68e89cc31ef488c803d5d7dd2f3190fb570598faa49fef69be8e5e70`).
DeepLabV3 loads the full checkpoint, including its auxiliary classifier, but
exports only the primary `out` logits. The acceptance input is random, not
normalized dataset images; no segmentation accuracy is measured.

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
| Dilated audio decoder | 1x2x5 | FP32 / BF16 | Random | 5.97e-8 / 3.91e-3 |
| Dilated image decoder | 1x2x5x4 | FP32 / BF16 | Random | 0 / 0 |
| Dilated video decoder | 1x2x5x4x3 | FP32 / BF16 | Random | 0 / 0 |
| Tiny U-Net | 1x3x32x32 | FP32 / FP16 / BF16 | Random | 6.56e-7 / 1.95e-3 / 1.56e-2 |
| Bilinear U-Net | 1x3x32x40 | FP32 / BF16 | Random | 2.38e-7 / 7.81e-3 |
| TransformerEncoderLayer | 1x4x16 | FP32 / FP16 / BF16 | Random | 4.77e-7 / 1.95e-3 / 7.81e-3 |
| Causal GQA | 1x4x16 | FP32 / FP16 / BF16 | Random | 1.19e-7 / 1.22e-4 / 3.91e-3 |
| Token encoder | 1x4 int64 | FP32 / FP16 / BF16 | Random | 5.96e-8 / 4.89e-4 / 1.96e-3 |
| Token decoder | 1x4 int64 | FP32 / FP16 / BF16 | Random | 1.79e-7 / 4.89e-4 / 3.91e-3 |
| ResNet18 | 1x3x224x224 | FP32 / FP16 | Official pretrained | 6.05e-6 / 3.91e-3 |
| ResNet18 | 1x3x64x64 | BF16 | Official pretrained | 0 |
| ResNet18 | 1x3x224x224 | BF16 | Official pretrained | 0 |
| MobileNetV2 | 1x3x224x224 | FP32 | Official pretrained | 2.03e-6 |
| MobileNetV2 | 1x3x64x64 | FP16 | Official pretrained | 1.37e-2 (`--atol 0.02`) |
| EfficientNet-B0 | 1x3x64x64 | FP32 | Official pretrained | 7.39e-6 (class 644) |
| EfficientNet-B0 | 1x3x224x224 | FP32 / FP16 / BF16 | Official pretrained | 1.79e-6 / 4.09e-3 / 2.83e-2 (class 314) |
| ViT-B/16 | 1x3x224x224 | FP32 | Official pretrained | 3.81e-6 (class 21) |
| ViT-B/16 | 1x3x224x224 | FP16 / BF16 | Official pretrained, initial single run only | 1.95e-3 / 3.13e-2; repeated execution fails |
| ViT-B/16 | batch 2/1/3 x3x224x224 | FP32 | Official pretrained, one dynamic PTE | <=7.15e-6 (class IDs match) |
| DeepLabV3 MobileNetV3 | 1x3x64x64 | FP32 | Official pretrained | 9.16e-5 (0/4096 differing pixel labels) |
| DeepLabV3 MobileNetV3 | 2x3x64x64 | FP32 | Official pretrained | 1.18e-4 (0/8192 differing pixel labels) |
| ResNet18 | 2x3x64x64 | FP32 | Official pretrained | 3.34e-5 |
| MobileNetV2 | 2x3x64x64 | FP32 | Official pretrained | 1.76e-5 |

With a single dynamic-batch PTE for each model, batches 2/1/4 of FP32 MLP
passed (maximum error 4.47e-8). Batches 2/1/3 also passed for FP32 simple CNN
(3.58e-7), pretrained ResNet18 (3.53e-5), and pretrained MobileNetV2
(1.76e-5); both pretrained models used 64x64 inputs.
Official pretrained EfficientNet-B0 also passed batches 2/1/3 using one FP32
PTE at 64x64 (maximum logit error 8.35e-6); predicted class IDs matched eager
for all batches. One dynamic-batch PTE for the FP32 TransformerEncoderLayer
also passed batches 2/1/3 (maximum error 4.77e-7). The FP16 model passed
An official pretrained ResNet18 BF16 PTE at 224x224 also passed dynamic batches
2/1/3 with maximum errors 0, 0, and 0.00390625, respectively.

The FP16 model passed
12 fresh-runner replays for each of batches 2/1/3 (maximum error 1.95e-3);
BF16 passed four executions for each batch (maximum error 1.56e-2).
`--runner-repeats` checks the same exported PTE/PTD and input against eager
on every execution, rather than relying on a single run. Official pretrained
FP32 ViT-B/16 passed the same batches at 224x224 using one PTE (maximum
error 7.15e-6); predicted class IDs matched MUSA eager for each input.
Dynamic ViT-B/16 FP16 and BF16 do **not** pass this check: the original
full-depth PTE produced finite batch-2 logits but 1,000 NaNs for batch 1,
while MUSA eager stayed finite. The `--vit-blocks` option truncates the
encoder after loading weights to help reproduce the issue. The
`--vit-layer-tokens` option outputs a `[batch, blocks + 1, 768]` tensor:
stage 0 is the class token after the positional embedding and stage `j + 1`
is the class token after encoder block `j`. On failure, the script reports
per-batch NaN counts at each stage without classifying them as image labels.
`--vit-attention-stages` adds a stage after each attention residual:
stage `2*j + 1` follows attention in block `j`, while stage `2*j + 2`
follows its MLP. In eight runs of a 12-block FP16 diagnostic PTE, the first
class-token NaNs in both batches appeared at odd-numbered stages, though the
block varied across runs. This implicates the attention half of the block
without proving which internal tensor first became invalid.

The acceptance exporter now uses `export_with_musa_sdpa_decomposition` for
both static and dynamic ViT-B/16 exports. This avoids the MUSA export-time
shape-specializing SDPA path; the backend already provides the same
decomposition for lowering. A newly exported static pretrained BF16 PTE passed
8/8 fresh runner replays with maximum error 0.03125, and a newly exported
static pretrained FP16 PTE passed 5/5 with maximum error 0.001953125. This
fixes the static acceptance path, but does not yet establish dynamic low
precision ViT support; the dynamic checks above remain failing.
The shim regression tests reproduce the ViT QKV projection layout with
`[197 or 394, 768] @ [768, 2304]`, a transposed weight and a broadcast bias.
Under repeated fresh processes, the standalone projection occasionally
produced a zero-valued prefix (4-16 elements) instead of the expected 2.5,
including FP32 and FP16 cases. One failed run read the input tensor's first
element back as zero too; this does not distinguish an input-transfer or
storage problem from corruption during GEMM. A device-wide synchronization
before GEMM and synchronous host-to-device copies did not eliminate the
intermittent failure. The projection stress test was initially disabled to
avoid flaky CI, and is enabled after the broadcast-copy improvement below.
The existing small addmm and BMM tests also intermittently returned zeros
in separate full-suite runs; the issue is not isolated to this new shape.
The separate ViT-sized attention-score BMM regression passed 30 fresh test
processes. These standalone tests do not yet prove the cause of the pretrained
low-precision model's NaNs: an extra QKV output inserted into a diagnostic
model compiled into a *second* projection GEMM, not an observation of the
original projection output.
Replacing the one-device-copy-per-row bias broadcast in `musa_addmm.cpp`
with non-overlapping doubling copies reduced a 197-row projection from 197
copies to 9. The standalone projection passed 120 fresh processes after
this change, versus intermittent failures before it; a five-row non-power-of-two
broadcast regression also passes. This does **not** fix the pretrained
low-precision ViT: replaying the same static PTE/PTD five times with the new
shim still yielded 1,000 NaN logits in 4/5 FP16 and 5/5 BF16 runs. The FP32
control passed 5/5 runs (maximum error 3.81e-6). More attention-path diagnosis
is required before counting pretrained low precision as supported. Across 12
full shim-test processes after this change, 7 passed and 5 still had
intermittent zero outputs, including the pre-existing padded addmm case and
an attention-score BMM case. The improved standalone projection is not a
general reliability fix.
A focused padded-mat1 check now compares transfers with and without GEMM:
three independent input allocations, including all seven physical elements
of a padded row layout, survived 100 fresh transfer-only processes. The
same layout with `beta=0` (no bias broadcast) returned second-row NaNs by
the third fresh process. A failing biased run returned the same NaNs on
repeated readback even though the original mat1, mat2, and bias read back
correctly afterward from distinct allocations. Thus neither bias broadcast
nor pageable device-to-host copies alone explains the remaining failure.
Packing padded mat1 into a contiguous scratch buffer still failed in an
isolated replay, as did synchronization before destroying the mublas handle;
both experiments were reverted. The no-bias stress check was disabled during
this investigation. After the later packing fix and repeated successful
validation, it is enabled in normal CI again; select it explicitly with
`--gtest_filter=MusaAddmmTest.PaddedMat1WithoutBias`.
These observations implicate the GEMM execution or its ordering but do not
identify a validated runtime or SDK fix.
For comparison, PyTorch MUSA eager `torch.addmm` with the same padded
`[2, 3]` input, `[3, 2]` weight, 1-D bias, three dtypes, and `beta=0/1`
matched exact hand-calculated results in 20 fresh processes, with ten
allocations per dtype per process (1,200 addmm calls total). Eager may use a
different algorithm, handle lifetime, or input layout; this contrast does
not establish which shim or SDK step fails. The next useful comparison is
the two paths' actual GEMM arguments, stream ordering, and input packing.
For a full-depth FP16 PTE in layer-token mode, five replays of the same
batch-2 input first produced NaNs at stages [none, 5], [4, 4], [6, 6],
[none, none], and [4, 2] for its two batches. Stage 0 and stage 1 stayed
finite in these runs, but the first failing block was not fixed. For a
six-block FP16 PTE, one acceptance run passed batches 2/1/3; repeating its
identical batch-2 input in fresh runner processes produced NaN counts of [0, 1000],
[1000, 1000], and [0, 1000]. A seven-block PTE first returned 2,000 NaNs
for batch 2, then three identical replays returned finite logits. These
results do not establish a specific failing layer or a reliable low-precision
dynamic path. Synchronizing after the full AOTI run also left all 2,000
six-block FP16 logits as NaNs in two separate replays. Device-wide
synchronization during some diagnostic matrix
operations removed FP16 NaNs, but synchronizing before mublas handle
destruction did not fix the full ViT; changing streams, retaining handles,
or disabling Inductor buffer reuse did not validate a fix. No synchronization
workaround is enabled for production. `MUSA_LAUNCH_BLOCKING=1` yielded five
finite six-block replays but only one of two finite full-depth FP16 replays
and no finite full-depth BF16 replays; it is not a validated workaround.
The static-batch pretrained FP16/BF16 entries above are not stable either:
their original PTE/PTD and input produced 1,000 NaN logits in each of eight
fresh runner processes per dtype, although the recorded first runs and MUSA
eager outputs were finite. A newly exported static FP16 PTE also returned
1,000 NaN logits on its first run. The issue is not limited to dynamic batch;
the original pretrained FP32 static PTE passed five identical replays with
maximum logit error 3.81e-6.

A subsequent MUSA-only shim change uses the same muDNN `MatMul` and
`BatchMatMul` classes as PyTorch MUSA eager for addmm/mm and packed,
non-broadcast BMM, including fused one-dimensional bias. Padded addmm input
rows are packed first; broadcast or padded BMM batches retain the existing
muBLAS path. SlimTensor's MUSA allocations use synchronous `cudaMalloc`
instead of stream-ordered allocation because the latter still gave intermittent
zero outputs in the focused matrix tests. This is not a measured performance
improvement and its allocation overhead and graph-capture compatibility need
evaluation. All 13 C++ shim tests, including the explicitly enabled padded
no-bias regression, passed in one fresh process; the focused addmm/BMM tests
passed in five more fresh processes. Replaying the *unchanged* static pretrained
ViT PTE/PTD and input eight times per dtype gave finite FP16 logits in 8/8
runs (maximum absolute error 0.001953125 versus recorded MUSA eager output)
and finite BF16 logits in 6/8 runs (maximum finite-run error 0.03125); the
other two BF16 runs each contained 1,000 NaN logits. The earlier muBLAS
results above describe the old shim, not this one. BF16 pretrained ViT is
still not a reliable supported configuration, and eight successful FP16
replays do not establish production reliability or throughput. The original
FP32 static PTE also produced finite logits in 3/3 fresh runs with maximum
absolute error 1.1921e-6 after the shim change. The 13 C++ shim tests and all
31 MUSA AOTI Python export tests passed with this implementation.

To narrow down the remaining BF16 NaNs, a new classifier-shaped shim regression
covers `[1, 768] @ [768, 1000]` with a zero first-row stride, transposed
weights, and one-dimensional bias. All three dtypes passed four calls each in
30 fresh processes, and the full shim suite (including the explicitly enabled
padded regression) passed 13/13. A diagnostic BF16 pretrained ViT PTE that
returns class tokens before and after each attention residual passed 12/12
fresh replays with no NaNs; its input matched the original PTE's input
byte-for-byte. Separately, a BF16 PTE containing the *actual pretrained*
final LayerNorm and classifier, fed the eager last-stage class token, matched
MUSA eager exactly in 16/16 fresh runs. Neither experiment preserves the
original full model's allocation and execution schedule, so these results do
not prove that its classifier or attention is sound. Switching the MUSA
SlimTensor free path from `cudaFreeAsync` to synchronous `cudaFree` still
produced 1,000 NaN logits in one of five original BF16 full-model replays;
the change was reverted. The intermittent full-model BF16 failure remains.
These checks use random inputs, not ImageNet accuracy or
throughput measurements.
A separate pretrained FP32 PTE passed independent spatial sizes 64x64, 48x54,
80x80, and 79x71 (maximum logit error 5.96e-6). Another FP32 PTE with jointly
dynamic batch and spatial dimensions passed 2x64x64, 1x48x54, 3x80x80, and
2x79x71 (maximum logit error 5.96e-6); predicted class IDs matched eager in
each case. This does not validate arbitrary combinations within the bounds.
For the jointly dynamic PTE at 2x64x64, pretrained FP16 and BF16 do **not**
meet their default tolerances: 5/2,000 and 7/2,000 logits exceed tolerance,
respectively (maximum differences 0.021484375 and 0.0703125). Both predicted
class IDs remain correct for this input, but neither dtype counts as passed
for combined dynamic shapes. Static 224x224 FP16/BF16 remain separately verified.
For the simple 3D CNN, batches 2/1/3 passed with a single PTE in both FP16
(maximum error 6.11e-5) and BF16 (4.89e-4).
For the simple decoder, batches 2/1/3 passed with a single PTE in both FP16
and BF16 (maximum error 0 for this test input).
With a separate bounded dynamic-batch PTE for each BF16 model, batches 2/1/3
passed for the grouped audio decoder (maximum error 1.95e-3), video decoder
(0), dilated audio decoder (3.91e-3), dilated image decoder (0), and dilated
video decoder (0). This checks dynamic batch only, not dynamic spatial sizes.
The tiny U-Net also passed BF16 batches 2/1/3 with one PTE (maximum error
1.56e-2), covering GroupNorm and the concatenated skip connection.
One dynamic-spatial bilinear U-Net PTE per dtype passed the 32x40, 24x30,
48x48, and 47x39 inputs in FP32 and BF16 (maximum errors 3.58e-7 and
7.81e-3, respectively). These are independent bounded spatial dimensions,
not a validation of arbitrary sizes or a pretrained segmentation model.
One jointly dynamic FP32 bilinear U-Net PTE also passed 2x32x40, 1x24x30,
3x48x48, and 2x47x39 (maximum error 4.17e-7). The last case previously
failed with an unsynchronized GPU-generated input snapshot; replaying clean
input through the same PTE showed no operator mismatch.
The official pretrained DeepLabV3 output varies spatially; all 86,016 FP32
logits met the default 1e-4 tolerances and all 4,096 predicted pixel labels
matched MUSA eager. Pretrained FP16 at 64x64 does **not** meet the default
tolerance: 2,412/86,016 logits differ beyond 1e-2 absolute and relative
tolerances (maximum error 0.03125), although the predicted labels match for
this input. Do not count it as a passed FP16 configuration; BF16 has not been
validated for this model.
The same pretrained FP32 DeepLabV3 also passed batches 2/1/3 using one
bounded-dynamic-batch PTE (maximum logit error 1.18e-4); all predicted pixel
labels matched for each batch. A separate bounded-dynamic-spatial PTE also
passed 64x64, 48x54, 80x80, and 79x71 (maximum logit error 1.63e-4), with
all predicted pixel labels matching eager. One jointly dynamic PTE also
passed inputs 2x64x64, 1x48x54, 3x80x80, and 2x79x71 (maximum logit error
1.63e-4); all pixel labels matched eager. This validates only the bounded
shapes exercised here, not arbitrary resolutions or every combination.
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
lacks a fake implementation for its dynamic SDPA math operator, and a fake
kernel alone still specializes sequence length. This export retains the
unaltered upstream SDPA module and uses
`export_with_musa_sdpa_decomposition` to expand functional SDPA before
TorchMUSA selects its native attention operator. Each runner step is checked
against upstream MUSA eager. On M1000, FP32/FP16/BF16 passed all three calls
with predicted IDs `[364, 76, 464]` and maximum logit differences 5.96e-7,
1.50e-3, and 9.77e-3, respectively. The long prefill and final decode also
passed all three dtypes with predicted IDs `[103, 344]` and maximum logit
differences 4.77e-7, 1.34e-3, and 7.81e-3. The checkpoint still has random
weights; this does not validate a production Llama model.

These are random-input correctness checks, not arbitrary shape-generalization
or throughput measurements. The previously recorded pretrained MobileNetV2
BF16 discrepancy was isolated to BatchNorm rounding. The backend now routes
inference BatchNorm1d/2d/3d through a native muDNN AOTI shim. The official
pretrained MobileNetV2 checkpoint now passes at 64x64 with `--atol 0.02
--rtol 0.04`, maximum difference `7.62939453125e-06`, and two independent
runner replays.

Without `--runner`, the script only exports the PTE/PTD and writes the eager
reference files. The runner command automatically adds the MUSA, Torch, and
Python shared-library directories used by the current environment.

## Latest MUSA AOTI validation (2026-10-18)

The current branch `codex/musa-aoti-poc` includes commits
`4c2151d066` (`stabilize Musa BF16 addmm paths`) and `b4d21a2493`
(`musa: avoid unstable strided batched bmm`). The BF16 small-batch and
padded-row addmm paths use muBLAS GEMM with explicit bias initialization;
larger BF16 addmm keeps the muDNN path. A zero-row-stride `[1, 768]` input,
transposed `[768, 1000]` weight, and padded `[2, 768]` input are covered by
repeated C++ regressions. The 13-test shim suite passed five complete
repetitions, and all 31 MUSA AOTI Python export tests passed.

End-to-end FP32 acceptance also passed with generated PTE/PTD and
`executor_runner`: `simple_cnn` with dynamic batches 2/1/3 (maximum error
3.58e-7), torchvision `resnet18` at 64x64 (maximum error 3.22e-6), and
`mobilenet_v2` at 64x64 (maximum error 1.28e-15); each model passed an
independent runner replay. These are random-weight functional checks, not
ImageNet accuracy or performance measurements.

Additional BF16 functional checks now pass with generated PTE/PTD and the
runner: `mlp` with dynamic batches 2/1/3 (maximum error 0.00390625),
`simple_cnn` with dynamic batches 2/1/3 (maximum error 0.001953125), and
random-weight `resnet18` and `mobilenet_v2` at 64x64 (maximum error 0.0 in
both cases). These checks cover the same SlimTensor/AOTI path but do not
establish pretrained-model accuracy.

The new `simple_audio_cnn` model exercises Conv1d, strided Conv1d, AvgPool1d,
and Linear end to end. Its dynamic batch 2/1/3 acceptance passes in FP32 with
maximum error 1.49e-8 and in BF16 with maximum error 0.00048828125, including
an independent runner replay for each batch size.

Commit `b13bf26773` (`musa: add muDNN batch norm AOTI shim`) adds an inference
BatchNorm fallback for contiguous tensors with or without affine parameters in
FP32/FP16/BF16. It uses a
custom decomposition plus an AOTInductor `ExternKernelAlloc` call, so generated
wrappers call `aoti_torch_musa_batch_norm` directly instead of ProxyExecutor.
The shim uses muDNN's inference API and supports rank-2/3/4/5 inputs, including
FP32 BatchNorm parameters with FP16/BF16 activations. BF16
runner acceptance now passes for `simple_audio_cnn` (maximum error
0.0001220703125), `simple_cnn` (0.001953125), `simple_video_cnn` (0), and
`narrow_resnet18` (0).

The rank-2 BatchNorm1d path and a mixed-dtype BF16 input/FP32 parameter path
also pass generated PTE/PTD plus `executor_runner` with maximum error `0`. The
inference path with `affine=False` is also covered and matches eager exactly.

The same commit closes the pretrained MobileNetV2 BF16 case using
`mobilenet_v2-7ebf99e0.pth`: the 64x64 export and two runner replays pass with
maximum error `7.62939453125e-06` under `--atol 0.02 --rtol 0.04`. The earlier
`0.140625` maximum error was caused by AOTI's pointwise BatchNorm arithmetic,
not by the convolution or BMM shims.

The deeper BF16 ViT failure was traced to the strided-batched BMM path used by
attention. The MUSA shim now uses one `mublasGemmEx` launch per batch for all
supported BMM dtypes in the fallback path, avoiding the unstable
`mublasGemmStridedBatchedEx` schedule. Full-depth pretrained ViT-B/16 BF16 now
passes generated PTE/PTD plus `executor_runner` for dynamic batches 2/1/3,
with three independent runner replays per batch and maximum error 0.03125.
This is a correctness-first fallback; batched-BMM performance optimization
and broader BF16 model coverage remain open.

The random-weight EfficientNet-B0 BF16 path also passes one combined dynamic
PTE/PTD with `executor_runner`: batch 2 at 64x64, batch 1 at 48x54, batch 3 at
80x80, and batch 2 at 79x71 all match MUSA eager with maximum error below
`7e-16`. This exercises the squeeze-and-excitation, depthwise convolution,
BatchNorm, SiLU, and dynamic-shape paths together.

Before the native activation and fused-bias precision fixes described below,
the official pretrained DeepLabV3-MobileNetV3 BF16 path passed the same four
dynamic batch/spatial shapes and two runner replays when invoked with
`--atol 0.5 --rtol 0.04`. Maximum raw-logit errors are `0.25`, `0.125`,
`0.25`, and `0.5`; all 86,016 predicted segmentation labels match MUSA eager
exactly across the four shapes. The wider absolute tolerance was required for
the accumulated BF16 logits at odd spatial sizes, while the exact label check
remains strict.

Random-weight DenseNet121 passes FP32 PTE/PTD export and two fresh runner
executions at 1x3x64x64 (maximum error `2.15e-6`). One combined dynamic PTE
also passes shapes 2x3x64x64, 1x3x32x40, 3x3x80x80, and 2x3x72x64, with
two runner executions per shape and maximum error `4.06e-6` at
`atol=rtol=1e-4`. This exposed a non-contiguous BatchNorm input produced by
dense-block buffer layouts; the BN shim now materializes contiguous inputs
and parameters only when necessary. Four native BN tests pass, including
non-contiguous input and padded-batch/strided-parameter cases.

The same four DenseNet shapes also pass BF16, with two runner executions per
shape and maximum errors `0`, `1.19e-7`, `0`, and `0.00390625` respectively
at `atol=rtol=0.04`. Predicted classes match eager for every execution.
Random-weight ConvNeXt-Tiny passes FP32 and BF16 at 64x64 with dynamic batches
2/1/3 and two runner executions per batch. After the fused-convolution-bias
change, regenerated PTEs pass again with maximum errors `3.58e-6`
and `0.0078125` respectively; predicted classes match on the initial runs.
These random-weight checks establish execution correctness for the tested
inputs, not pretrained classification accuracy or performance.

Reproduce the DenseNet and ConvNeXt runner checks using the environment above:

```bash
$PYTHON examples/musa/aoti_model_acceptance.py \
  --model densenet121 --dtype fp32 --batch-size 2 \
  --dynamic-batch-max 3 --dynamic-spatial-max 80 --runner-repeats 2 \
  --output-dir /tmp/executorch-musa-acceptance/densenet121_fp32 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model convnext_tiny --dtype fp32 --batch-size 2 \
  --dynamic-batch-max 3 --runner-repeats 2 \
  --output-dir /tmp/executorch-musa-acceptance/convnext_tiny_fp32 \
  --runner "$RUNNER"
```

`aoti_backbone_diagnostics.py` measures per-stage DeepLab-MobileNetV3 errors
against MUSA eager. It exports all backbone stages as outputs and records
max/mean/p99 absolute errors and NaN/Inf counts in `result.json`.
Use `--stage 4`, for example, to export just stage 4 with its eager predecessor
as input; this separates local error from accumulation through the backbone.
Run the same command with `--dtype fp32` and `--dtype bf16` into separate
directories. These are diagnostic measurements, not an accuracy acceptance
threshold. Retaining intermediate outputs can change fusion and allocation,
so also run the ordinary full-model acceptance path.

With the official pretrained checkpoint, seed 0 and a 2x3x64x64 input,
the FP32 diagnostic has maximum error at most `4.58e-5` across 17 stages.
The BF16 diagnostic has stage-0 maximum error `0.015625`, stage-6 error
`0.3125`, and final-stage error `0.171875`; no NaN/Inf values appear.
Isolated stage 4 has maximum error `0.0625` (versus `0.1201171875` in the
cumulative run). This confirms a local BF16 discrepancy as well as propagated
error; it does not yet identify a particular convolution algorithm as the
cause. These stage outputs are backbone features, not segmentation logits.

The subsequent operator-level probe (`--stem-operators`) identifies the
stem's BF16 discrepancy in Hardswish: Conv2d and BatchNorm2d match exactly,
while decomposed Hardswish differs by `0.015625`, including when isolated
with `--stem-operator 2`. The MUSA backend now calls a native muDNN Hardswish
shim instead. With that shim all three stem outputs and backbone stages 0–3
match exactly; the remaining backbone discrepancy starts at stage 4 and the
last-stage maximum error is `0.140625`. This only resolves the Hardswish
component of the BF16 difference; full-model acceptance remains separate.

Hardswish's C++ tests cover FP32/FP16/BF16, contiguous and strided offset views,
scalar and empty inputs, and preservation of input storage. All three tests
pass. The fallback registration and dynamic-batch AOTI compile tests pass for
all three dtypes. The muDNN C unary API binds data addresses into its tensor
descriptors, so the shim uses separate input and output descriptors.

`--block-operators 4` emits the intermediate operators in the main branch of
inverted-residual stage 4, fed by its eager predecessor. For residual stages,
this mode intentionally excludes the final residual addition. Diagnostic
outputs are cloned before the next operator runs so in-place activations do
not overwrite an earlier observation.

Further SE isolation finds another source of BF16 error: the eager convolution
fuses bias into the accumulator, whereas Inductor's generic GPU lowering
removes bias from the convolution and adds it after rounding the output.
The MUSA compile context now keeps bias in the native forward-convolution
call. The shim uses `mudnnConvolutionBiasActivationForward` with identity
activation and a one-dimensional bias descriptor. Transposed convolution keeps
its separate bias addition, matching the eager implementation.

The convolution ABI also now matches AOTI's optional tensor argument: bias is
passed as a pointer to a tensor handle, and `transposed` is an `int32_t`.
Previously the signature mismatch was hidden because lowered calls always
passed null bias. Both compiler lowering and runtime changes are needed;
existing PTEs that already separated the bias must be regenerated.

With these changes, all six observed operations in pretrained stage-4 SE
(pool, fc1, ReLU, fc2, Hardsigmoid, multiply) match BF16 eager exactly on the
diagnostic input. Eleven native convolution tests pass, including a two-output
channel BF16 cancellation test: `(1 + 0.00390625) - 1` must return `0.00390625`
with fused bias, rather than `0` after an intermediate BF16 rounding.
Six PTE/PTD runner cases also pass exactly: Conv1d/2d/3d in FP16 and BF16,
each exported with a dynamic batch bound and executed at batch 2. Run them with:

```bash
EXECUTORCH_MUSA_RUNNER="$RUNNER" $PYTHON -m unittest \
  backends.musa.tests.test_musa_runner -v
```

Without `EXECUTORCH_MUSA_RUNNER`, these runner tests are explicitly skipped.

Before the native Hardsigmoid change, the regenerated pretrained DeepLab BF16
full model still fails the default
`atol=rtol=0.04` check at 2x3x64x64: 1,005 of 172,032 values fail
`torch.testing.assert_close`, down from 16,419 with native Hardswish alone.
The maximum absolute error over all logits is `0.125`; all 8,192 pixel labels
match eager. This is not a passing default-tolerance acceptance, and the
remaining dynamic shapes have not passed that stricter check.

The diagnostic script accepts `--input /path/to/input-0.bin` to reuse an
acceptance input with the specified dtype, batch, height, and width. Trailing
zero padding from the dynamic input allocator is allowed. With
`--include-classifier`, it also emits the ASPP, classifier, and final bilinear
upsampling outputs. `--classifier-operators` instead feeds the classifier with
the eager backbone output to isolate head-local error.
On the failing acceptance input, backbone stages 0–4 match exactly, the first
discrepancy is at stage 5 (`0.0625`), and the final backbone error is `0.078125`.
The full diagnostic's final logits reproduce the same `0.125` maximum and
`0.0289871` mean absolute error as the ordinary full-model runner.

Further stage-5 SE isolation finds exact pooling, fc1, ReLU, and fc2 outputs,
followed by a `0.00390625` Hardsigmoid difference and a `0.0078125` multiply
difference. TorchMUSA's native Hardsigmoid uses muDNN with alpha `0.166667`
and beta `0.5`, rather than decomposed `clamp(x + 3, 0, 6) / 6`. At BF16
rounding boundaries, these are not interchangeable: for example, input
`0.85546875` produces eager `0.64453125` versus decomposed `0.640625`.
The backend now preserves native Hardsigmoid through a dedicated custom op
and shared unary SlimTensor shim, using the eager implementation's parameters.
All four native activation tests pass. The runner regression covers those
boundary inputs in FP32/FP16/BF16 and batches 1/2/3, with exact eager equality
for all nine runs. This operator result does not alone establish full-model
accuracy.

With native Hardswish, fused forward-convolution bias, and native Hardsigmoid
together, the regenerated pretrained DeepLab BF16 PTE/PTD passes the default
`atol=rtol=0.04` without overrides. One dynamic PTE passes two fresh runner
executions for each of these shapes:

| Input shape | Maximum absolute logit error (both runs) |
| --- | ---: |
| 2x3x64x64 | 0 |
| 1x3x48x54 | 0.015625 |
| 3x3x80x80 | 0 |
| 2x3x79x71 | 0.03125 |

An additional readback of all eight saved outputs verifies exact segmentation
labels for all 82,404 pixels, including the repeated executions.
After these precision fixes, the complete 38-test MUSA AOTI Python suite also
passes, including the existing convolution, normalization, KV-cache, SDPA,
Transformer, EfficientNet, DeepLab, and ViT compile regressions.

These checks use the official pretrained weights with generated inputs and
MUSA eager as the reference; they are not dataset-level segmentation accuracy
or performance measurements. Reproduce the stricter acceptance with:

```bash
$PYTHON examples/musa/aoti_model_acceptance.py \
  --model deeplabv3_mobilenet_v3_large --dtype bf16 \
  --weights /path/to/deeplabv3_mobilenet_v3_large-fc3c493d.pth \
  --batch-size 2 --dynamic-batch-max 3 --dynamic-spatial-max 80 \
  --runner-repeats 2 --runner "$RUNNER" \
  --output-dir /tmp/executorch-musa-acceptance/deeplab_bf16_native
```

```bash
$PYTHON examples/musa/aoti_backbone_diagnostics.py \
  --dtype bf16 --batch-size 2 --height 64 --width 64 \
  --weights /path/to/deeplabv3_mobilenet_v3_large-fc3c493d.pth \
  --runner "$RUNNER" \
  --output-dir /tmp/executorch-musa-acceptance/deeplab_backbone_bf16
```

The BatchNorm decomposition now also covers direct `native_batch_norm`,
`_native_batch_norm_legit`, `_native_batch_norm_legit_no_training`, and
`_native_batch_norm_legit_functional` inference graphs, plus
`_batch_norm_no_update`. Their empty saved-stat/reserve-space outputs and
functional running-stat outputs are preserved; the complete 38-test MUSA AOTI
Python suite passes on the current MUSA environment. A `BatchNorm2d` module with
`track_running_stats=False` also passes a complete PTE/PTD runner check with
maximum FP32 error `2.38e-7`.

A random-weight ViT-B/16 with one encoder block also passes FP32 export and
two fresh `executor_runner` executions at 1x3x224x224 with maximum error `0`.
This provides a runner-level smoke test for the attention and LayerNorm path;
full-depth ViT coverage is tracked separately above.
