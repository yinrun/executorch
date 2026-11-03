#pragma once

#ifdef TORCH_INDUCTOR_CPP_WRAPPER
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#define MUSA_CAUSAL_ATTENTION_EXPORT AOTI_TORCH_EXPORT
using MusaCausalAttentionError = AOTITorchError;
using MusaCausalAttentionTensor = AtenTensorHandle;
#else
#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/runtime/core/error.h>

namespace executorch::backends::cuda {
#define MUSA_CAUSAL_ATTENTION_EXPORT AOTI_SHIM_EXPORT
using MusaCausalAttentionError = executorch::runtime::Error;
using MusaCausalAttentionTensor = executorch::backends::aoti::slim::SlimTensor*;
#endif

extern "C" MUSA_CAUSAL_ATTENTION_EXPORT MusaCausalAttentionError
aoti_torch_musa_causal_attention_out(
    MusaCausalAttentionTensor output,
    MusaCausalAttentionTensor query,
    MusaCausalAttentionTensor key,
    MusaCausalAttentionTensor value,
    MusaCausalAttentionTensor position,
    MusaCausalAttentionTensor mask);

extern "C" MUSA_CAUSAL_ATTENTION_EXPORT MusaCausalAttentionError
aoti_torch_musa_causal_prefix_out(
    MusaCausalAttentionTensor output,
    MusaCausalAttentionTensor position,
    MusaCausalAttentionTensor mask);

extern "C" MUSA_CAUSAL_ATTENTION_EXPORT MusaCausalAttentionError
aoti_torch_musa_prefix_attention_out(
    MusaCausalAttentionTensor output,
    MusaCausalAttentionTensor query,
    MusaCausalAttentionTensor key,
    MusaCausalAttentionTensor value,
    MusaCausalAttentionTensor control);

#ifndef TORCH_INDUCTOR_CPP_WRAPPER
}
#endif

#undef MUSA_CAUSAL_ATTENTION_EXPORT
