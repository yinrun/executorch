#pragma once

#include <cstdint>

#ifdef TORCH_INDUCTOR_CPP_WRAPPER
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#define MUSA_GGUF_EXPORT AOTI_TORCH_EXPORT
using MusaGgufError = AOTITorchError;
using MusaGgufTensor = AtenTensorHandle;
#else
#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/runtime/core/error.h>
namespace executorch::backends::cuda {
#define MUSA_GGUF_EXPORT AOTI_SHIM_EXPORT
using MusaGgufError = executorch::runtime::Error;
using MusaGgufTensor = executorch::backends::aoti::slim::SlimTensor*;
#endif

extern "C" {
MUSA_GGUF_EXPORT MusaGgufError
aoti_torch_musa_gguf_linear_out(MusaGgufTensor output, MusaGgufTensor input,
                                MusaGgufTensor packed, int64_t quant_type);
MUSA_GGUF_EXPORT MusaGgufError aoti_torch_musa_gguf_embedding_out(
    MusaGgufTensor output, MusaGgufTensor indices, MusaGgufTensor packed,
    int64_t quant_type);
}

#ifndef TORCH_INDUCTOR_CPP_WRAPPER
}
#endif
#undef MUSA_GGUF_EXPORT
