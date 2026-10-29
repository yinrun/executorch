#pragma once

#ifdef TORCH_INDUCTOR_CPP_WRAPPER
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#define MUSA_VISION_GELU_EXPORT AOTI_TORCH_EXPORT
using MusaVisionGeluError = AOTITorchError;
using MusaVisionGeluTensor = AtenTensorHandle;
#else
#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/runtime/core/error.h>

namespace executorch::backends::cuda {
#define MUSA_VISION_GELU_EXPORT AOTI_SHIM_EXPORT
using MusaVisionGeluError = executorch::runtime::Error;
using MusaVisionGeluTensor = executorch::backends::aoti::slim::SlimTensor*;
#endif

extern "C" MUSA_VISION_GELU_EXPORT MusaVisionGeluError
aoti_torch_musa_vision_gelu_out(
    MusaVisionGeluTensor output,
    MusaVisionGeluTensor input,
    int64_t mode);

#ifndef TORCH_INDUCTOR_CPP_WRAPPER
}
#endif

#undef MUSA_VISION_GELU_EXPORT
