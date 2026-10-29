#pragma once

#ifdef TORCH_INDUCTOR_CPP_WRAPPER
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#define MUSA_VISION_SOFTMAX_EXPORT AOTI_TORCH_EXPORT
using MusaVisionSoftmaxError = AOTITorchError;
using MusaVisionSoftmaxTensor = AtenTensorHandle;
#else
#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/runtime/core/error.h>

namespace executorch::backends::cuda {
#define MUSA_VISION_SOFTMAX_EXPORT AOTI_SHIM_EXPORT
using MusaVisionSoftmaxError = executorch::runtime::Error;
using MusaVisionSoftmaxTensor = executorch::backends::aoti::slim::SlimTensor*;
#endif

extern "C" MUSA_VISION_SOFTMAX_EXPORT MusaVisionSoftmaxError
aoti_torch_musa_vision_softmax_out(
    MusaVisionSoftmaxTensor output,
    MusaVisionSoftmaxTensor input);

#ifndef TORCH_INDUCTOR_CPP_WRAPPER
}
#endif

#undef MUSA_VISION_SOFTMAX_EXPORT
