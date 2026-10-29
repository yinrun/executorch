#pragma once

#ifdef TORCH_INDUCTOR_CPP_WRAPPER
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#define MUSA_VISION_NORM_EXPORT AOTI_TORCH_EXPORT
using MusaVisionNormError = AOTITorchError;
using MusaVisionNormTensor = AtenTensorHandle;
#else
#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/runtime/core/error.h>

namespace executorch::backends::cuda {
#define MUSA_VISION_NORM_EXPORT AOTI_SHIM_EXPORT
using MusaVisionNormError = executorch::runtime::Error;
using MusaVisionNormTensor = executorch::backends::aoti::slim::SlimTensor*;
#endif

extern "C" MUSA_VISION_NORM_EXPORT MusaVisionNormError
aoti_torch_musa_vision_layer_norm_out(
    MusaVisionNormTensor output,
    MusaVisionNormTensor input,
    MusaVisionNormTensor weight,
    MusaVisionNormTensor bias,
    double epsilon);

#ifndef TORCH_INDUCTOR_CPP_WRAPPER
}
#endif

#undef MUSA_VISION_NORM_EXPORT
