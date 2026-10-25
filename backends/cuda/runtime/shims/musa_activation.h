#pragma once

#ifdef TORCH_INDUCTOR_CPP_WRAPPER
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#define MUSA_ACTIVATION_EXPORT AOTI_TORCH_EXPORT
using MusaActivationError = AOTITorchError;
using MusaActivationTensorHandle = AtenTensorHandle;
#else
#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/runtime/core/error.h>

namespace executorch::backends::cuda {
#define MUSA_ACTIVATION_EXPORT AOTI_SHIM_EXPORT
using MusaActivationError = executorch::runtime::Error;
using MusaActivationTensorHandle =
    executorch::backends::aoti::slim::SlimTensor*;
#endif

extern "C" {
MUSA_ACTIVATION_EXPORT MusaActivationError aoti_torch_musa_hardswish(
    MusaActivationTensorHandle input,
    MusaActivationTensorHandle* ret_output);
MUSA_ACTIVATION_EXPORT MusaActivationError aoti_torch_musa_hardsigmoid(
    MusaActivationTensorHandle input,
    MusaActivationTensorHandle* ret_output);
}

#ifndef TORCH_INDUCTOR_CPP_WRAPPER
}
#endif

#undef MUSA_ACTIVATION_EXPORT
