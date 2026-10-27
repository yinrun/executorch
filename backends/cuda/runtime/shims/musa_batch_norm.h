#pragma once

#ifdef TORCH_INDUCTOR_CPP_WRAPPER

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#define MUSA_BATCH_NORM_EXPORT AOTI_TORCH_EXPORT
using MusaBatchNormError = AOTITorchError;
using MusaBatchNormTensorHandle = AtenTensorHandle;

#else

#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/runtime/core/error.h>

namespace executorch::backends::cuda {

using executorch::backends::aoti::slim::SlimTensor;
using executorch::runtime::Error;
using AOTITorchError = Error;

#define MUSA_BATCH_NORM_EXPORT AOTI_SHIM_EXPORT
using MusaBatchNormError = AOTITorchError;
using MusaBatchNormTensorHandle = SlimTensor*;

#endif

extern "C" {

MUSA_BATCH_NORM_EXPORT MusaBatchNormError aoti_torch_musa_batch_norm_out(
    MusaBatchNormTensorHandle output,
    MusaBatchNormTensorHandle input,
    MusaBatchNormTensorHandle weight,
    MusaBatchNormTensorHandle bias,
    MusaBatchNormTensorHandle running_mean,
    MusaBatchNormTensorHandle running_var,
    double eps);

MUSA_BATCH_NORM_EXPORT MusaBatchNormError aoti_torch_musa_batch_norm(
    MusaBatchNormTensorHandle input,
    MusaBatchNormTensorHandle weight,
    MusaBatchNormTensorHandle bias,
    MusaBatchNormTensorHandle running_mean,
    MusaBatchNormTensorHandle running_var,
    double eps,
    MusaBatchNormTensorHandle* ret_output);

}

#ifndef TORCH_INDUCTOR_CPP_WRAPPER
}
#endif

#undef MUSA_BATCH_NORM_EXPORT
