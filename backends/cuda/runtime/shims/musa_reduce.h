#pragma once

#include <cstdint>

#ifdef TORCH_INDUCTOR_CPP_WRAPPER
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#define MUSA_REDUCE_EXPORT AOTI_TORCH_EXPORT
using MusaReduceError = AOTITorchError;
using MusaReduceTensor = AtenTensorHandle;
#else
#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/runtime/core/error.h>

namespace executorch::backends::cuda {
#define MUSA_REDUCE_EXPORT AOTI_SHIM_EXPORT
using MusaReduceError = executorch::runtime::Error;
using MusaReduceTensor = executorch::backends::aoti::slim::SlimTensor*;
#endif

extern "C" {
MUSA_REDUCE_EXPORT MusaReduceError aoti_torch_musa_reduce_out(
    MusaReduceTensor output,
    MusaReduceTensor input,
    int64_t axis,
    bool keepdim,
    bool mean);
MUSA_REDUCE_EXPORT MusaReduceError aoti_torch_musa_bf16_sum_out(
    MusaReduceTensor output,
    MusaReduceTensor input,
    int64_t axis,
    bool keepdim,
    bool mean);
}

#ifndef TORCH_INDUCTOR_CPP_WRAPPER
}
#endif

#undef MUSA_REDUCE_EXPORT
