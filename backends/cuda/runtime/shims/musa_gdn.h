#pragma once

#ifdef TORCH_INDUCTOR_CPP_WRAPPER
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#define MUSA_GDN_EXPORT AOTI_TORCH_EXPORT
using MusaGdnError = AOTITorchError;
using MusaGdnTensor = AtenTensorHandle;
#else
#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/runtime/core/error.h>

namespace executorch::backends::cuda {
#define MUSA_GDN_EXPORT AOTI_SHIM_EXPORT
using MusaGdnError = executorch::runtime::Error;
using MusaGdnTensor = executorch::backends::aoti::slim::SlimTensor*;
#endif

extern "C" MUSA_GDN_EXPORT MusaGdnError aoti_torch_musa_gdn_recurrence_out(
    MusaGdnTensor output,
    MusaGdnTensor query,
    MusaGdnTensor key,
    MusaGdnTensor value,
    MusaGdnTensor decay,
    MusaGdnTensor beta,
    MusaGdnTensor state);

#ifndef TORCH_INDUCTOR_CPP_WRAPPER
}
#endif

#undef MUSA_GDN_EXPORT
