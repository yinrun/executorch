#pragma once

#ifdef TORCH_INDUCTOR_CPP_WRAPPER

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#define MUSA_CONVOLUTION_EXPORT AOTI_TORCH_EXPORT
using MusaConvolutionError = AOTITorchError;
using MusaConvolutionTensorHandle = AtenTensorHandle;

#else

#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/runtime/core/error.h>

namespace executorch::backends::cuda {

#define MUSA_CONVOLUTION_EXPORT AOTI_SHIM_EXPORT
using MusaConvolutionError = executorch::runtime::Error;
using MusaConvolutionTensorHandle =
    executorch::backends::aoti::slim::SlimTensor*;

#endif

extern "C" {

MUSA_CONVOLUTION_EXPORT MusaConvolutionError aoti_torch_musa_convolution_out(
    MusaConvolutionTensorHandle output,
    MusaConvolutionTensorHandle input,
    MusaConvolutionTensorHandle weight,
    MusaConvolutionTensorHandle const* bias,
    const int64_t* stride,
    int64_t stride_len,
    const int64_t* padding,
    int64_t padding_len,
    const int64_t* dilation,
    int64_t dilation_len,
    int32_t transposed,
    const int64_t* output_padding,
    int64_t output_padding_len,
    int64_t groups);

#ifndef TORCH_INDUCTOR_CPP_WRAPPER
MUSA_CONVOLUTION_EXPORT MusaConvolutionError aoti_torch_musa_convolution(
    MusaConvolutionTensorHandle input,
    MusaConvolutionTensorHandle weight,
    MusaConvolutionTensorHandle const* bias,
    const int64_t* stride,
    int64_t stride_len,
    const int64_t* padding,
    int64_t padding_len,
    const int64_t* dilation,
    int64_t dilation_len,
    int32_t transposed,
    const int64_t* output_padding,
    int64_t output_padding_len,
    int64_t groups,
    MusaConvolutionTensorHandle* ret_output);
#endif
}

#ifndef TORCH_INDUCTOR_CPP_WRAPPER
}
#endif

#undef MUSA_CONVOLUTION_EXPORT
