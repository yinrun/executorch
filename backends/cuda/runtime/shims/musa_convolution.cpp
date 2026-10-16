#include <mudnn.h>

#include <cstdint>
#include <limits>
#include <memory>

#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {

using executorch::backends::aoti::slim::SlimTensor;
using executorch::backends::aoti::slim::c10::DeviceType;
using executorch::backends::aoti::slim::c10::ScalarType;
using executorch::runtime::Error;
using AOTITorchError = Error;

namespace {

AOTITorchError check_mudnn(mudnnStatus_t status, const char* operation) {
  if (status == MUDNN_STATUS_SUCCESS) {
    return Error::Ok;
  }
  ET_LOG(
      Error,
      "MUSA convolution: %s failed with mudnn status %d",
      operation,
      static_cast<int>(status));
  return Error::Internal;
}

bool fits_mudnn_dimension(int64_t value) {
  return value > 0 &&
      value <= static_cast<int64_t>(std::numeric_limits<int>::max());
}

mudnnDataType_t mudnn_data_type(ScalarType dtype) {
  if (dtype == ScalarType::Float) {
    return MUDNN_DATA_FLOAT;
  }
  if (dtype == ScalarType::Half) {
    return MUDNN_DATA_HALF;
  }
  return MUDNN_DATA_BFLOAT16;
}

struct MudnnContext {
  mudnnHandle_t handle = nullptr;
  mudnnTensorDescriptor_t input = nullptr;
  mudnnTensorDescriptor_t bias = nullptr;
  mudnnTensorDescriptor_t output = nullptr;
  mudnnFilterDescriptor_t weight = nullptr;
  mudnnConvolutionDescriptor_t convolution = nullptr;

  ~MudnnContext() {
    if (convolution != nullptr) {
      mudnnDestroyConvolutionDescriptor(convolution);
    }
    if (weight != nullptr) {
      mudnnDestroyFilterDescriptor(weight);
    }
    if (output != nullptr) {
      mudnnDestroyTensorDescriptor(output);
    }
    if (bias != nullptr) {
      mudnnDestroyTensorDescriptor(bias);
    }
    if (input != nullptr) {
      mudnnDestroyTensorDescriptor(input);
    }
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
  }
};

struct DeviceWorkspace {
  void* data = nullptr;

  ~DeviceWorkspace() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }
};

} // namespace

extern "C" {

AOTI_SHIM_EXPORT AOTITorchError aoti_torch_musa_convolution(
    SlimTensor* input,
    SlimTensor* weight,
    SlimTensor* bias,
    const int64_t* stride,
    int64_t stride_len,
    const int64_t* padding,
    int64_t padding_len,
    const int64_t* dilation,
    int64_t dilation_len,
    bool transposed,
    const int64_t* output_padding,
    int64_t output_padding_len,
    int64_t groups,
    SlimTensor** ret_output) {
  ET_CHECK_OR_RETURN_ERROR(
      input != nullptr && weight != nullptr && ret_output != nullptr,
      InvalidArgument,
      "MUSA convolution received a null tensor handle");
  *ret_output = nullptr;

  const auto dtype = input->dtype();
  ET_CHECK_OR_RETURN_ERROR(
      (dtype == ScalarType::Float || dtype == ScalarType::Half ||
       dtype == ScalarType::BFloat16) && weight->dtype() == dtype &&
          (bias == nullptr || bias->dtype() == dtype),
      NotSupported,
      "MUSA convolution supports float32, float16, and bfloat16 tensors only");
  ET_CHECK_OR_RETURN_ERROR(
      input->is_cuda() && weight->is_cuda() &&
          (bias == nullptr || bias->is_cuda()),
      InvalidArgument,
      "MUSA convolution expects device tensors");
  ET_CHECK_OR_RETURN_ERROR(
      input->device_index() == weight->device_index() &&
          (bias == nullptr || input->device_index() == bias->device_index()),
      InvalidArgument,
      "MUSA convolution tensors must use the same device");
  ET_CHECK_OR_RETURN_ERROR(
      input->dim() == 4 && weight->dim() == 4,
      NotSupported,
      "MUSA convolution currently supports 2D convolution only");
  ET_CHECK_OR_RETURN_ERROR(
      !transposed && stride != nullptr && padding != nullptr &&
          dilation != nullptr && output_padding != nullptr && stride_len == 2 &&
          padding_len == 2 && dilation_len == 2 && output_padding_len == 2,
      NotSupported,
      "MUSA convolution supports non-transposed 2D convolution only");
  ET_CHECK_OR_RETURN_ERROR(
      groups > 0 && stride[0] > 0 && stride[1] > 0 && dilation[0] > 0 &&
          dilation[1] > 0 && padding[0] >= 0 && padding[1] >= 0 &&
          output_padding[0] == 0 && output_padding[1] == 0,
      InvalidArgument,
      "MUSA convolution received invalid convolution parameters");
  ET_CHECK_OR_RETURN_ERROR(
      input->is_contiguous() && weight->is_contiguous() &&
          (bias == nullptr || bias->is_contiguous()),
      NotSupported,
      "MUSA convolution currently supports contiguous tensors only");

  const int64_t batch = input->size(0);
  const int64_t input_channels = input->size(1);
  const int64_t input_height = input->size(2);
  const int64_t input_width = input->size(3);
  const int64_t output_channels = weight->size(0);
  const int64_t kernel_height = weight->size(2);
  const int64_t kernel_width = weight->size(3);
  ET_CHECK_OR_RETURN_ERROR(
      input_channels % groups == 0 && weight->size(1) == input_channels / groups &&
          (bias == nullptr || bias->dim() == 1) &&
          (bias == nullptr || bias->size(0) == output_channels),
      InvalidArgument,
      "MUSA convolution tensor dimensions are incompatible");

  const int64_t output_height =
      (input_height + 2 * padding[0] -
       dilation[0] * (kernel_height - 1) - 1) /
          stride[0] +
      1;
  const int64_t output_width =
      (input_width + 2 * padding[1] -
       dilation[1] * (kernel_width - 1) - 1) /
          stride[1] +
      1;
  ET_CHECK_OR_RETURN_ERROR(
      fits_mudnn_dimension(batch) && fits_mudnn_dimension(input_channels) &&
          fits_mudnn_dimension(input_height) && fits_mudnn_dimension(input_width) &&
          fits_mudnn_dimension(output_channels) &&
          fits_mudnn_dimension(kernel_height) && fits_mudnn_dimension(kernel_width) &&
          fits_mudnn_dimension(output_height) && fits_mudnn_dimension(output_width),
      NotSupported,
      "MUSA convolution dimensions exceed muDNN limits");

  const int64_t output_sizes[] = {
      batch, output_channels, output_height, output_width};
  const int64_t output_strides[] = {
      output_channels * output_height * output_width,
      output_height * output_width,
      output_width,
      1};
  SlimTensor* raw_output = nullptr;
  auto error = aoti_torch_empty_strided(
      4,
      output_sizes,
      output_strides,
      static_cast<int32_t>(dtype),
      static_cast<int32_t>(DeviceType::CUDA),
      input->device_index(),
      &raw_output);
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  std::unique_ptr<SlimTensor> output(raw_output);

  MudnnContext context;
  error = check_mudnn(mudnnCreate(&context.handle), "mudnnCreate");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  auto stream_result = getCurrentCUDAStream(input->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(stream_result.error());
  error = check_mudnn(
      mudnnSetStream(context.handle, stream_result.get()), "mudnnSetStream");
  ET_CHECK_OK_OR_RETURN_ERROR(error);

  error = check_mudnn(
      mudnnCreateTensorDescriptor(&context.input), "mudnnCreateTensorDescriptor");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  error = check_mudnn(
      mudnnCreateTensorDescriptor(&context.output), "mudnnCreateTensorDescriptor");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  error = check_mudnn(
      mudnnCreateFilterDescriptor(&context.weight), "mudnnCreateFilterDescriptor");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  error = check_mudnn(
      mudnnCreateConvolutionDescriptor(&context.convolution),
      "mudnnCreateConvolutionDescriptor");
  ET_CHECK_OK_OR_RETURN_ERROR(error);

  const auto data_type = mudnn_data_type(dtype);
  error = check_mudnn(
      mudnnSetTensor4dDescriptor(
          context.input,
          MUDNN_TENSOR_NCHW,
          data_type,
          static_cast<int>(batch),
          static_cast<int>(input_channels),
          static_cast<int>(input_height),
          static_cast<int>(input_width)),
      "mudnnSetTensor4dDescriptor(input)");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  error = check_mudnn(
      mudnnSetTensor4dDescriptor(
          context.output,
          MUDNN_TENSOR_NCHW,
          data_type,
          static_cast<int>(batch),
          static_cast<int>(output_channels),
          static_cast<int>(output_height),
          static_cast<int>(output_width)),
      "mudnnSetTensor4dDescriptor(output)");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  error = check_mudnn(
      mudnnSetFilter4dDescriptor(
          context.weight,
          data_type,
          MUDNN_TENSOR_NCHW,
          static_cast<int>(output_channels),
          static_cast<int>(weight->size(1)),
          static_cast<int>(kernel_height),
          static_cast<int>(kernel_width)),
      "mudnnSetFilter4dDescriptor");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  error = check_mudnn(
      mudnnSetConvolution2dDescriptor(
          context.convolution,
          static_cast<int>(padding[0]),
          static_cast<int>(padding[1]),
          static_cast<int>(stride[0]),
          static_cast<int>(stride[1]),
          static_cast<int>(dilation[0]),
          static_cast<int>(dilation[1]),
          MUDNN_CROSS_CORRELATION,
          MUDNN_DATA_FLOAT),
      "mudnnSetConvolution2dDescriptor");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  error = check_mudnn(
      mudnnSetConvolutionGroupCount(
          context.convolution, static_cast<int>(groups)),
      "mudnnSetConvolutionGroupCount");
  ET_CHECK_OK_OR_RETURN_ERROR(error);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  size_t workspace_size = 0;
  error = check_mudnn(
      mudnnGetConvolutionForwardWorkspaceSize(
          context.handle,
          context.input,
          context.weight,
          context.convolution,
          context.output,
          MUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
          &workspace_size),
      "mudnnGetConvolutionForwardWorkspaceSize");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  DeviceWorkspace workspace;
  if (workspace_size != 0) {
    if (cudaMalloc(&workspace.data, workspace_size) != cudaSuccess) {
      ET_LOG(Error, "MUSA convolution: cudaMalloc workspace failed");
      return Error::Internal;
    }
  }
  error = check_mudnn(
      mudnnConvolutionForward(
          context.handle,
          &alpha,
          context.input,
          input->data_ptr(),
          context.weight,
          weight->data_ptr(),
          context.convolution,
          MUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
          workspace.data,
          workspace_size,
          &beta,
          context.output,
          output->data_ptr()),
      "mudnnConvolutionForward");
  ET_CHECK_OK_OR_RETURN_ERROR(error);

  if (bias != nullptr) {
    error = check_mudnn(
        mudnnCreateTensorDescriptor(&context.bias), "mudnnCreateTensorDescriptor");
    ET_CHECK_OK_OR_RETURN_ERROR(error);
    error = check_mudnn(
        mudnnSetTensor4dDescriptor(
            context.bias,
            MUDNN_TENSOR_NCHW,
            data_type,
            1,
            static_cast<int>(output_channels),
            1,
            1),
        "mudnnSetTensor4dDescriptor(bias)");
    ET_CHECK_OK_OR_RETURN_ERROR(error);
    error = check_mudnn(
        mudnnAddTensor(
            context.handle,
            &alpha,
            context.bias,
            bias->data_ptr(),
            &alpha,
            context.output,
            output->data_ptr()),
        "mudnnAddTensor");
    ET_CHECK_OK_OR_RETURN_ERROR(error);
  }

  *ret_output = output.release();
  return Error::Ok;
}

}
}
