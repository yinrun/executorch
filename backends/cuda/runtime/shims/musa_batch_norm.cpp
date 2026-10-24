#include <mudnn.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/musa_batch_norm.h>
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
      "MUSA batch norm: %s failed with mudnn status %d",
      operation,
      static_cast<int>(status));
  return Error::Internal;
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
  mudnnTensorDescriptor_t output = nullptr;
  mudnnTensorDescriptor_t scale = nullptr;
  mudnnTensorDescriptor_t bias = nullptr;
  mudnnTensorDescriptor_t running_mean = nullptr;
  mudnnTensorDescriptor_t running_var = nullptr;
  mudnnBatchNormDescriptor_t batch_norm = nullptr;

  ~MudnnContext() {
    if (batch_norm != nullptr) {
      mudnnDestroyBatchNormDescriptor(batch_norm);
    }
    if (running_var != nullptr) {
      mudnnDestroyTensorDescriptor(running_var);
    }
    if (running_mean != nullptr) {
      mudnnDestroyTensorDescriptor(running_mean);
    }
    if (bias != nullptr) {
      mudnnDestroyTensorDescriptor(bias);
    }
    if (scale != nullptr) {
      mudnnDestroyTensorDescriptor(scale);
    }
    if (output != nullptr) {
      mudnnDestroyTensorDescriptor(output);
    }
    if (input != nullptr) {
      mudnnDestroyTensorDescriptor(input);
    }
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
  }
};

}

extern "C" {

AOTI_SHIM_EXPORT AOTITorchError aoti_torch_musa_batch_norm(
    SlimTensor* input,
    SlimTensor* weight,
    SlimTensor* bias,
    SlimTensor* running_mean,
    SlimTensor* running_var,
    double eps,
    SlimTensor** ret_output) {
  ET_CHECK_OR_RETURN_ERROR(
      input != nullptr && weight != nullptr && bias != nullptr &&
          running_mean != nullptr && running_var != nullptr &&
          ret_output != nullptr,
      InvalidArgument,
      "MUSA batch norm received a null tensor handle");
  *ret_output = nullptr;

  const auto dtype = input->dtype();
  const auto parameter_dtype = weight->dtype();
  ET_CHECK_OR_RETURN_ERROR(
      (dtype == ScalarType::Float || dtype == ScalarType::Half ||
       dtype == ScalarType::BFloat16) &&
          (parameter_dtype == dtype || parameter_dtype == ScalarType::Float) &&
          bias->dtype() == parameter_dtype &&
          running_mean->dtype() == parameter_dtype &&
          running_var->dtype() == parameter_dtype,
      NotSupported,
      "MUSA batch norm supports float32, float16, and bfloat16 tensors only");
  ET_CHECK_OR_RETURN_ERROR(
      input->is_cuda() && weight->is_cuda() && bias->is_cuda() &&
          running_mean->is_cuda() && running_var->is_cuda(),
      InvalidArgument,
      "MUSA batch norm expects device tensors");
  ET_CHECK_OR_RETURN_ERROR(
      input->device_index() == weight->device_index() &&
          input->device_index() == bias->device_index() &&
          input->device_index() == running_mean->device_index() &&
          input->device_index() == running_var->device_index(),
      InvalidArgument,
      "MUSA batch norm tensors must use the same device");
  ET_CHECK_OR_RETURN_ERROR(
      input->dim() >= 2 && input->dim() <= 5,
      NotSupported,
      "MUSA batch norm supports BatchNorm1d, BatchNorm2d, and BatchNorm3d inputs only");
  ET_CHECK_OR_RETURN_ERROR(
      weight->dim() == 1 && bias->dim() == 1 && running_mean->dim() == 1 &&
          running_var->dim() == 1 && weight->size(0) == input->size(1) &&
          bias->size(0) == input->size(1) &&
          running_mean->size(0) == input->size(1) &&
          running_var->size(0) == input->size(1),
      InvalidArgument,
      "MUSA batch norm expects one parameter per channel");
  ET_CHECK_OR_RETURN_ERROR(
      eps >= 0.0,
      InvalidArgument,
      "MUSA batch norm received a negative epsilon");

  std::unique_ptr<SlimTensor> contiguous_input;
  std::unique_ptr<SlimTensor> contiguous_weight;
  std::unique_ptr<SlimTensor> contiguous_bias;
  std::unique_ptr<SlimTensor> contiguous_running_mean;
  std::unique_ptr<SlimTensor> contiguous_running_var;
  SlimTensor* input_for_mudnn = input;
  SlimTensor* weight_for_mudnn = weight;
  SlimTensor* bias_for_mudnn = bias;
  SlimTensor* running_mean_for_mudnn = running_mean;
  SlimTensor* running_var_for_mudnn = running_var;
  if (!input->is_contiguous()) {
    contiguous_input = std::make_unique<SlimTensor>(input->clone_contiguous());
    input_for_mudnn = contiguous_input.get();
  }
  if (!weight->is_contiguous()) {
    contiguous_weight =
        std::make_unique<SlimTensor>(weight->clone_contiguous());
    weight_for_mudnn = contiguous_weight.get();
  }
  if (!bias->is_contiguous()) {
    contiguous_bias = std::make_unique<SlimTensor>(bias->clone_contiguous());
    bias_for_mudnn = contiguous_bias.get();
  }
  if (!running_mean->is_contiguous()) {
    contiguous_running_mean =
        std::make_unique<SlimTensor>(running_mean->clone_contiguous());
    running_mean_for_mudnn = contiguous_running_mean.get();
  }
  if (!running_var->is_contiguous()) {
    contiguous_running_var =
        std::make_unique<SlimTensor>(running_var->clone_contiguous());
    running_var_for_mudnn = contiguous_running_var.get();
  }

  const int64_t channel_count = input->size(1);
  const int64_t descriptor_dims = input->dim() <= 3 ? 4 : input->dim();
  const auto data_type = mudnn_data_type(dtype);
  const auto parameter_data_type = mudnn_data_type(parameter_dtype);
  const auto tensor_format = descriptor_dims == 4 ? MUDNN_TENSOR_NCHW
                                                  : MUDNN_TENSOR_NCDHW;

  std::vector<int> input_dims(descriptor_dims);
  for (int64_t index = 0; index < input->dim(); ++index) {
    input_dims[index] = static_cast<int>(input->size(index));
  }
  if (input->dim() == 2) {
    input_dims[2] = 1;
    input_dims[3] = 1;
  } else if (input->dim() == 3) {
    input_dims[3] = 1;
  }
  std::vector<int> parameter_dims(descriptor_dims, 1);
  parameter_dims[1] = static_cast<int>(channel_count);

  std::vector<int64_t> output_sizes(input->sizes().begin(), input->sizes().end());
  std::vector<int64_t> output_strides(output_sizes.size(), 1);
  for (int64_t index = static_cast<int64_t>(output_sizes.size()) - 2;
       index >= 0;
       --index) {
    output_strides[index] = output_strides[index + 1] * output_sizes[index + 1];
  }
  SlimTensor* raw_output = nullptr;
  auto error = aoti_torch_empty_strided(
      output_sizes.size(),
      output_sizes.data(),
      output_strides.data(),
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

  mudnnTensorDescriptor_t* descriptors[] = {
      &context.input,
      &context.output,
      &context.scale,
      &context.bias,
      &context.running_mean,
      &context.running_var};
  for (auto* descriptor : descriptors) {
    error = check_mudnn(
        mudnnCreateTensorDescriptor(descriptor), "mudnnCreateTensorDescriptor");
    ET_CHECK_OK_OR_RETURN_ERROR(error);
  }
  error = check_mudnn(
      mudnnSetTensorNdDescriptorEx(
          context.input,
          tensor_format,
          data_type,
          static_cast<int>(descriptor_dims),
          input_dims.data()),
      "mudnnSetTensorNdDescriptorEx(input)");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  error = check_mudnn(
      mudnnSetTensorNdDescriptorEx(
          context.output,
          tensor_format,
          data_type,
          static_cast<int>(descriptor_dims),
          input_dims.data()),
      "mudnnSetTensorNdDescriptorEx(output)");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  for (auto* descriptor :
       {context.scale, context.bias, context.running_mean, context.running_var}) {
    error = check_mudnn(
        mudnnSetTensorNdDescriptorEx(
            descriptor,
            tensor_format,
            parameter_data_type,
            static_cast<int>(descriptor_dims),
            parameter_dims.data()),
        "mudnnSetTensorNdDescriptorEx(parameter)");
    ET_CHECK_OK_OR_RETURN_ERROR(error);
  }
  error = check_mudnn(
      mudnnCreateBatchNormDescriptor(&context.batch_norm),
      "mudnnCreateBatchNormDescriptor");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  error = check_mudnn(
      mudnnSetBatchNormDescriptor(
          context.batch_norm, MUDNN_BATCHNORM_SPATIAL, eps, false),
      "mudnnSetBatchNormDescriptor");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  error = check_mudnn(
      mudnnBatchNormForwardInference(
          context.handle,
          context.batch_norm,
          context.output,
          output->data_ptr(),
          context.input,
          input_for_mudnn->data_ptr(),
          context.running_mean,
          running_mean_for_mudnn->data_ptr(),
          context.running_var,
          running_var_for_mudnn->data_ptr(),
          context.scale,
          weight_for_mudnn->data_ptr(),
          context.bias,
          bias_for_mudnn->data_ptr()),
      "mudnnBatchNormForwardInference");
  ET_CHECK_OK_OR_RETURN_ERROR(error);

  *ret_output = output.release();
  return Error::Ok;
}

}

}
