#include <mudnn.h>

#include <limits>
#include <memory>
#include <vector>

#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/musa_activation.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {
namespace {

using executorch::backends::aoti::slim::c10::DeviceType;
using executorch::backends::aoti::slim::c10::ScalarType;

Error check_mudnn(mudnnStatus_t status) {
  ET_CHECK_OR_RETURN_ERROR(
      status == MUDNN_STATUS_SUCCESS,
      Internal,
      "MUSA activation failed with muDNN status %d",
      static_cast<int>(status));
  return Error::Ok;
}

struct UnaryContext {
  mudnnHandle_t handle = nullptr;
  mudnnTensorDescriptor_t input = nullptr;
  mudnnTensorDescriptor_t output = nullptr;
  mudnnUnaryDescriptor_t unary = nullptr;

  ~UnaryContext() {
    if (unary != nullptr) {
      mudnnDestroyUnaryDescriptor(unary);
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

Error run_activation(
    SlimTensor* input,
    SlimTensor** ret_output,
    mudnnUnaryMode_t mode) {
  ET_CHECK_OR_RETURN_ERROR(
      input != nullptr && ret_output != nullptr,
      InvalidArgument,
      "MUSA activation received a null tensor handle");
  *ret_output = nullptr;
  ET_CHECK_OR_RETURN_ERROR(
      input->is_cuda(),
      InvalidArgument,
      "MUSA activation expects a device tensor");
  const auto dtype = input->dtype();
  ET_CHECK_OR_RETURN_ERROR(
      dtype == ScalarType::Float || dtype == ScalarType::Half ||
          dtype == ScalarType::BFloat16,
      NotSupported,
      "MUSA activation supports float32, float16, and bfloat16 only");
  ET_CHECK_OR_RETURN_ERROR(
      input->numel() <= std::numeric_limits<int>::max(),
      NotSupported,
      "MUSA activation exceeds the muDNN descriptor size limit");

  std::vector<int64_t> sizes(input->sizes().begin(), input->sizes().end());
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int64_t dimension = static_cast<int64_t>(sizes.size()) - 2;
       dimension >= 0;
       --dimension) {
    strides[dimension] = strides[dimension + 1] * sizes[dimension + 1];
  }
  SlimTensor* raw_output = nullptr;
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      strides.data(),
      static_cast<int32_t>(dtype),
      static_cast<int32_t>(DeviceType::CUDA),
      input->device_index(),
      &raw_output));
  std::unique_ptr<SlimTensor> output(raw_output);
  if (input->numel() == 0) {
    *ret_output = output.release();
    return Error::Ok;
  }
  std::unique_ptr<SlimTensor> contiguous_input;
  if (!input->is_contiguous()) {
    contiguous_input = std::make_unique<SlimTensor>(input->clone_contiguous());
    input = contiguous_input.get();
  }

  UnaryContext context;
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(mudnnCreate(&context.handle)));
  auto stream = getCurrentCUDAStream(input->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(stream.error());
  ET_CHECK_OK_OR_RETURN_ERROR(
      check_mudnn(mudnnSetStream(context.handle, stream.get())));
  const auto data_type = dtype == ScalarType::Float ? MUDNN_DATA_FLOAT
      : dtype == ScalarType::Half                   ? MUDNN_DATA_HALF
                                                    : MUDNN_DATA_BFLOAT16;
  const int dimensions[] = {1, 1, 1, static_cast<int>(input->numel())};
  for (auto* descriptor : {&context.input, &context.output}) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        check_mudnn(mudnnCreateTensorDescriptor(descriptor)));
    ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(mudnnSetTensorNdDescriptorEx(
        *descriptor, MUDNN_TENSOR_NCHW, data_type, 4, dimensions)));
  }
  ET_CHECK_OK_OR_RETURN_ERROR(
      check_mudnn(mudnnCreateUnaryDescriptor(&context.unary)));
  double alpha = 0.166667;
  double beta = 0.5;
  const bool hard_sigmoid = mode == MUDNN_UNARY_HARDSIGMOID;
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(mudnnSetUnaryDescriptor(
      context.unary,
      mode,
      hard_sigmoid ? MUDNN_PARAM_DATA_DOUBLE : MUDNN_PARAM_DATA_NONE,
      hard_sigmoid ? &alpha : nullptr,
      hard_sigmoid ? MUDNN_PARAM_DATA_DOUBLE : MUDNN_PARAM_DATA_NONE,
      hard_sigmoid ? &beta : nullptr)));
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(mudnnUnary(
      context.handle,
      context.unary,
      context.output,
      output->data_ptr(),
      context.input,
      input->data_ptr())));
  *ret_output = output.release();
  return Error::Ok;
}

}

extern "C" AOTI_SHIM_EXPORT Error
aoti_torch_musa_hardswish(SlimTensor* input, SlimTensor** ret_output) {
  return run_activation(input, ret_output, MUDNN_UNARY_HARDSWISH);
}

extern "C" AOTI_SHIM_EXPORT Error
aoti_torch_musa_hardsigmoid(SlimTensor* input, SlimTensor** ret_output) {
  return run_activation(input, ret_output, MUDNN_UNARY_HARDSIGMOID);
}

}
