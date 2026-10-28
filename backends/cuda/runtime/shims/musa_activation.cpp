#include <mudnn.h>

#include <limits>
#include <memory>
#include <vector>

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/musa_activation.h>
#include <executorch/backends/cuda/runtime/shims/musa_tensor_utils.h>
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
    SlimTensor* output,
    SlimTensor** ret_output,
    mudnnUnaryMode_t mode) {
  ET_CHECK_OR_RETURN_ERROR(
      input != nullptr && input->storage() != nullptr &&
          (ret_output != nullptr || output != nullptr),
      InvalidArgument,
      "MUSA activation received a null tensor handle");
  if (ret_output != nullptr) {
    *ret_output = nullptr;
  }
  ET_CHECK_OR_RETURN_ERROR(
      input->is_gpu(),
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

  const bool fp32_math =
      mode == MUDNN_UNARY_EXP || mode == MUDNN_UNARY_SOFTPLUS;
  ET_CHECK_OR_RETURN_ERROR(
      !fp32_math || dtype == ScalarType::Float,
      NotSupported,
      "MUSA native exp and softplus require float32 input");
  const auto input_span = checked_musa_tensor_span(*input);
  ET_CHECK_OK_OR_RETURN_ERROR(input_span.error());
  if (output != nullptr) {
    ET_CHECK_OR_RETURN_ERROR(
        output->storage() != nullptr && output->is_gpu() &&
            output->device_index() == input->device_index() &&
            output->dtype() == dtype && output->sizes() == input->sizes() &&
            output->is_contiguous(),
        InvalidArgument,
        "MUSA activation output must be contiguous with matching shape, dtype "
        "and device");
    const auto output_span = checked_musa_tensor_span(*output);
    ET_CHECK_OK_OR_RETURN_ERROR(output_span.error());
    ET_CHECK_OR_RETURN_ERROR(
        !musa_tensor_spans_overlap(input_span.get(), output_span.get()),
        InvalidArgument,
        "MUSA activation output must not overlap input storage");
  }

  auto device_guard = CUDAGuard::create(input->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(device_guard.error());
  std::unique_ptr<SlimTensor> owned_output;
  if (output == nullptr) {
    std::vector<int64_t> sizes(input->sizes().begin(), input->sizes().end());
    std::vector<int64_t> strides(sizes.size(), 1);
    for (int64_t dimension = static_cast<int64_t>(sizes.size()) - 2;
         dimension >= 0;
         --dimension) {
      ET_CHECK_OR_RETURN_ERROR(
          sizes[dimension + 1] == 0 ||
              strides[dimension + 1] <=
                  std::numeric_limits<int64_t>::max() / sizes[dimension + 1],
          NotSupported,
          "MUSA activation contiguous strides exceed int64 limits");
      strides[dimension] = strides[dimension + 1] * sizes[dimension + 1];
    }
    SlimTensor* raw_output = nullptr;
    ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        strides.data(),
        static_cast<int32_t>(dtype),
        executorch::backends::aoti::slim::c10::device_type_to_aoti(
            executorch::backends::aoti::slim::c10::kGPU),
        input->device_index(),
        &raw_output));
    owned_output.reset(raw_output);
    output = raw_output;
  }
  if (input->numel() == 0) {
    if (ret_output != nullptr) {
      *ret_output = owned_output.release();
    }
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
  const bool softplus = mode == MUDNN_UNARY_SOFTPLUS;
  double alpha = softplus ? 1.0 : 0.166667;
  double beta = softplus ? 20.0 : 0.5;
  const bool hard_sigmoid = mode == MUDNN_UNARY_HARDSIGMOID;
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(mudnnSetUnaryDescriptor(
      context.unary,
      mode,
      hard_sigmoid || softplus ? MUDNN_PARAM_DATA_DOUBLE
                               : MUDNN_PARAM_DATA_NONE,
      hard_sigmoid || softplus ? &alpha : nullptr,
      hard_sigmoid || softplus ? MUDNN_PARAM_DATA_DOUBLE
                               : MUDNN_PARAM_DATA_NONE,
      hard_sigmoid || softplus ? &beta : nullptr)));
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(mudnnUnary(
      context.handle,
      context.unary,
      context.output,
      output->data_ptr(),
      context.input,
      input->data_ptr())));
  if (ret_output != nullptr) {
    *ret_output = owned_output.release();
  }
  return Error::Ok;
}

}

extern "C" AOTI_SHIM_EXPORT Error
aoti_torch_musa_hardswish(SlimTensor* input, SlimTensor** ret_output) {
  return run_activation(input, nullptr, ret_output, MUDNN_UNARY_HARDSWISH);
}

extern "C" AOTI_SHIM_EXPORT Error
aoti_torch_musa_hardsigmoid(SlimTensor* input, SlimTensor** ret_output) {
  return run_activation(input, nullptr, ret_output, MUDNN_UNARY_HARDSIGMOID);
}

extern "C" AOTI_SHIM_EXPORT Error
aoti_torch_musa_hardswish_out(SlimTensor* output, SlimTensor* input) {
  return run_activation(input, output, nullptr, MUDNN_UNARY_HARDSWISH);
}

extern "C" AOTI_SHIM_EXPORT Error
aoti_torch_musa_hardsigmoid_out(SlimTensor* output, SlimTensor* input) {
  return run_activation(input, output, nullptr, MUDNN_UNARY_HARDSIGMOID);
}

extern "C" AOTI_SHIM_EXPORT Error
aoti_torch_musa_exp_out(SlimTensor* output, SlimTensor* input) {
  return run_activation(input, output, nullptr, MUDNN_UNARY_EXP);
}

extern "C" AOTI_SHIM_EXPORT Error aoti_torch_musa_softplus_out(
    SlimTensor* output,
    SlimTensor* input,
    double beta,
    double threshold) {
  ET_CHECK_OR_RETURN_ERROR(
      beta == 1.0 && threshold == 20.0,
      NotSupported,
      "MUSA native softplus requires beta=1 and threshold=20");
  return run_activation(input, output, nullptr, MUDNN_UNARY_SOFTPLUS);
}

}
