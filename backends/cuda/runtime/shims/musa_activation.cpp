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

  const size_t storage_elements =
      input->storage()->nbytes() / input->itemsize();
  ET_CHECK_OR_RETURN_ERROR(
      input->storage_offset() >= 0 &&
          static_cast<uint64_t>(input->storage_offset()) <= storage_elements,
      InvalidArgument,
      "MUSA activation input exceeds its storage");
  size_t source_bytes = 0;
  if (input->numel() != 0) {
    const size_t available_elements =
        storage_elements - input->storage_offset();
    ET_CHECK_OR_RETURN_ERROR(
        available_elements > 0,
        InvalidArgument,
        "MUSA activation input exceeds its storage");
    size_t last_element = 0;
    for (int64_t dimension = 0; dimension < input->dim(); ++dimension) {
      const auto size = input->size(dimension);
      const auto stride = input->stride(dimension);
      ET_CHECK_OR_RETURN_ERROR(
          size > 0 && stride >= 0 &&
              (stride == 0 ||
               static_cast<uint64_t>(size - 1) <=
                   (available_elements - 1 - last_element) / stride),
          InvalidArgument,
          "MUSA activation input strides exceed storage");
      last_element += (size - 1) * stride;
    }
    source_bytes = (last_element + 1) * input->itemsize();
  }
  if (output != nullptr) {
    ET_CHECK_OR_RETURN_ERROR(
        output->storage() != nullptr && output->is_gpu() &&
            output->device_index() == input->device_index() &&
            output->dtype() == dtype && output->sizes() == input->sizes() &&
            output->is_contiguous(),
        InvalidArgument,
        "MUSA activation output must be contiguous with matching shape, dtype and device");
    ET_CHECK_OR_RETURN_ERROR(
        output->storage_offset() >= 0 &&
            static_cast<uint64_t>(output->storage_offset()) <=
                output->storage()->nbytes() / output->itemsize() &&
            output->nbytes() <= output->storage()->nbytes() -
                    output->storage_offset() * output->itemsize(),
        InvalidArgument,
        "MUSA activation output exceeds its storage");
    if (source_bytes != 0) {
      const auto source_start = reinterpret_cast<uintptr_t>(input->data_ptr());
      const auto output_start = reinterpret_cast<uintptr_t>(output->data_ptr());
      ET_CHECK_OR_RETURN_ERROR(
          output_start <= source_start
              ? source_start - output_start >= output->nbytes()
              : output_start - source_start >= source_bytes,
          InvalidArgument,
          "MUSA activation output must not overlap input storage");
    }
  }

  std::unique_ptr<SlimTensor> owned_output;
  if (output == nullptr) {
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

}
