#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/cuda/runtime/shims/musa_gguf.h>
#include <executorch/backends/cuda/runtime/shims/musa_tensor_utils.h>
#include <executorch/extension/cuda/runtime_api.h>

#include <limits>

extern "C" musaError_t executorch_musa_gguf_linear_launch(float*, const float*,
                                                          const uint8_t*,
                                                          int64_t, int64_t,
                                                          int64_t, int64_t,
                                                          musaStream_t);
extern "C" musaError_t executorch_musa_gguf_embedding_launch(
    float*, const int64_t*, const uint8_t*, int*, int64_t, int64_t, int64_t,
    int64_t, musaStream_t);

namespace executorch::backends::cuda {
namespace {

namespace slim = executorch::backends::aoti::slim;
using executorch::runtime::Error;
using slim::SlimTensor;
using slim::c10::ScalarType;

Error validate(SlimTensor* output, SlimTensor* input, SlimTensor* packed,
               int64_t quant_type, bool embedding, int64_t& rows,
               int64_t& columns, int64_t& batches) {
  const int64_t bytes = quant_type == 12   ? 144
                        : quant_type == 13 ? 176
                        : quant_type == 14 ? 210
                        : quant_type == 114 ? 320
                                           : 0;
  ET_CHECK_OR_RETURN_ERROR(bytes != 0, InvalidArgument,
                           "Unknown GGUF K-quant format");
  ET_CHECK_OR_RETURN_ERROR(
      output != nullptr && input != nullptr && packed != nullptr &&
          output->storage() != nullptr && input->storage() != nullptr &&
          packed->storage() != nullptr && input->is_gpu() && output->is_gpu() &&
          packed->is_gpu() && input->device_index() == output->device_index() &&
          input->device_index() == packed->device_index() &&
          packed->dtype() == ScalarType::Byte &&
          output->dtype() == ScalarType::Float &&
          input->dtype() ==
              (embedding ? ScalarType::Long : ScalarType::Float) &&
          packed->is_contiguous() && input->is_contiguous() &&
          output->is_contiguous(),
      InvalidArgument,
      "GGUF requires contiguous, matching-device packed bytes and FP32 tensors "
      "or int64 indices");
  ET_CHECK_OR_RETURN_ERROR(
      packed->dim() == 2 && packed->size(0) > 0 && packed->size(1) > 0 &&
          packed->size(1) % bytes == 0 &&
          packed->size(1) / bytes <=
              std::numeric_limits<int64_t>::max() / 256 &&
          input->dim() >= 1 && input->dim() <= 4,
      InvalidArgument,
      "GGUF requires complete nonempty rows and input rank one through four");
  rows = packed->size(0);
  columns = packed->size(1) / bytes * 256;
  ET_CHECK_OR_RETURN_ERROR(
      rows <= std::numeric_limits<int64_t>::max() / packed->size(1),
      InvalidArgument, "GGUF packed storage size overflows");
  const int64_t leading = embedding ? input->dim() : input->dim() - 1;
  ET_CHECK_OR_RETURN_ERROR(
      (embedding || input->size(input->dim() - 1) == columns) &&
          output->dim() == leading + 1 &&
          output->size(leading) == (embedding ? columns : rows),
      InvalidArgument,
      "GGUF input/output dimensions do not match the packed weight");
  batches = 1;
  for (int64_t index = 0; index < leading; ++index) {
    const auto size = input->size(index);
    ET_CHECK_OR_RETURN_ERROR(
        size > 0 && size == output->size(index) &&
            size <= std::numeric_limits<int64_t>::max() / batches,
        InvalidArgument,
        "GGUF batch dimensions are empty, mismatched or overflowing");
    batches *= size;
  }
  const int64_t width = embedding ? columns : rows;
  ET_CHECK_OR_RETURN_ERROR(
      columns <= std::numeric_limits<int64_t>::max() / batches &&
          width <= (std::numeric_limits<int32_t>::max() - 255) / batches,
      NotSupported, "GGUF launch size exceeds the supported grid");
  auto input_span = checked_musa_tensor_span(*input);
  ET_CHECK_OK_OR_RETURN_ERROR(input_span.error());
  auto packed_span = checked_musa_tensor_span(*packed);
  ET_CHECK_OK_OR_RETURN_ERROR(packed_span.error());
  auto output_span = checked_musa_tensor_span(*output);
  ET_CHECK_OK_OR_RETURN_ERROR(output_span.error());
  ET_CHECK_OR_RETURN_ERROR(
      !musa_tensor_spans_overlap(output_span.get(), input_span.get()) &&
          !musa_tensor_spans_overlap(output_span.get(), packed_span.get()),
      InvalidArgument, "GGUF output must not overlap inputs or packed weights");
  return Error::Ok;
}

}

extern "C" AOTI_SHIM_EXPORT Error
aoti_torch_musa_gguf_linear_out(SlimTensor* output, SlimTensor* input,
                                SlimTensor* packed, int64_t quant_type) {
  int64_t rows, columns, batches;
  ET_CHECK_OK_OR_RETURN_ERROR(validate(output, input, packed, quant_type, false,
                                       rows, columns, batches));
  auto guard = CUDAGuard::create(input->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(guard.error());
  auto stream = getCurrentCUDAStream(input->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(stream.error());
  const auto status = executorch_musa_gguf_linear_launch(
      static_cast<float*>(output->data_ptr()),
      static_cast<const float*>(input->data_ptr()),
      static_cast<const uint8_t*>(packed->data_ptr()), rows, columns, batches,
      quant_type, stream.get());
  ET_CHECK_OR_RETURN_ERROR(status == musaSuccess, Internal,
                           "GGUF linear kernel launch failed: %s",
                           musaGetErrorString(status));
  return Error::Ok;
}

extern "C" AOTI_SHIM_EXPORT Error
aoti_torch_musa_gguf_embedding_out(SlimTensor* output, SlimTensor* input,
                                   SlimTensor* packed, int64_t quant_type) {
  int64_t rows, columns, batches;
  ET_CHECK_OK_OR_RETURN_ERROR(validate(output, input, packed, quant_type, true,
                                       rows, columns, batches));
  auto guard = CUDAGuard::create(input->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(guard.error());
  auto stream = getCurrentCUDAStream(input->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(stream.error());
  auto invalid = slim::empty({1}, ScalarType::Int, input->device());
  auto status =
      cudaMemsetAsync(invalid.data_ptr(), 0, sizeof(int), stream.get());
  ET_CHECK_OR_RETURN_ERROR(status == musaSuccess, Internal,
                           "GGUF index validation initialization failed");
  status = executorch_musa_gguf_embedding_launch(
      static_cast<float*>(output->data_ptr()),
      static_cast<const int64_t*>(input->data_ptr()),
      static_cast<const uint8_t*>(packed->data_ptr()),
      static_cast<int*>(invalid.data_ptr()), rows, columns, batches, quant_type,
      stream.get());
  ET_CHECK_OR_RETURN_ERROR(status == musaSuccess, Internal,
                           "GGUF embedding kernel launch failed: %s",
                           musaGetErrorString(status));
  int invalid_index = 0;
  status = cudaMemcpyAsync(&invalid_index, invalid.data_ptr(), sizeof(int),
                           cudaMemcpyDeviceToHost, stream.get());
  ET_CHECK_OR_RETURN_ERROR(status == musaSuccess, Internal,
                           "GGUF index validation readback failed");
  status = cudaStreamSynchronize(stream.get());
  ET_CHECK_OR_RETURN_ERROR(status == musaSuccess, Internal,
                           "GGUF index validation synchronization failed");
  ET_CHECK_OR_RETURN_ERROR(invalid_index == 0, InvalidArgument,
                           "GGUF embedding index is outside the vocabulary");
  return Error::Ok;
}

}
