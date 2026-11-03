#include <array>

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/cuda/runtime/shims/musa_gdn.h>
#include <executorch/backends/cuda/runtime/shims/musa_tensor_utils.h>
#include <executorch/extension/cuda/runtime_api.h>

extern "C" musaError_t executorch_musa_gdn_launch(
    float*,
    float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    int,
    musaStream_t);

namespace executorch::backends::cuda {

extern "C" AOTI_SHIM_EXPORT executorch::runtime::Error
aoti_torch_musa_gdn_recurrence_out(
    MusaGdnTensor output,
    MusaGdnTensor query,
    MusaGdnTensor key,
    MusaGdnTensor value,
    MusaGdnTensor decay,
    MusaGdnTensor beta,
    MusaGdnTensor state) {
  using executorch::backends::aoti::slim::c10::ScalarType;
  using executorch::runtime::Error;
  const std::array<MusaGdnTensor, 7> tensors = {
      output, query, key, value, decay, beta, state};
  for (auto* tensor : tensors) {
    ET_CHECK_OR_RETURN_ERROR(
        tensor != nullptr && tensor->storage() != nullptr && tensor->is_gpu() &&
            tensor->dtype() == ScalarType::Float && tensor->is_contiguous(),
        InvalidArgument,
        "GDN recurrence requires backed contiguous GPU FP32 tensors");
  }
  for (auto* tensor : tensors) {
    ET_CHECK_OR_RETURN_ERROR(
        tensor->device_index() == query->device_index(),
        InvalidArgument,
        "GDN recurrence tensors must share a device");
  }
  ET_CHECK_OR_RETURN_ERROR(
      query->dim() == 4 && query->size(0) == 1 &&
          (query->size(1) == 16 || query->size(1) == 32) &&
          query->size(2) == 1 && query->size(3) == 128,
      InvalidArgument,
      "GDN recurrence supports one token and 16 or 32 128-wide heads");
  const auto heads = query->size(1);
  for (auto* tensor : {key, value}) {
    ET_CHECK_OR_RETURN_ERROR(
        tensor->dim() == 4 && tensor->size(0) == 1 &&
            tensor->size(1) == heads && tensor->size(2) == 1 &&
            tensor->size(3) == 128,
        InvalidArgument,
        "GDN query, key and value shapes must match");
  }
  for (auto* tensor : {decay, beta}) {
    ET_CHECK_OR_RETURN_ERROR(
        tensor->dim() == 3 && tensor->size(0) == 1 &&
            tensor->size(1) == heads && tensor->size(2) == 1,
        InvalidArgument,
        "GDN decay and beta require one scalar per head");
  }
  ET_CHECK_OR_RETURN_ERROR(
      state->dim() == 4 && state->size(0) == 1 && state->size(1) == heads &&
          state->size(2) == 128 && state->size(3) == 128 &&
          output->dim() == 1 && output->size(0) == heads * (128 * 128 + 128),
      InvalidArgument,
      "GDN state or combined output shape is invalid");
  auto output_span = checked_musa_tensor_span(*output);
  ET_CHECK_OK_OR_RETURN_ERROR(output_span.error());
  for (auto* tensor : {query, key, value, decay, beta, state}) {
    auto span = checked_musa_tensor_span(*tensor);
    ET_CHECK_OK_OR_RETURN_ERROR(span.error());
    ET_CHECK_OR_RETURN_ERROR(
        !musa_tensor_spans_overlap(output_span.get(), span.get()),
        InvalidArgument,
        "GDN output must not overlap inputs");
  }
  auto guard = CUDAGuard::create(query->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(guard.error());
  auto stream = getCurrentCUDAStream(query->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(stream.error());
  const auto status = executorch_musa_gdn_launch(
      static_cast<float*>(output->data_ptr()) + heads * 128 * 128,
      static_cast<float*>(output->data_ptr()),
      static_cast<const float*>(query->data_ptr()),
      static_cast<const float*>(key->data_ptr()),
      static_cast<const float*>(value->data_ptr()),
      static_cast<const float*>(decay->data_ptr()),
      static_cast<const float*>(beta->data_ptr()),
      static_cast<const float*>(state->data_ptr()),
      static_cast<int>(heads),
      stream.get());
  ET_CHECK_OR_RETURN_ERROR(
      status == musaSuccess,
      Internal,
      "MUSA GDN recurrence launch failed: %s",
      musaGetErrorString(status));
  return Error::Ok;
}

}
