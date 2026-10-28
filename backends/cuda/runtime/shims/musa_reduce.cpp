#include <mudnncxx/mudnn.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/cuda/runtime/shims/musa_reduce.h>
#include <executorch/backends/cuda/runtime/shims/musa_tensor_utils.h>
#include <executorch/extension/cuda/cuda_allocator.h>
#include <executorch/extension/cuda/runtime_api.h>

namespace executorch::backends::cuda {
namespace {

namespace dnn = ::musa::dnn;
using executorch::backends::aoti::slim::SlimTensor;
using executorch::backends::aoti::slim::c10::ScalarType;
using executorch::extension::cuda::CudaAllocator;
using executorch::runtime::Error;

Error check_mudnn(dnn::Status status) {
  ET_CHECK_OR_RETURN_ERROR(
      status == dnn::Status::SUCCESS,
      Internal,
      "MUSA reduction failed with muDNN status %d",
      static_cast<int>(status));
  return Error::Ok;
}

Error run_reduction(
    SlimTensor* output,
    SlimTensor* input,
    int64_t axis,
    bool keepdim,
    bool mean,
    ScalarType dtype) {
  ET_CHECK_OR_RETURN_ERROR(
      input != nullptr && output != nullptr && input->storage() != nullptr &&
          output->storage() != nullptr && input->is_gpu() && output->is_gpu() &&
          input->device_index() == output->device_index() &&
          input->dtype() == dtype && output->dtype() == dtype,
      InvalidArgument,
      "MUSA reduction requires backed tensors with matching dtype and device");
  ET_CHECK_OR_RETURN_ERROR(
      input->dim() >= 1 && input->dim() <= 8,
      NotSupported,
      "MUSA reduction supports ranks one through eight");
  const auto rank = static_cast<int64_t>(input->dim());
  ET_CHECK_OR_RETURN_ERROR(
      axis >= -rank && axis < rank,
      InvalidArgument,
      "MUSA reduction axis is outside the input rank");
  if (axis < 0) {
    axis += rank;
  }
  int64_t elements = 1;
  for (const auto size : input->sizes()) {
    ET_CHECK_OR_RETURN_ERROR(
        size > 0, NotSupported, "MUSA reduction requires nonempty input");
    ET_CHECK_OR_RETURN_ERROR(
        size <= std::numeric_limits<int64_t>::max() / elements,
        InvalidArgument,
        "MUSA reduction element count overflows");
    elements *= size;
  }
  ET_CHECK_OR_RETURN_ERROR(
      static_cast<uint64_t>(elements) <=
          std::numeric_limits<size_t>::max() / input->itemsize(),
      InvalidArgument,
      "MUSA reduction byte count overflows");
  const bool gdn_shape = input->dim() == 4 && input->size(0) == 1 &&
      input->size(1) == 16 && input->size(2) == 128 && input->size(3) == 128;
  const bool bf16_shape = input->dim() == 4 && input->size(0) == 1 &&
      input->size(1) == 1 && input->size(2) == 16 && input->size(3) == 128;
  const auto width = input->size(input->dim() - 1);
  const bool fp32_shape = (axis == input->dim() - 1 &&
                           (width == 128 || width == 256 || width == 1024)) ||
      (!mean && axis == 2 && gdn_shape);
  ET_CHECK_OR_RETURN_ERROR(
      dtype == ScalarType::Float ? fp32_shape
                                 : bf16_shape && axis == 3 && !mean,
      NotSupported,
      "MUSA reduction shape or mode is outside the supported scope");
  std::vector<int64_t> expected(input->sizes().begin(), input->sizes().end());
  if (keepdim) {
    expected[axis] = 1;
  } else {
    expected.erase(expected.begin() + axis);
  }
  ET_CHECK_OR_RETURN_ERROR(
      static_cast<size_t>(output->dim()) == expected.size() &&
          std::equal(
              expected.begin(), expected.end(), output->sizes().begin()) &&
          output->is_contiguous(),
      InvalidArgument,
      "MUSA reduction output must have the exact contiguous reduced shape");
  auto input_span = checked_musa_tensor_span(*input);
  ET_CHECK_OK_OR_RETURN_ERROR(input_span.error());
  auto output_span = checked_musa_tensor_span(*output);
  ET_CHECK_OK_OR_RETURN_ERROR(output_span.error());
  ET_CHECK_OR_RETURN_ERROR(
      !musa_tensor_spans_overlap(input_span.get(), output_span.get()),
      InvalidArgument,
      "MUSA reduction output must not overlap its input");
  auto guard = CUDAGuard::create(input->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(guard.error());
  std::unique_ptr<SlimTensor> contiguous;
  if (!input->is_contiguous()) {
    contiguous = std::make_unique<SlimTensor>(input->clone_contiguous());
    input = contiguous.get();
  }
  auto stream_result = getCurrentCUDAStream(input->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(stream_result.error());
  const auto stream = stream_result.get();
  const auto native_dtype = dtype == ScalarType::Float
      ? dnn::Tensor::Type::FLOAT
      : dnn::Tensor::Type::BFLOAT16;
  dnn::Handle handle(input->device_index());
  dnn::Tensor source, destination;
  dnn::Reduce operation;
  std::vector<int64_t> destination_sizes(
      input->sizes().begin(), input->sizes().end());
  destination_sizes[axis] = 1;
  for (auto status :
       {handle.SetStream(stream),
        source.SetType(native_dtype),
        destination.SetType(native_dtype),
        source.SetNdInfo(
            static_cast<int64_t>(input->dim()), input->sizes().data()),
        destination.SetNdInfo(
            static_cast<int64_t>(input->dim()), destination_sizes.data()),
        source.SetAddr(input->data_ptr()),
        destination.SetAddr(output->data_ptr()),
        operation.SetMode(
            mean ? dnn::Reduce::Mode::MEAN : dnn::Reduce::Mode::ADD),
        operation.SetDim({static_cast<int>(axis)})}) {
    ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(status));
  }
  Error workspace_error = Error::Ok;
  dnn::MemoryMaintainer workspace = [&](size_t bytes) {
    auto allocation =
        CudaAllocator::allocate_async(bytes, input->device_index(), stream);
    if (!allocation.ok()) {
      workspace_error = allocation.error();
      return dnn::MemoryHandler(nullptr, [](void*) {});
    }
    const auto device = input->device_index();
    return dnn::MemoryHandler(
        allocation.get(), [device, stream](void* address) {
          CudaAllocator::deallocate_async(address, device, stream);
        });
  };
  const auto status = operation.Run(handle, destination, source, workspace);
  ET_CHECK_OK_OR_RETURN_ERROR(workspace_error);
  return check_mudnn(status);
}

}

extern "C" AOTI_SHIM_EXPORT Error aoti_torch_musa_reduce_out(
    SlimTensor* output,
    SlimTensor* input,
    int64_t axis,
    bool keepdim,
    bool mean) {
  return run_reduction(output, input, axis, keepdim, mean, ScalarType::Float);
}

extern "C" AOTI_SHIM_EXPORT Error aoti_torch_musa_bf16_sum_out(
    SlimTensor* output,
    SlimTensor* input,
    int64_t axis,
    bool keepdim,
    bool mean) {
  return run_reduction(
      output, input, axis, keepdim, mean, ScalarType::BFloat16);
}

}
