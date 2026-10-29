#include <mudnncxx/mudnn.h>

#include <array>

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/cuda/runtime/shims/musa_tensor_utils.h>
#include <executorch/backends/cuda/runtime/shims/musa_vision_softmax.h>
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
      "MUSA vision softmax failed with muDNN status %d",
      static_cast<int>(status));
  return Error::Ok;
}

}

extern "C" AOTI_SHIM_EXPORT Error aoti_torch_musa_vision_softmax_out(
    SlimTensor* output,
    SlimTensor* input) {
  ET_CHECK_OR_RETURN_ERROR(
      output != nullptr && input != nullptr,
      InvalidArgument,
      "MUSA vision softmax requires two tensors");
  for (const auto* tensor : {input, output}) {
    ET_CHECK_OR_RETURN_ERROR(
        tensor->is_gpu() && tensor->device_index() == input->device_index() &&
            tensor->dtype() == ScalarType::Float && tensor->is_contiguous() &&
            tensor->dim() == 3 && tensor->size(0) == 16 &&
            (tensor->size(1) == 800 || tensor->size(1) == 1200) &&
            tensor->size(2) == tensor->size(1),
        InvalidArgument,
        "MUSA vision softmax requires contiguous FP32 [16,N,N], N=800/1200");
  }
  ET_CHECK_OR_RETURN_ERROR(
      output->size(1) == input->size(1),
      InvalidArgument,
      "MUSA vision softmax requires matching input and output shapes");
  auto input_span = checked_musa_tensor_span(*input);
  auto output_span = checked_musa_tensor_span(*output);
  ET_CHECK_OK_OR_RETURN_ERROR(input_span.error());
  ET_CHECK_OK_OR_RETURN_ERROR(output_span.error());
  ET_CHECK_OR_RETURN_ERROR(
      !musa_tensor_spans_overlap(input_span.get(), output_span.get()),
      InvalidArgument,
      "MUSA vision softmax output overlaps its input");
  auto guard = CUDAGuard::create(input->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(guard.error());
  auto stream_result = getCurrentCUDAStream(input->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(stream_result.error());
  const auto stream = stream_result.get();
  const auto device = input->device_index();
  dnn::Handle handle(device);
  dnn::Tensor source, destination;
  dnn::Softmax operation;
  for (auto status : {
           handle.SetStream(stream),
           source.SetType(dnn::Tensor::Type::FLOAT),
           source.SetNdInfo(3, input->sizes().data()),
           source.SetAddr(input->data_ptr()),
           destination.SetType(dnn::Tensor::Type::FLOAT),
           destination.SetNdInfo(3, output->sizes().data()),
           destination.SetAddr(output->data_ptr()),
           operation.SetDim(2),
           operation.SetMode(dnn::Softmax::Mode::SOFTMAX),
           operation.SetAlgorithm(dnn::Softmax::Algorithm::ACCURATE)}) {
    ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(status));
  }
  Error workspace_error = Error::Ok;
  dnn::MemoryMaintainer workspace = [&](size_t bytes) {
    auto allocation = CudaAllocator::allocate_async(bytes, device, stream);
    if (!allocation.ok()) {
      workspace_error = allocation.error();
      return dnn::MemoryHandler(nullptr, [](void*) {});
    }
    return dnn::MemoryHandler(allocation.get(), [device, stream](void* address) {
      CudaAllocator::deallocate_async(address, device, stream);
    });
  };
  const auto status = operation.Run(handle, destination, source, workspace);
  ET_CHECK_OK_OR_RETURN_ERROR(workspace_error);
  return check_mudnn(status);
}

}
