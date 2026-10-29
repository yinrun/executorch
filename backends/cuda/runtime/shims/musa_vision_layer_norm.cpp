#include <mudnncxx/mudnn.h>

#include <array>
#include <cmath>

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/cuda/runtime/shims/musa_tensor_utils.h>
#include <executorch/backends/cuda/runtime/shims/musa_vision_layer_norm.h>
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
      "MUSA vision LayerNorm failed with muDNN status %d",
      static_cast<int>(status));
  return Error::Ok;
}

}

extern "C" AOTI_SHIM_EXPORT Error aoti_torch_musa_vision_layer_norm_out(
    SlimTensor* output,
    SlimTensor* input,
    SlimTensor* weight,
    SlimTensor* bias,
    double epsilon) {
  ET_CHECK_OR_RETURN_ERROR(
      output != nullptr && input != nullptr && weight != nullptr && bias != nullptr,
      InvalidArgument,
      "MUSA vision LayerNorm requires four tensors");
  const std::array<SlimTensor*, 4> tensors{output, input, weight, bias};
  std::array<MusaTensorSpan, 4> spans;
  for (size_t index = 0; index < tensors.size(); ++index) {
    const auto* tensor = tensors[index];
    ET_CHECK_OR_RETURN_ERROR(
        tensor->is_gpu() && tensor->device_index() == input->device_index() &&
            tensor->dtype() == ScalarType::Float && tensor->is_contiguous(),
        InvalidArgument,
        "MUSA vision LayerNorm requires contiguous FP32 tensors on one GPU");
    auto span = checked_musa_tensor_span(*tensor);
    ET_CHECK_OK_OR_RETURN_ERROR(span.error());
    spans[index] = span.get();
  }
  ET_CHECK_OR_RETURN_ERROR(
      input->dim() == 2 && (input->size(0) == 800 || input->size(0) == 1200) &&
          input->size(1) == 1024 && output->dim() == 2 &&
          output->size(0) == input->size(0) && output->size(1) == 1024 &&
          weight->dim() == 1 && weight->size(0) == 1024 && bias->dim() == 1 &&
          bias->size(0) == 1024 && std::isfinite(epsilon) && epsilon > 0,
      InvalidArgument,
      "MUSA vision LayerNorm requires an 800/1200 by 1024 affine input and positive epsilon");
  for (size_t index = 1; index < spans.size(); ++index) {
    ET_CHECK_OR_RETURN_ERROR(
        !musa_tensor_spans_overlap(spans[0], spans[index]),
        InvalidArgument,
        "MUSA vision LayerNorm output overlaps an input");
  }
  auto guard = CUDAGuard::create(input->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(guard.error());
  auto stream_result = getCurrentCUDAStream(input->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(stream_result.error());
  const auto stream = stream_result.get();
  const auto device = input->device_index();
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
  const int64_t rows = input->size(0);
  auto statistics = workspace(static_cast<size_t>(rows) * 2 * sizeof(float));
  ET_CHECK_OK_OR_RETURN_ERROR(workspace_error);
  dnn::Handle handle(device);
  std::array<dnn::Tensor, 4> descriptors;
  dnn::Tensor mean, reciprocal_std;
  dnn::LayerNorm operation;
  for (size_t index = 0; index < tensors.size(); ++index) {
    for (auto status : {
             descriptors[index].SetType(dnn::Tensor::Type::FLOAT),
             descriptors[index].SetNdInfo(
                 static_cast<int64_t>(tensors[index]->dim()),
                 tensors[index]->sizes().data()),
             descriptors[index].SetAddr(tensors[index]->data_ptr())}) {
      ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(status));
    }
  }
  const int64_t statistics_shape[]{rows, 1};
  for (auto status : {
           handle.SetStream(stream),
           mean.SetType(dnn::Tensor::Type::FLOAT),
           mean.SetNdInfo(2, statistics_shape),
           mean.SetAddr(statistics.get()),
           reciprocal_std.SetType(dnn::Tensor::Type::FLOAT),
           reciprocal_std.SetNdInfo(2, statistics_shape),
           reciprocal_std.SetAddr(static_cast<float*>(statistics.get()) + rows),
           operation.SetAxis({1}),
           operation.SetEpsilon(epsilon),
           operation.SetVarMode(dnn::LayerNorm::VarMode::WELFORD)}) {
    ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(status));
  }
  const auto status = operation.Run(
      handle,
      descriptors[0],
      mean,
      reciprocal_std,
      descriptors[1],
      descriptors[2],
      descriptors[3],
      workspace);
  ET_CHECK_OK_OR_RETURN_ERROR(workspace_error);
  return check_mudnn(status);
}

}
