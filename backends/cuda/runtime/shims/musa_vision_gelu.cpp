#include <mudnncxx/mudnn.h>

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/cuda/runtime/shims/musa_tensor_utils.h>
#include <executorch/backends/cuda/runtime/shims/musa_vision_gelu.h>
#include <executorch/extension/cuda/runtime_api.h>

namespace executorch::backends::cuda {
namespace {

namespace dnn = ::musa::dnn;
using executorch::backends::aoti::slim::SlimTensor;
using executorch::backends::aoti::slim::c10::ScalarType;
using executorch::runtime::Error;

Error check_mudnn(dnn::Status status) {
  ET_CHECK_OR_RETURN_ERROR(
      status == dnn::Status::SUCCESS,
      Internal,
      "MUSA vision gelu failed with muDNN status %d",
      static_cast<int>(status));
  return Error::Ok;
}

}

extern "C" AOTI_SHIM_EXPORT Error aoti_torch_musa_vision_gelu_out(
    SlimTensor* output,
    SlimTensor* input,
    int64_t mode) {
  ET_CHECK_OR_RETURN_ERROR(
      output != nullptr && input != nullptr && (mode == 0 || mode == 1),
      InvalidArgument,
      "MUSA vision GELU requires two tensors and mode 0 (erf) or 1 (tanh)");
  for (const auto* tensor : {input, output}) {
    ET_CHECK_OR_RETURN_ERROR(
        tensor->is_gpu() && tensor->device_index() == input->device_index() &&
            tensor->dtype() == ScalarType::Float && tensor->is_contiguous() &&
            tensor->dim() == 2 &&
            (tensor->size(0) == 200 || tensor->size(0) == 300 ||
             tensor->size(0) == 800 || tensor->size(0) == 1200) &&
            tensor->size(1) == 4096,
        InvalidArgument,
        "MUSA vision GELU requires contiguous FP32 [N,4096], N=200/300/800/1200");
  }
  ET_CHECK_OR_RETURN_ERROR(
      output->size(0) == input->size(0),
      InvalidArgument,
      "MUSA vision gelu requires matching input and output shapes");
  auto input_span = checked_musa_tensor_span(*input);
  auto output_span = checked_musa_tensor_span(*output);
  ET_CHECK_OK_OR_RETURN_ERROR(input_span.error());
  ET_CHECK_OK_OR_RETURN_ERROR(output_span.error());
  ET_CHECK_OR_RETURN_ERROR(
      !musa_tensor_spans_overlap(input_span.get(), output_span.get()),
      InvalidArgument,
      "MUSA vision gelu output overlaps its input");
  auto guard = CUDAGuard::create(input->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(guard.error());
  auto stream_result = getCurrentCUDAStream(input->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(stream_result.error());
  const auto stream = stream_result.get();
  const auto device = input->device_index();
  dnn::Handle handle(device);
  dnn::Tensor source, destination;
  dnn::Unary operation;
  for (auto status : {
           handle.SetStream(stream),
           source.SetType(dnn::Tensor::Type::FLOAT),
           source.SetNdInfo(2, input->sizes().data()),
           source.SetAddr(input->data_ptr()),
           destination.SetType(dnn::Tensor::Type::FLOAT),
           destination.SetNdInfo(2, output->sizes().data()),
           destination.SetAddr(output->data_ptr()),
           operation.SetMode(mode == 1 ? dnn::Unary::Mode::GELU_TANH
                                       : dnn::Unary::Mode::GELU)}) {
    ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(status));
  }
  return check_mudnn(operation.Run(handle, destination, source));
}

}
