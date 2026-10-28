#include <mudnncxx/mudnn.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/cuda/runtime/shims/musa_causal_attention.h>
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
      "MUSA causal attention failed with muDNN status %d",
      static_cast<int>(status));
  return Error::Ok;
}

Error set_tensor(
    dnn::Tensor& tensor,
    const void* address,
    dnn::Tensor::Type dtype,
    std::initializer_list<int64_t> sizes) {
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(tensor.SetType(dtype)));
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(
      tensor.SetNdInfo(static_cast<int64_t>(sizes.size()), sizes.begin())));
  return check_mudnn(tensor.SetAddr(address));
}

struct AttentionScratch {
  void* data = nullptr;
  int device;
  cudaStream_t stream;

  ~AttentionScratch() {
    if (data != nullptr) {
      CudaAllocator::deallocate_async(data, device, stream);
    }
  }
};

Error run_unary(
    dnn::Handle& handle,
    dnn::Tensor& output,
    const dnn::Tensor& input,
    dnn::Unary::Mode mode) {
  dnn::Unary operation;
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(operation.SetMode(mode)));
  if (mode == dnn::Unary::Mode::MUL) {
    ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(operation.SetAlpha(0.0625)));
  }
  return check_mudnn(operation.Run(handle, output, input));
}

}

extern "C" AOTI_SHIM_EXPORT Error aoti_torch_musa_causal_attention_out(
    SlimTensor* output,
    SlimTensor* query,
    SlimTensor* key,
    SlimTensor* value,
    SlimTensor* position,
    SlimTensor* mask) {
  for (auto* tensor : {output, query, key, value, position, mask}) {
    ET_CHECK_OR_RETURN_ERROR(
        tensor != nullptr && tensor->storage() != nullptr && tensor->is_gpu(),
        InvalidArgument,
        "MUSA causal attention requires backed device tensors");
  }
  for (auto* tensor : {output, query, key, value}) {
    ET_CHECK_OR_RETURN_ERROR(
        tensor->dtype() == ScalarType::BFloat16 && tensor->dim() == 4,
        NotSupported,
        "MUSA causal attention requires rank-four BF16 tensors");
  }
  const int64_t capacity = key->size(2);
  ET_CHECK_OR_RETURN_ERROR(
      query->size(0) == 1 && query->size(1) == 8 && query->size(2) == 1 &&
          query->size(3) == 256 && output->sizes() == query->sizes() &&
          key->size(0) == 1 && key->size(1) == 8 && key->size(3) == 256 &&
          (capacity == 64 || capacity == 128 || capacity == 256) &&
          value->sizes() == key->sizes(),
      NotSupported,
      "MUSA causal attention supports Qwen single-token geometry only");
  ET_CHECK_OR_RETURN_ERROR(
      position->dtype() == ScalarType::Long && position->dim() == 1 &&
          position->size(0) == 1 && mask->dtype() == ScalarType::Bool &&
          mask->dim() == 2 && mask->size(0) == 1 && mask->size(1) == capacity,
      InvalidArgument,
      "MUSA causal attention requires int64 position[1] and bool "
      "mask[1,capacity]");
  const auto output_span = checked_musa_tensor_span(*output);
  ET_CHECK_OK_OR_RETURN_ERROR(output_span.error());
  for (auto* tensor : {output, query, key, value, position, mask}) {
    ET_CHECK_OR_RETURN_ERROR(
        tensor->device_index() == query->device_index(),
        InvalidArgument,
        "MUSA causal attention tensors must share a device");
    ET_CHECK_OR_RETURN_ERROR(
        tensor->is_contiguous(),
        NotSupported,
        "MUSA causal attention requires contiguous tensors");
    const auto span = checked_musa_tensor_span(*tensor);
    ET_CHECK_OK_OR_RETURN_ERROR(span.error());
    ET_CHECK_OR_RETURN_ERROR(
        tensor == output ||
            !musa_tensor_spans_overlap(output_span.get(), span.get()),
        InvalidArgument,
        "MUSA causal attention output must not overlap its inputs");
  }
  ET_CHECK_OR_RETURN_ERROR(
      output != query && output != key && output != value &&
          output != position && output != mask,
      InvalidArgument,
      "MUSA causal attention does not support in-place output");
  auto device_guard = CUDAGuard::create(query->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(device_guard.error());
  auto stream_result = getCurrentCUDAStream(query->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(stream_result.error());
  const auto stream = stream_result.get();
  alignas(16) int64_t index = -1;
  alignas(16) std::array<uint8_t, 256> host_mask{};
  {
    struct ControlScratch {
      void* data = nullptr;
      cudaStream_t stream;
      bool pending = false;
      ~ControlScratch() {
        if (pending) {
          cudaStreamSynchronize(stream);
        }
        if (data != nullptr) {
          cudaFreeHost(data);
        }
      }
    };
    ControlScratch control{nullptr, stream};
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMallocHost(&control.data, 272));
    control.pending = true;
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpyAsync(
        control.data,
        position->data_ptr(),
        sizeof(index),
        cudaMemcpyDeviceToHost,
        stream));
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpyAsync(
        static_cast<char*>(control.data) + 16,
        mask->data_ptr(),
        capacity,
        cudaMemcpyDeviceToHost,
        stream));
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaStreamSynchronize(stream));
    control.pending = false;
    std::memcpy(&index, control.data, sizeof(index));
    std::memcpy(host_mask.data(), static_cast<char*>(control.data) + 16, capacity);
  }
  ET_CHECK_OR_RETURN_ERROR(
      index >= 0 && index < capacity,
      InvalidArgument,
      "MUSA causal attention position is outside the KV cache");
  const int64_t length = index + 1;
  for (int64_t column = 0; column < capacity; ++column) {
    ET_CHECK_OR_RETURN_ERROR(
        host_mask[column] == static_cast<uint8_t>(column < length),
        InvalidArgument,
        "MUSA causal attention mask must describe the valid causal prefix");
  }

  const size_t score_bytes = 8 * length * sizeof(uint16_t);
  const std::array<size_t, 5> sizes = {
      score_bytes,
      score_bytes,
      score_bytes * 2,
      score_bytes * 2,
      score_bytes};
  std::array<size_t, 5> offsets{};
  size_t allocation_size = 0;
  for (size_t buffer = 0; buffer < sizes.size(); ++buffer) {
    offsets[buffer] = allocation_size;
    allocation_size += (sizes[buffer] + 255) / 256 * 256;
  }
  AttentionScratch scratch{nullptr, query->device_index(), stream};
  auto allocation = CudaAllocator::allocate_async(
      allocation_size, query->device_index(), stream);
  ET_CHECK_OK_OR_RETURN_ERROR(allocation.error());
  scratch.data = allocation.get();
  std::array<void*, 5> buffers{};
  for (size_t buffer = 0; buffer < buffers.size(); ++buffer) {
    buffers[buffer] = static_cast<char*>(scratch.data) + offsets[buffer];
  }

  dnn::Handle handle(query->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(handle.SetStream(stream)));
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(handle.SetAllowTF32(false)));
  dnn::Tensor query_tensor, key_tensor, value_tensor, score_tensor,
      scaled_tensor;
  dnn::Tensor fp32_scores, fp32_probabilities, probabilities, output_tensor;
  const auto bf16 = dnn::Tensor::Type::BFLOAT16;
  const auto fp32 = dnn::Tensor::Type::FLOAT;
  ET_CHECK_OK_OR_RETURN_ERROR(
      set_tensor(query_tensor, query->data_ptr(), bf16, {8, 1, 256}));
  ET_CHECK_OK_OR_RETURN_ERROR(
      set_tensor(key_tensor, key->data_ptr(), bf16, {8, length, 256}));
  ET_CHECK_OK_OR_RETURN_ERROR(
      set_tensor(value_tensor, value->data_ptr(), bf16, {8, length, 256}));
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(key_tensor.SetNdInfo(
      {8, length, 256}, {capacity * 256, 256, 1})));
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(value_tensor.SetNdInfo(
      {8, length, 256}, {capacity * 256, 256, 1})));
  ET_CHECK_OK_OR_RETURN_ERROR(
      set_tensor(score_tensor, buffers[0], bf16, {8, 1, length}));
  ET_CHECK_OK_OR_RETURN_ERROR(
      set_tensor(scaled_tensor, buffers[1], bf16, {8, 1, length}));
  ET_CHECK_OK_OR_RETURN_ERROR(
      set_tensor(fp32_scores, buffers[2], fp32, {8, 1, length}));
  ET_CHECK_OK_OR_RETURN_ERROR(
      set_tensor(fp32_probabilities, buffers[3], fp32, {8, 1, length}));
  ET_CHECK_OK_OR_RETURN_ERROR(
      set_tensor(probabilities, buffers[4], bf16, {8, 1, length}));
  ET_CHECK_OK_OR_RETURN_ERROR(
      set_tensor(output_tensor, output->data_ptr(), bf16, {8, 1, 256}));
  dnn::BatchMatMul product;
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(
      product.SetComputeMode(dnn::BatchMatMul::ComputeMode::TENSOR)));
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(product.SetAlpha(1.0)));
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(product.SetBeta(0.0)));
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(product.SetTranspose(false, true)));
  ET_CHECK_OK_OR_RETURN_ERROR(
      check_mudnn(product.Run(handle, score_tensor, query_tensor, key_tensor)));
  ET_CHECK_OK_OR_RETURN_ERROR(
      run_unary(handle, scaled_tensor, score_tensor, dnn::Unary::Mode::MUL));
  ET_CHECK_OK_OR_RETURN_ERROR(
      run_unary(handle, fp32_scores, scaled_tensor, dnn::Unary::Mode::CAST));
  ET_CHECK_OK_OR_RETURN_ERROR(
      set_tensor(fp32_scores, buffers[2], fp32, {1, 8, 1, length}));
  ET_CHECK_OK_OR_RETURN_ERROR(
      set_tensor(fp32_probabilities, buffers[3], fp32, {1, 8, 1, length}));
  dnn::Softmax softmax;
  ET_CHECK_OK_OR_RETURN_ERROR(
      check_mudnn(softmax.SetMode(dnn::Softmax::Mode::SOFTMAX)));
  ET_CHECK_OK_OR_RETURN_ERROR(
      check_mudnn(softmax.SetAlgorithm(dnn::Softmax::Algorithm::ACCURATE)));
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(softmax.SetDim(3)));
  Error workspace_error = Error::Ok;
  dnn::MemoryMaintainer workspace = [&](size_t bytes) {
    auto memory =
        CudaAllocator::allocate_async(bytes, query->device_index(), stream);
    if (!memory.ok()) {
      workspace_error = memory.error();
      return dnn::MemoryHandler(nullptr, [](void*) {});
    }
    const auto device = query->device_index();
    return dnn::MemoryHandler(memory.get(), [device, stream](void* address) {
      CudaAllocator::deallocate_async(address, device, stream);
    });
  };
  const auto softmax_status =
      softmax.Run(handle, fp32_probabilities, fp32_scores, workspace);
  ET_CHECK_OK_OR_RETURN_ERROR(workspace_error);
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(softmax_status));
  ET_CHECK_OK_OR_RETURN_ERROR(
      set_tensor(fp32_probabilities, buffers[3], fp32, {8, 1, length}));
  ET_CHECK_OK_OR_RETURN_ERROR(run_unary(
      handle, probabilities, fp32_probabilities, dnn::Unary::Mode::CAST));
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(product.SetTranspose(false, false)));
  return check_mudnn(
      product.Run(handle, output_tensor, probabilities, value_tensor));
}

}
