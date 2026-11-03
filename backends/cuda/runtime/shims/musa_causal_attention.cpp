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

Error validate_attention(
    SlimTensor* output,
    SlimTensor* query,
    SlimTensor* key,
    SlimTensor* value) {
  for (auto* tensor : {output, query, key, value}) {
    ET_CHECK_OR_RETURN_ERROR(
        tensor != nullptr && tensor->storage() != nullptr && tensor->is_gpu(),
        InvalidArgument,
        "MUSA causal attention requires backed device tensors");
  }
  const bool grouped_fp32 = query->dtype() == ScalarType::Float;
  const int64_t query_heads = grouped_fp32 ? 16 : 8;
  const int64_t kv_heads = grouped_fp32 ? 4 : 8;
  for (auto* tensor : {output, query, key, value}) {
    ET_CHECK_OR_RETURN_ERROR(
        tensor->dtype() == query->dtype() && tensor->dim() == 4 &&
            (grouped_fp32 || tensor->dtype() == ScalarType::BFloat16),
        NotSupported,
        "MUSA causal attention requires matching rank-four BF16 or FP32 tensors");
  }
  const int64_t capacity = key->size(2);
  ET_CHECK_OR_RETURN_ERROR(
      query->size(0) == 1 && query->size(1) == query_heads &&
          query->size(2) == 1 && query->size(3) == 256 &&
          output->sizes() == query->sizes() && key->size(0) == 1 &&
          key->size(1) == kv_heads && key->size(3) == 256 &&
          (grouped_fp32
               ? (capacity > 0 && capacity <= 4096)
               : (capacity == 64 || capacity == 128 || capacity == 256)) &&
          value->sizes() == key->sizes(),
      NotSupported,
      "MUSA causal attention supports Qwen single-token geometry only");
  const auto output_span = checked_musa_tensor_span(*output);
  ET_CHECK_OK_OR_RETURN_ERROR(output_span.error());
  for (auto* tensor : {output, query, key, value}) {
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
      output != query && output != key && output != value,
      InvalidArgument,
      "MUSA causal attention does not support in-place output");
  return Error::Ok;
}

Error validate_prefix_inputs(SlimTensor* position, SlimTensor* mask) {
  ET_CHECK_OR_RETURN_ERROR(
      position != nullptr && mask != nullptr &&
          position->storage() != nullptr && mask->storage() != nullptr &&
          position->is_gpu() && mask->is_gpu() &&
          position->device_index() == mask->device_index() &&
          position->dtype() == ScalarType::Long && position->dim() == 1 &&
          position->size(0) == 1 && mask->dtype() == ScalarType::Bool &&
          mask->dim() == 2 && mask->size(0) == 1 && mask->size(1) > 0 &&
          mask->size(1) <= 4096,
      InvalidArgument,
      "Causal prefix requires matching-device int64 position[1] and bool mask[1,capacity]");
  ET_CHECK_OR_RETURN_ERROR(
      position->is_contiguous() && mask->is_contiguous(),
      NotSupported,
      "Causal prefix requires contiguous controls");
  ET_CHECK_OK_OR_RETURN_ERROR(checked_musa_tensor_span(*position).error());
  ET_CHECK_OK_OR_RETURN_ERROR(checked_musa_tensor_span(*mask).error());
  return Error::Ok;
}

Error validate_host_prefix(SlimTensor* control) {
  ET_CHECK_OR_RETURN_ERROR(
      control != nullptr && control->storage() != nullptr &&
          control->is_cpu() && control->dtype() == ScalarType::Long &&
          control->dim() == 1 && control->size(0) == 2 &&
          control->is_contiguous(),
      InvalidArgument,
      "Causal prefix control requires a backed contiguous CPU int64[2] tensor");
  return checked_musa_tensor_span(*control).error();
}

Error read_prefix(SlimTensor* position, SlimTensor* mask, int64_t& index) {
  ET_CHECK_OK_OR_RETURN_ERROR(validate_prefix_inputs(position, mask));
  auto device_guard = CUDAGuard::create(position->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(device_guard.error());
  auto stream_result = getCurrentCUDAStream(position->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(stream_result.error());
  const auto stream = stream_result.get();
  const int64_t capacity = mask->size(1);
  index = -1;
  alignas(16) std::array<uint8_t, 4096> host_mask{};
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
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMallocHost(&control.data, capacity + 16));
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
    std::memcpy(
        host_mask.data(), static_cast<char*>(control.data) + 16, capacity);
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

  return Error::Ok;
}

Error run_attention(
    SlimTensor* output,
    SlimTensor* query,
    SlimTensor* key,
    SlimTensor* value,
    int64_t length) {
  const bool grouped_fp32 = query->dtype() == ScalarType::Float;
  const int64_t query_heads = grouped_fp32 ? 16 : 8;
  const int64_t kv_heads = grouped_fp32 ? 4 : 8;
  const int64_t group_size = query_heads / kv_heads;
  const int64_t capacity = key->size(2);
  auto device_guard = CUDAGuard::create(query->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(device_guard.error());
  auto stream_result = getCurrentCUDAStream(query->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(stream_result.error());
  const auto stream = stream_result.get();
  const size_t score_bytes = query_heads * length * query->itemsize();
  const std::array<size_t, 5> sizes = {
      score_bytes,
      score_bytes,
      grouped_fp32 ? 0 : score_bytes * 2,
      grouped_fp32 ? score_bytes : score_bytes * 2,
      grouped_fp32 ? 0 : score_bytes};
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
  const auto fp32 = dnn::Tensor::Type::FLOAT;
  const auto dtype = grouped_fp32 ? fp32 : dnn::Tensor::Type::BFLOAT16;
  ET_CHECK_OK_OR_RETURN_ERROR(set_tensor(
      query_tensor, query->data_ptr(), dtype, {kv_heads, group_size, 256}));
  ET_CHECK_OK_OR_RETURN_ERROR(
      set_tensor(key_tensor, key->data_ptr(), dtype, {kv_heads, length, 256}));
  ET_CHECK_OK_OR_RETURN_ERROR(set_tensor(
      value_tensor, value->data_ptr(), dtype, {kv_heads, length, 256}));
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(
      key_tensor.SetNdInfo({kv_heads, length, 256}, {capacity * 256, 256, 1})));
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(value_tensor.SetNdInfo(
      {kv_heads, length, 256}, {capacity * 256, 256, 1})));
  ET_CHECK_OK_OR_RETURN_ERROR(set_tensor(
      score_tensor, buffers[0], dtype, {kv_heads, group_size, length}));
  ET_CHECK_OK_OR_RETURN_ERROR(set_tensor(
      scaled_tensor, buffers[1], dtype, {kv_heads, group_size, length}));
  ET_CHECK_OK_OR_RETURN_ERROR(set_tensor(
      output_tensor, output->data_ptr(), dtype, {kv_heads, group_size, 256}));
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
  if (!grouped_fp32) {
    ET_CHECK_OK_OR_RETURN_ERROR(set_tensor(
        fp32_scores, buffers[2], fp32, {kv_heads, group_size, length}));
    ET_CHECK_OK_OR_RETURN_ERROR(
        run_unary(handle, fp32_scores, scaled_tensor, dnn::Unary::Mode::CAST));
  }
  ET_CHECK_OK_OR_RETURN_ERROR(set_tensor(
      fp32_scores,
      buffers[grouped_fp32 ? 1 : 2],
      fp32,
      {1, query_heads, 1, length}));
  ET_CHECK_OK_OR_RETURN_ERROR(set_tensor(
      fp32_probabilities, buffers[3], fp32, {1, query_heads, 1, length}));
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
  ET_CHECK_OK_OR_RETURN_ERROR(set_tensor(
      probabilities,
      buffers[grouped_fp32 ? 3 : 4],
      dtype,
      {kv_heads, group_size, length}));
  if (!grouped_fp32) {
    ET_CHECK_OK_OR_RETURN_ERROR(set_tensor(
        fp32_probabilities, buffers[3], fp32, {kv_heads, group_size, length}));
    ET_CHECK_OK_OR_RETURN_ERROR(run_unary(
        handle, probabilities, fp32_probabilities, dnn::Unary::Mode::CAST));
  }
  ET_CHECK_OK_OR_RETURN_ERROR(check_mudnn(product.SetTranspose(false, false)));
  return check_mudnn(
      product.Run(handle, output_tensor, probabilities, value_tensor));
}
}

extern "C" AOTI_SHIM_EXPORT Error aoti_torch_musa_causal_attention_out(
    SlimTensor* output,
    SlimTensor* query,
    SlimTensor* key,
    SlimTensor* value,
    SlimTensor* position,
    SlimTensor* mask) {
  ET_CHECK_OK_OR_RETURN_ERROR(validate_attention(output, query, key, value));
  ET_CHECK_OK_OR_RETURN_ERROR(validate_prefix_inputs(position, mask));
  ET_CHECK_OR_RETURN_ERROR(
      position->device_index() == query->device_index() &&
          mask->size(1) == key->size(2),
      InvalidArgument,
      "Causal prefix device and capacity must match the KV cache");
  const auto output_span = checked_musa_tensor_span(*output);
  for (auto* input : {position, mask}) {
    ET_CHECK_OR_RETURN_ERROR(
        output != input &&
            !musa_tensor_spans_overlap(
                output_span.get(), checked_musa_tensor_span(*input).get()),
        InvalidArgument,
        "Causal attention output must not overlap controls");
  }
  int64_t index = -1;
  ET_CHECK_OK_OR_RETURN_ERROR(read_prefix(position, mask, index));
  return run_attention(output, query, key, value, index + 1);
}

extern "C" AOTI_SHIM_EXPORT Error aoti_torch_musa_causal_prefix_out(
    SlimTensor* output,
    SlimTensor* position,
    SlimTensor* mask) {
  ET_CHECK_OK_OR_RETURN_ERROR(validate_host_prefix(output));
  ET_CHECK_OK_OR_RETURN_ERROR(validate_prefix_inputs(position, mask));
  const auto output_span = checked_musa_tensor_span(*output);
  for (auto* input : {position, mask}) {
    ET_CHECK_OR_RETURN_ERROR(
        !musa_tensor_spans_overlap(
            output_span.get(), checked_musa_tensor_span(*input).get()),
        InvalidArgument,
        "Causal prefix output must not overlap inputs");
  }
  int64_t index = -1;
  ET_CHECK_OK_OR_RETURN_ERROR(read_prefix(position, mask, index));
  auto* control = static_cast<int64_t*>(output->data_ptr());
  control[0] = index + 1;
  control[1] = mask->size(1);
  return Error::Ok;
}

extern "C" AOTI_SHIM_EXPORT Error aoti_torch_musa_prefix_attention_out(
    SlimTensor* output,
    SlimTensor* query,
    SlimTensor* key,
    SlimTensor* value,
    SlimTensor* control) {
  ET_CHECK_OK_OR_RETURN_ERROR(validate_attention(output, query, key, value));
  ET_CHECK_OK_OR_RETURN_ERROR(validate_host_prefix(control));
  const auto* prefix = static_cast<const int64_t*>(control->data_ptr());
  ET_CHECK_OR_RETURN_ERROR(
      prefix[1] == key->size(2) && prefix[0] > 0 && prefix[0] <= prefix[1],
      InvalidArgument,
      "Causal prefix length and capacity must match the KV cache");
  return run_attention(output, query, key, value, prefix[0]);
}

} // namespace executorch::backends::cuda
