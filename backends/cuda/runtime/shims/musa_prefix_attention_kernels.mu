#include <musa_runtime.h>
#include <cmath>
#include <cstdint>

__global__ void fused_prefix_attention(
    float* output,
    const float* query,
    const float* key,
    const float* value,
    int capacity,
    int length) {
  __shared__ float query_shared[256];
  __shared__ float scores[512];
  __shared__ float partials[512 / 32];
  const int thread = threadIdx.x;
  const int lane = thread % 32;
  const int warp = thread / 32;
  const int head = blockIdx.x;
  const int kv_head = head / 4;
  for (int channel = thread; channel < 256; channel += 512) {
    query_shared[channel] = query[head * 256 + channel];
  }
  __syncthreads();
  for (int position = warp; position < length; position += 512 / 32) {
    float dot = 0;
#pragma unroll
    for (int component = 0; component < 8; ++component) {
      const int channel = lane + component * 32;
      dot = fmaf(
          query_shared[channel],
          key[(kv_head * capacity + position) * 256 + channel],
          dot);
    }
    for (int stride = 16; stride > 0; stride /= 2) {
      dot += __shfl_down_sync(0xffffffff, dot, stride, 32);
    }
    if (lane == 0) {
      scores[position] = dot * 0.0625f;
    }
  }
  __syncthreads();
  float maximum = -INFINITY;
  for (int position = thread; position < length; position += 512) {
    maximum = fmaxf(maximum, scores[position]);
  }
  for (int stride = 16; stride > 0; stride /= 2) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, stride, 32));
  }
  if (lane == 0) {
    partials[warp] = maximum;
  }
  __syncthreads();
  maximum = -INFINITY;
#pragma unroll
  for (int subgroup = 0; subgroup < 512 / 32; ++subgroup) {
    maximum = fmaxf(maximum, partials[subgroup]);
  }
  __syncthreads();
  float denominator = 0;
  for (int position = thread; position < length; position += 512) {
    const float probability = expf(scores[position] - maximum);
    scores[position] = probability;
    denominator += probability;
  }
  for (int stride = 16; stride > 0; stride /= 2) {
    denominator += __shfl_down_sync(0xffffffff, denominator, stride, 32);
  }
  if (lane == 0) {
    partials[warp] = denominator;
  }
  __syncthreads();
  denominator = 0;
#pragma unroll
  for (int subgroup = 0; subgroup < 512 / 32; ++subgroup) {
    denominator += partials[subgroup];
  }
  for (int position = thread; position < length; position += 512) {
    scores[position] /= denominator;
  }
  __syncthreads();
  constexpr int lanes = 512 * 8 / 256;
  const int subgroup_lane = thread % lanes;
  const int channel_base = (thread / lanes) * 8;
  float accumulated[8] = {};
  for (int position = subgroup_lane; position < length; position += lanes) {
    const float probability = scores[position];
#pragma unroll
    for (int packet = 0; packet < 8 / 4; ++packet) {
      const float4 input = *reinterpret_cast<const float4*>(
          value + (kv_head * capacity + position) * 256 + channel_base +
          packet * 4);
      accumulated[packet * 4] =
          fmaf(probability, input.x, accumulated[packet * 4]);
      accumulated[packet * 4 + 1] =
          fmaf(probability, input.y, accumulated[packet * 4 + 1]);
      accumulated[packet * 4 + 2] =
          fmaf(probability, input.z, accumulated[packet * 4 + 2]);
      accumulated[packet * 4 + 3] =
          fmaf(probability, input.w, accumulated[packet * 4 + 3]);
    }
  }
#pragma unroll
  for (int channel = 0; channel < 8; ++channel) {
    for (int stride = lanes / 2; stride > 0; stride /= 2) {
      accumulated[channel] +=
          __shfl_down_sync(0xffffffff, accumulated[channel], stride, lanes);
    }
    if (subgroup_lane == 0) {
      output[head * 256 + channel_base + channel] = accumulated[channel];
    }
  }
}

extern "C" musaError_t executorch_musa_fused_prefix_attention_launch(
    float* output,
    const float* query,
    const float* key,
    const float* value,
    int capacity,
    int length,
    musaStream_t stream) {
  if (!output || !query || !key || !value || capacity < length || length < 1 ||
      length > 128 || capacity > 4096 ||
      reinterpret_cast<uintptr_t>(value) % 16) {
    return musaErrorInvalidValue;
  }
  fused_prefix_attention<<<16, 512, 0, stream>>>(
      output, query, key, value, capacity, length);
  return musaGetLastError();
}
