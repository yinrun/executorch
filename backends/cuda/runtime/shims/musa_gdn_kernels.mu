#include <musa_runtime.h>

namespace {
__global__ void recurrence_kernel(
    float* output,
    float* next_state,
    const float* query,
    const float* key,
    const float* value,
    const float* decay,
    const float* beta,
    const float* state) {
  constexpr int key_values = 128 / 16;
  const int value_lane = threadIdx.x;
  const int key_part = threadIdx.y;
  const int head = blockIdx.y;
  const int column = blockIdx.x * 32 + value_lane;
  const int base = head * 128 * 128 + column;
  const float decay_value = decay[head];
  float states[key_values];
  float partial = 0;
#pragma unroll
  for (int index = 0; index < key_values; ++index) {
    const int row = key_part + index * 16;
    states[index] = state[base + row * 128] * decay_value;
    partial += states[index] * key[head * 128 + row];
  }
  __shared__ float sums[16][32];
  sums[key_part][value_lane] = partial;
  __syncthreads();
  float total = 0;
#pragma unroll
  for (int part = 0; part < 16; ++part) {
    total += sums[part][value_lane];
  }
  const float delta = (value[head * 128 + column] - total) * beta[head];
  float result = 0;
#pragma unroll
  for (int index = 0; index < key_values; ++index) {
    const int row = key_part + index * 16;
    const float updated = states[index] + key[head * 128 + row] * delta;
    next_state[base + row * 128] = updated;
    result += updated * query[head * 128 + row];
  }
  __syncthreads();
  sums[key_part][value_lane] = result;
  __syncthreads();
  if (key_part == 0) {
    float final_value = 0;
#pragma unroll
    for (int part = 0; part < 16; ++part) {
      final_value += sums[part][value_lane];
    }
    output[head * 128 + column] = final_value;
  }
}
}

extern "C" musaError_t executorch_musa_gdn_launch(
    float* output,
    float* next_state,
    const float* query,
    const float* key,
    const float* value,
    const float* decay,
    const float* beta,
    const float* state,
    int heads,
    musaStream_t stream) {
  if (heads != 16 && heads != 32) {
    return musaErrorInvalidValue;
  }
  recurrence_kernel<<<dim3(128 / 32, heads), dim3(32, 16), 0, stream>>>(
      output, next_state, query, key, value, decay, beta, state);
  return musaGetLastError();
}
