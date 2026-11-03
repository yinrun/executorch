#include <musa_runtime.h>

namespace {
__device__ __forceinline__ float sum_eight(const float (&values)[8]) {
  return ((values[0] + values[4]) + (values[2] + values[6])) +
      ((values[1] + values[5]) + (values[3] + values[7]));
}

__device__ __forceinline__ float memory_sum(
    const float (&values)[32][32],
    int column) {
  float groups[8];
#pragma unroll
  for (int group = 0; group < 8; ++group) {
    const int row = group * 4;
    groups[group] = (values[row][column] + values[row + 2][column]) +
        (values[row + 1][column] + values[row + 3][column]);
  }
  return sum_eight(groups);
}

__device__ __forceinline__ float output_sum(
    const float (&values)[32][32],
    int column) {
  float groups[4];
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    float partial[8];
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      partial[index] = values[group * 8 + index][column];
    }
    groups[group] = sum_eight(partial);
  }
  return (groups[0] + groups[2]) + (groups[1] + groups[3]);
}

__global__ void recurrence_kernel(
    float* output,
    float* next_state,
    const float* query,
    const float* key,
    const float* value,
    const float* decay,
    const float* beta,
    const float* state) {
  const int value_lane = threadIdx.x;
  const int key_part = threadIdx.y;
  const int head = blockIdx.y;
  const int column = blockIdx.x * 32 + value_lane;
  const int base = head * 128 * 128 + column;
  const float decay_value = decay[head];
  float states[8];
  float products[8];
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    const int row = key_part + index * 16;
    states[index] = state[base + row * 128] * decay_value;
    products[index] = states[index] * key[head * 128 + row];
  }
  __shared__ float sums[32][32];
  sums[key_part][value_lane] =
      ((products[0] + products[2]) + products[4]) + products[6];
  sums[key_part + 16][value_lane] =
      ((products[1] + products[3]) + products[5]) + products[7];
  __syncthreads();
  const float total = memory_sum(sums, value_lane);
  const float delta = (value[head * 128 + column] - total) * beta[head];
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    const int row = key_part + index * 16;
    const float updated = states[index] + key[head * 128 + row] * delta;
    next_state[base + row * 128] = updated;
    products[index] = updated * query[head * 128 + row];
  }
  __syncthreads();
  sums[key_part][value_lane] =
      ((products[0] + products[2]) + products[4]) + products[6];
  sums[key_part + 16][value_lane] =
      ((products[1] + products[3]) + products[5]) + products[7];
  __syncthreads();
  if (key_part == 0) {
    output[head * 128 + column] = output_sum(sums, value_lane);
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
  recurrence_kernel<<<dim3(4, heads), dim3(32, 16), 0, stream>>>(
      output, next_state, query, key, value, decay, beta, state);
  return musaGetLastError();
}
