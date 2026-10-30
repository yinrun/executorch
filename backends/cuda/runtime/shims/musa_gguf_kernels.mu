#include <musa_runtime.h>
#include <musa_fp16.h>

#include <cstdint>
#include <limits>

namespace {

__device__ float scale_value(const uint8_t* bytes) {
  const auto bits = static_cast<unsigned short>(bytes[0] | (bytes[1] << 8));
  return __half2float(__ushort_as_half(bits));
}

template <int QuantType>
__device__ float decode_weight(const uint8_t* block, int element) {
  if constexpr (QuantType == 14) {
    const int half = element / 128;
    const int quarter = (element % 128) / 32;
    const int lane = element % 32;
    const int low =
        (block[half * 64 + (quarter % 2) * 32 + lane] >> ((quarter / 2) * 4)) &
        15;
    const int high = (block[128 + half * 32 + lane] >> (quarter * 2)) & 3;
    const int value = (low | (high << 4)) - 32;
    const int scale = static_cast<int8_t>(block[192 + element / 16]);
    return (scale_value(block + 208) * static_cast<float>(scale)) *
           static_cast<float>(value);
  } else {
    const int group = element / 32;
    const int lane = element % 32;
    const auto* scales = block + 4;
    const int scale =
        group < 4 ? scales[group] & 63
                  : (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
    const int minimum =
        group < 4 ? scales[group + 4] & 63
                  : (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    constexpr int offset = QuantType == 12 ? 16 : 48;
    int value =
        (block[offset + (group / 2) * 32 + lane] >> ((group % 2) * 4)) & 15;
    if constexpr (QuantType == 13) {
      value |= ((block[16 + lane] >> group) & 1) << 4;
    }
    return (scale_value(block) * static_cast<float>(scale)) *
               static_cast<float>(value) -
           scale_value(block + 2) * static_cast<float>(minimum);
  }
}

template <int QuantType>
__global__ void linear_kernel(float* output, const float* input,
                              const uint8_t* packed, int64_t rows,
                              int64_t columns) {
  constexpr int bytes = QuantType == 12 ? 144 : QuantType == 13 ? 176 : 210;
  const int64_t output_row = blockIdx.x % rows;
  const int64_t batch = blockIdx.x / rows;
  const auto* weight_row = packed + output_row * (columns / 256) * bytes;
  float accumulator = 0;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    const auto value = decode_weight<QuantType>(
        weight_row + (column / 256) * bytes, column % 256);
    accumulator = fmaf(value, input[batch * columns + column], accumulator);
  }
  __shared__ float partial[128];
  partial[threadIdx.x] = accumulator;
  __syncthreads();
  for (int stride = 64; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[batch * rows + output_row] = partial[0];
  }
}

template <int QuantType>
__global__ void linear_decode_kernel(float* output, const float* input,
                                    const uint8_t* packed, int rows, int columns) {
  constexpr int bytes = QuantType == 12 ? 144 : QuantType == 13 ? 176 : 210;
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * 4 + threadIdx.x / 32;
  if (row >= rows) {
    return;
  }
  const auto* weight_row =
      packed + static_cast<int64_t>(row) * (columns / 256) * bytes;
  float partial[4] = {};
  for (int block = 0; block < columns / 256; ++block) {
    const auto* weight_block = weight_row + static_cast<int64_t>(block) * bytes;
    #pragma unroll
    for (int group = 0; group < 4; ++group) {
      const int element = lane + group * 32;
      const float first = decode_weight<QuantType>(weight_block, element);
      const float second = decode_weight<QuantType>(weight_block, element + 128);
      partial[group] = fmaf(first, input[block * 256 + element], partial[group]);
      partial[group] = fmaf(second, input[block * 256 + element + 128], partial[group]);
    }
  }
  float accumulator = (partial[0] + partial[2]) + (partial[1] + partial[3]);
  for (int stride = 16; stride > 0; stride /= 2) {
    accumulator += __shfl_down_sync(0xffffffff, accumulator, stride, 32);
  }
  if (lane == 0) {
    output[row] = accumulator;
  }
}

template <int QuantType>
__global__ void embedding_kernel(float* output, const int64_t* indices,
                                 const uint8_t* packed, int* invalid,
                                 int64_t rows, int64_t columns, int64_t count) {
  constexpr int bytes = QuantType == 12 ? 144 : QuantType == 13 ? 176 : 210;
  const int64_t position =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (position >= count * columns) {
    return;
  }
  const int64_t row = indices[position / columns];
  const int64_t column = position % columns;
  if (row < 0 || row >= rows) {
    if (column == 0) {
      atomicExch(invalid, 1);
    }
    output[position] = 0;
    return;
  }
  output[position] = decode_weight<QuantType>(
      packed + (row * (columns / 256) + column / 256) * bytes, column % 256);
}

}

extern "C" musaError_t executorch_musa_gguf_linear_launch(
    float* output, const float* input, const uint8_t* packed, int64_t rows,
    int64_t columns, int64_t batches, int64_t quant_type, musaStream_t stream) {
  const auto grid = static_cast<unsigned int>(rows * batches);
  if (batches == 1 && columns <= std::numeric_limits<int>::max()) {
    const auto decode_grid = static_cast<unsigned int>((rows + 3) / 4);
    if (quant_type == 12) {
      linear_decode_kernel<12><<<decode_grid, 128, 0, stream>>>(
          output, input, packed, static_cast<int>(rows), static_cast<int>(columns));
    } else if (quant_type == 13) {
      linear_decode_kernel<13><<<decode_grid, 128, 0, stream>>>(
          output, input, packed, static_cast<int>(rows), static_cast<int>(columns));
    } else if (quant_type == 14) {
      linear_decode_kernel<14><<<decode_grid, 128, 0, stream>>>(
          output, input, packed, static_cast<int>(rows), static_cast<int>(columns));
    } else {
      return musaErrorInvalidValue;
    }
    return musaGetLastError();
  }
  if (quant_type == 12) {
    linear_kernel<12>
        <<<grid, 128, 0, stream>>>(output, input, packed, rows, columns);
  } else if (quant_type == 13) {
    linear_kernel<13>
        <<<grid, 128, 0, stream>>>(output, input, packed, rows, columns);
  } else if (quant_type == 14) {
    linear_kernel<14>
        <<<grid, 128, 0, stream>>>(output, input, packed, rows, columns);
  } else {
    return musaErrorInvalidValue;
  }
  return musaGetLastError();
}

extern "C" musaError_t executorch_musa_gguf_embedding_launch(
    float* output, const int64_t* indices, const uint8_t* packed, int* invalid,
    int64_t rows, int64_t columns, int64_t count, int64_t quant_type,
    musaStream_t stream) {
  const auto grid = static_cast<unsigned int>((count * columns + 255) / 256);
  if (quant_type == 12) {
    embedding_kernel<12><<<grid, 256, 0, stream>>>(
        output, indices, packed, invalid, rows, columns, count);
  } else if (quant_type == 13) {
    embedding_kernel<13><<<grid, 256, 0, stream>>>(
        output, indices, packed, invalid, rows, columns, count);
  } else if (quant_type == 14) {
    embedding_kernel<14><<<grid, 256, 0, stream>>>(
        output, indices, packed, invalid, rows, columns, count);
  } else {
    return musaErrorInvalidValue;
  }
  return musaGetLastError();
}
