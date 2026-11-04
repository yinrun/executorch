#include <musa_runtime.h>
#include <musa_fp16.h>

#include <cstdint>
#include <limits>

namespace {

#if defined(EXECUTORCH_MUSA_GGUF_BYTE_CONVERT) ||    \
    defined(EXECUTORCH_MUSA_GGUF_QUAD_ARITHMETIC) || \
    defined(EXECUTORCH_MUSA_GGUF_GROUPED_Q6_HEAD)
using UnsignedByteQuad = unsigned char __attribute__((ext_vector_type(4)));
using FloatQuad = float __attribute__((ext_vector_type(4)));
#endif

#if defined(EXECUTORCH_MUSA_GGUF_QUAD_ARITHMETIC) || \
    defined(EXECUTORCH_MUSA_GGUF_GROUPED_Q6_HEAD)
using SignedByteQuad = signed char __attribute__((ext_vector_type(4)));

__device__ __forceinline__ FloatQuad
fma_decode_quad(FloatQuad first, FloatQuad second, FloatQuad accum) {
  return {
      fmaf(first[0], second[0], accum[0]),
      fmaf(first[1], second[1], accum[1]),
      fmaf(first[2], second[2], accum[2]),
      fmaf(first[3], second[3], accum[3])};
}
#endif

__device__ float scale_value(const uint8_t* bytes) {
  const auto bits = static_cast<unsigned short>(bytes[0] | (bytes[1] << 8));
  return __half2float(__ushort_as_half(bits));
}

#if defined(EXECUTORCH_MUSA_GGUF_GROUPED_Q6_HEAD)
__global__ void grouped_predecoded_q6_head_kernel(
    float* output,
    const float* input,
    const uint8_t* packed) {
  constexpr int columns = 2560;
  constexpr int rows_per_block = 8;
  constexpr int row_tiles = 4;
  const int group = threadIdx.x % 16;
  const int row = blockIdx.x * rows_per_block * row_tiles + threadIdx.x / 16;
  float accumulators[row_tiles] = {};
  for (int block = 0; block < columns / 256; ++block) {
    FloatQuad inputs[4];
#pragma unroll
    for (int word = 0; word < 4; ++word) {
      const float4 values = __ldg(reinterpret_cast<const float4*>(
          input + block * 256 + group * 16 + word * 4));
      inputs[word] = FloatQuad{values.x, values.y, values.z, values.w};
    }
#pragma unroll
    for (int tile = 0; tile < row_tiles; ++tile) {
      const auto* weight = packed +
          (int64_t(row + tile * rows_per_block) * (columns / 256) + block) *
              320;
      const uint4 packed_values =
          __ldg(reinterpret_cast<const uint4*>(weight + group * 16));
      const uint32_t words[4] = {
          packed_values.x, packed_values.y, packed_values.z, packed_values.w};
      const float scale =
          __ldg(reinterpret_cast<const float*>(weight + 256) + group);
      FloatQuad partial = {};
#pragma unroll
      for (int word = 0; word < 4; ++word) {
        const FloatQuad values = __builtin_convertvector(
            __builtin_bit_cast(SignedByteQuad, words[word]), FloatQuad);
        partial = fma_decode_quad(values, inputs[word], partial);
      }
      const float sum = (partial[0] + partial[2]) + (partial[1] + partial[3]);
      accumulators[tile] = fmaf(scale, sum, accumulators[tile]);
    }
  }
#pragma unroll
  for (int tile = 0; tile < row_tiles; ++tile) {
    for (int distance = 8; distance > 0; distance /= 2) {
      accumulators[tile] +=
          __shfl_down_sync(0xffffffff, accumulators[tile], distance, 16);
    }
    if (group == 0) {
      output[row + tile * rows_per_block] = accumulators[tile];
    }
  }
}
#endif

template <int QuantType>
__device__ float decode_weight(const uint8_t* block, int element) {
  if constexpr (QuantType == 114) {
    const auto* scale = block + 256 + (element / 16) * 4;
    const uint32_t bits = static_cast<uint32_t>(scale[0]) |
        (static_cast<uint32_t>(scale[1]) << 8) |
        (static_cast<uint32_t>(scale[2]) << 16) |
        (static_cast<uint32_t>(scale[3]) << 24);
    return __uint_as_float(bits) *
        static_cast<float>(static_cast<int8_t>(block[element]));
  } else if constexpr (QuantType == 14) {
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
  constexpr int bytes = QuantType == 114 ? 320
      : QuantType == 12                  ? 144
      : QuantType == 13                  ? 176
                                         : 210;
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

template <int QuantType, typename Value>
__device__ Value load_decode_value(const Value* address) {
  if constexpr (QuantType == 12) {
    return __ldg(address);
  } else {
    return *address;
  }
}

template <int QuantType>
__device__ uint32_t load_packed_four(const uint8_t* address) {
  if constexpr (QuantType == 14) {
    const auto* halves = reinterpret_cast<const uint16_t*>(address);
    return static_cast<uint32_t>(halves[0]) |
        (static_cast<uint32_t>(halves[1]) << 16);
  } else {
    return load_decode_value<QuantType>(
        reinterpret_cast<const uint32_t*>(address));
  }
}

template <int QuantType, int Width, int Parts = 1>
__global__ void linear_vector_decode_kernel(
    float* output,
    const float* input,
    const uint8_t* packed,
    int rows,
    int columns) {
  constexpr int bytes = QuantType == 12 ? 144 : QuantType == 13 ? 176 : 210;
  static_assert(Width == 4 || Width == 16);
  static_assert(QuantType != 14 || Width == 4);
  constexpr int width = Width;
  constexpr int lanes = 128 / width;
  const int part = Parts == 1 ? 0 : blockIdx.y;
  const int lane = threadIdx.x % lanes;
  const int row = blockIdx.x * width + threadIdx.x / lanes;
  if (row >= rows) {
    return;
  }
  const int element = lane * width;
  const int group = element / 32;
  const int channel = element % 32;
  const auto* weight_row =
      packed + static_cast<int64_t>(row) * (columns / 256) * bytes;
  float partial[width] = {};
  for (int block = (columns / 256) * part / Parts;
       block < (columns / 256) * (part + 1) / Parts;
       ++block) {
    const auto* weight_block = weight_row + static_cast<int64_t>(block) * bytes;
    float first_inputs[width];
    float second_inputs[width];
#pragma unroll
    for (int word = 0; word < width / 4; ++word) {
      const float4 first_input =
          load_decode_value<QuantType>(reinterpret_cast<const float4*>(
              input + block * 256 + element + word * 4));
      const float4 second_input =
          load_decode_value<QuantType>(reinterpret_cast<const float4*>(
              input + block * 256 + element + 128 + word * 4));
      first_inputs[word * 4] = first_input.x;
      first_inputs[word * 4 + 1] = first_input.y;
      first_inputs[word * 4 + 2] = first_input.z;
      first_inputs[word * 4 + 3] = first_input.w;
      second_inputs[word * 4] = second_input.x;
      second_inputs[word * 4 + 1] = second_input.y;
      second_inputs[word * 4 + 2] = second_input.z;
      second_inputs[word * 4 + 3] = second_input.w;
    }
    if constexpr (QuantType == 14) {
      const float scale = scale_value(weight_block + 208);
      const float first_scale = scale *
          static_cast<float>(static_cast<int8_t>(
              weight_block[192 + element / 16]));
      const float second_scale = scale *
          static_cast<float>(static_cast<int8_t>(
              weight_block[200 + element / 16]));
#pragma unroll
      for (int component = 0; component < width; ++component) {
        const uint32_t low_first = load_packed_four<14>(
            weight_block + (group % 2) * 32 + channel + (component / 4) * 4);
        const uint32_t low_second = load_packed_four<14>(
            weight_block + 64 + (group % 2) * 32 + channel +
            (component / 4) * 4);
        const uint32_t high_first = load_packed_four<14>(
            weight_block + 128 + channel + (component / 4) * 4);
        const uint32_t high_second = load_packed_four<14>(
            weight_block + 160 + channel + (component / 4) * 4);
        const uint32_t first_levels = __vsub4(
            ((low_first >> ((group / 2) * 4)) & 0x0f0f0f0fU) |
                (((high_first >> (group * 2)) & 0x03030303U) << 4),
            0x20202020U);
        const uint32_t second_levels = __vsub4(
            ((low_second >> ((group / 2) * 4)) & 0x0f0f0f0fU) |
                (((high_second >> (group * 2)) & 0x03030303U) << 4),
            0x20202020U);
        const int first_value =
            static_cast<int8_t>(first_levels >> ((component % 4) * 8));
        const int second_value =
            static_cast<int8_t>(second_levels >> ((component % 4) * 8));
        partial[component] = fmaf(
            first_scale * static_cast<float>(first_value),
            first_inputs[component],
            partial[component]);
        partial[component] = fmaf(
            second_scale * static_cast<float>(second_value),
            second_inputs[component],
            partial[component]);
      }
    } else {
      constexpr int offset = QuantType == 12 ? 16 : 48;
      float first_scale;
      float first_minimum;
      float second_scale;
      float second_minimum;
      if constexpr (width == 16) {
        const uint4 metadata = load_decode_value<QuantType>(
            reinterpret_cast<const uint4*>(weight_block));
        const float scale = __half2float(__ushort_as_half(metadata.x & 65535));
        const float minimum = __half2float(__ushort_as_half(metadata.x >> 16));
        const uint32_t low_scale = (metadata.y >> (group * 8)) & 255;
        const uint32_t low_minimum = (metadata.z >> (group * 8)) & 255;
        const uint32_t high = (metadata.w >> (group * 8)) & 255;
        first_scale = scale * static_cast<float>(low_scale & 63);
        first_minimum = minimum * static_cast<float>(low_minimum & 63);
        second_scale =
            scale * static_cast<float>((high & 15) | ((low_scale >> 6) << 4));
        second_minimum = minimum *
            static_cast<float>((high >> 4) | ((low_minimum >> 6) << 4));
      } else {
        const auto* scales = weight_block + 4;
        const float scale = scale_value(weight_block);
        const float minimum = scale_value(weight_block + 2);
        first_scale = scale * static_cast<float>(scales[group] & 63);
        first_minimum = minimum * static_cast<float>(scales[group + 4] & 63);
        second_scale =
            scale *
            static_cast<float>(
                (scales[group + 8] & 15) | ((scales[group] >> 6) << 4));
        second_minimum =
            minimum *
            static_cast<float>(
                (scales[group + 8] >> 4) | ((scales[group + 4] >> 6) << 4));
      }
      uint32_t first_low[width / 4];
      uint32_t second_low[width / 4];
      uint32_t high_words[width / 4] = {};
      const auto* first_address =
          weight_block + offset + (group / 2) * 32 + channel;
      const auto* second_address = first_address + 64;
      if constexpr (width == 16) {
        const uint4 first_words = load_decode_value<QuantType>(
            reinterpret_cast<const uint4*>(first_address));
        const uint4 second_words = load_decode_value<QuantType>(
            reinterpret_cast<const uint4*>(second_address));
        first_low[0] = first_words.x;
        first_low[1] = first_words.y;
        first_low[2] = first_words.z;
        first_low[3] = first_words.w;
        second_low[0] = second_words.x;
        second_low[1] = second_words.y;
        second_low[2] = second_words.z;
        second_low[3] = second_words.w;
      } else {
        first_low[0] = load_packed_four<QuantType>(first_address);
        second_low[0] = load_packed_four<QuantType>(second_address);
      }
      if constexpr (QuantType == 13) {
        if constexpr (width == 16) {
          const uint4 words = load_decode_value<QuantType>(
              reinterpret_cast<const uint4*>(weight_block + 16 + channel));
          high_words[0] = words.x;
          high_words[1] = words.y;
          high_words[2] = words.z;
          high_words[3] = words.w;
        } else {
          high_words[0] = load_packed_four<13>(weight_block + 16 + channel);
        }
      }
#if defined(EXECUTORCH_MUSA_GGUF_QUAD_ARITHMETIC)
#pragma unroll
      for (int word = 0; word < width / 4; ++word) {
        const uint32_t low_first = first_low[word];
        const uint32_t low_second = second_low[word];
        const uint32_t high = high_words[word];
        uint32_t first_levels = (low_first >> ((group % 2) * 4)) & 0x0f0f0f0fU;
        uint32_t second_levels =
            (low_second >> ((group % 2) * 4)) & 0x0f0f0f0fU;
        if constexpr (QuantType == 13) {
          first_levels |= ((high >> group) & 0x01010101U) << 4;
          second_levels |= ((high >> (group + 4)) & 0x01010101U) << 4;
        }
        const FloatQuad first_values = __builtin_convertvector(
            __builtin_bit_cast(UnsignedByteQuad, first_levels), FloatQuad);
        const FloatQuad second_values = __builtin_convertvector(
            __builtin_bit_cast(UnsignedByteQuad, second_levels), FloatQuad);
        const FloatQuad first_weights =
            first_scale * first_values - first_minimum;
        const FloatQuad second_weights =
            second_scale * second_values - second_minimum;
        const int component = word * 4;
        const FloatQuad first_input = {
            first_inputs[component],
            first_inputs[component + 1],
            first_inputs[component + 2],
            first_inputs[component + 3]};
        const FloatQuad second_input = {
            second_inputs[component],
            second_inputs[component + 1],
            second_inputs[component + 2],
            second_inputs[component + 3]};
        FloatQuad accum = {
            partial[component],
            partial[component + 1],
            partial[component + 2],
            partial[component + 3]};
        accum = fma_decode_quad(first_weights, first_input, accum);
        accum = fma_decode_quad(second_weights, second_input, accum);
#pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
          partial[component + byte] = accum[byte];
        }
      }
#else
#pragma unroll
      for (int component = 0; component < width; ++component) {
        const uint32_t low_first = first_low[component / 4];
        const uint32_t low_second = second_low[component / 4];
        const uint32_t high = high_words[component / 4];
        uint32_t first_levels = (low_first >> ((group % 2) * 4)) & 0x0f0f0f0fU;
        uint32_t second_levels =
            (low_second >> ((group % 2) * 4)) & 0x0f0f0f0fU;
        if constexpr (QuantType == 13) {
          first_levels |= ((high >> group) & 0x01010101U) << 4;
          second_levels |= ((high >> (group + 4)) & 0x01010101U) << 4;
        }
#if defined(EXECUTORCH_MUSA_GGUF_BYTE_CONVERT)
        const FloatQuad first_values = __builtin_convertvector(
            __builtin_bit_cast(UnsignedByteQuad, first_levels), FloatQuad);
        const FloatQuad second_values = __builtin_convertvector(
            __builtin_bit_cast(UnsignedByteQuad, second_levels), FloatQuad);
        const float first_value = first_values[component % 4];
        const float second_value = second_values[component % 4];
#else
        const int first_value = (first_levels >> ((component % 4) * 8)) & 255;
        const int second_value = (second_levels >> ((component % 4) * 8)) & 255;
#endif
        const float first_weight =
            first_scale * static_cast<float>(first_value) - first_minimum;
        const float second_weight =
            second_scale * static_cast<float>(second_value) - second_minimum;
        partial[component] =
            fmaf(first_weight, first_inputs[component], partial[component]);
        partial[component] =
            fmaf(second_weight, second_inputs[component], partial[component]);
      }
#endif
    }
  }
#pragma unroll
  for (int component = 0; component < width; ++component) {
    for (int stride = lanes / 2; stride > 0; stride /= 2) {
      partial[component] +=
          __shfl_down_sync(0xffffffff, partial[component], stride, lanes);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int stride = width / 2; stride > 0; stride /= 2) {
#pragma unroll
      for (int component = 0; component < stride; ++component) {
        partial[component] += partial[component + stride];
      }
    }
    output[part * rows + row] = partial[0];
  }
}

__global__ void linear_q6_tiled_decode_kernel(
    float* output,
    const float* input,
    const uint8_t* packed,
    int rows,
    int columns) {
  constexpr int bytes = 210;
  constexpr int width = 4;
  constexpr int RowsPerThread = 2;
  constexpr int lanes = 128 / width;
  const int lane = threadIdx.x % lanes;
  const int row = blockIdx.x * (width * RowsPerThread) + threadIdx.x / lanes;
  if (row >= rows) {
    return;
  }
  const int element = lane * width;
  const int group = element / 32;
  const int channel = element % 32;
  float partial[RowsPerThread][width] = {};
  for (int block = 0; block < columns / 256; ++block) {
    float first_inputs[width];
    float second_inputs[width];
#pragma unroll
    for (int word = 0; word < width / 4; ++word) {
      const float4 first_input =
          load_decode_value<14>(reinterpret_cast<const float4*>(
              input + block * 256 + element + word * 4));
      const float4 second_input =
          load_decode_value<14>(reinterpret_cast<const float4*>(
              input + block * 256 + element + 128 + word * 4));
      first_inputs[word * 4] = first_input.x;
      first_inputs[word * 4 + 1] = first_input.y;
      first_inputs[word * 4 + 2] = first_input.z;
      first_inputs[word * 4 + 3] = first_input.w;
      second_inputs[word * 4] = second_input.x;
      second_inputs[word * 4 + 1] = second_input.y;
      second_inputs[word * 4 + 2] = second_input.z;
      second_inputs[word * 4 + 3] = second_input.w;
    }
#pragma unroll
    for (int tile = 0; tile < RowsPerThread; ++tile) {
      if (row + tile * width < rows) {
        const auto* weight_row = packed +
            static_cast<int64_t>(row + tile * width) * (columns / 256) * bytes;
        const auto* weight_block =
            weight_row + static_cast<int64_t>(block) * bytes;

        const float scale = scale_value(weight_block + 208);
        const float first_scale = scale *
            static_cast<float>(static_cast<int8_t>(
                weight_block[192 + element / 16]));
        const float second_scale = scale *
            static_cast<float>(static_cast<int8_t>(
                weight_block[200 + element / 16]));
#pragma unroll
        for (int component = 0; component < width; ++component) {
          const uint32_t low_first = load_packed_four<14>(
              weight_block + (group % 2) * 32 + channel + (component / 4) * 4);
          const uint32_t low_second = load_packed_four<14>(
              weight_block + 64 + (group % 2) * 32 + channel +
              (component / 4) * 4);
          const uint32_t high_first = load_packed_four<14>(
              weight_block + 128 + channel + (component / 4) * 4);
          const uint32_t high_second = load_packed_four<14>(
              weight_block + 160 + channel + (component / 4) * 4);
          const uint32_t first_levels = __vsub4(
              ((low_first >> ((group / 2) * 4)) & 0x0f0f0f0fU) |
                  (((high_first >> (group * 2)) & 0x03030303U) << 4),
              0x20202020U);
          const uint32_t second_levels = __vsub4(
              ((low_second >> ((group / 2) * 4)) & 0x0f0f0f0fU) |
                  (((high_second >> (group * 2)) & 0x03030303U) << 4),
              0x20202020U);
          const int first_value =
              static_cast<int8_t>(first_levels >> ((component % 4) * 8));
          const int second_value =
              static_cast<int8_t>(second_levels >> ((component % 4) * 8));
          partial[tile][component] = fmaf(
              first_scale * static_cast<float>(first_value),
              first_inputs[component],
              partial[tile][component]);
          partial[tile][component] = fmaf(
              second_scale * static_cast<float>(second_value),
              second_inputs[component],
              partial[tile][component]);
        }
      }
    }
  }
#pragma unroll
  for (int tile = 0; tile < RowsPerThread; ++tile) {
#pragma unroll
    for (int component = 0; component < width; ++component) {
      for (int stride = lanes / 2; stride > 0; stride /= 2) {
        partial[tile][component] += __shfl_down_sync(
            0xffffffff, partial[tile][component], stride, lanes);
      }
    }
    if (lane == 0 && row + tile * width < rows) {
      partial[tile][0] += partial[tile][2];
      partial[tile][1] += partial[tile][3];
      partial[tile][0] += partial[tile][1];
      output[row + tile * width] = partial[tile][0];
    }
  }
}

template <int Tile, int Parts = 1>
__global__ void linear_predecoded_q6_decode_kernel(
    float* output,
    const float* input,
    const uint8_t* packed,
    int rows,
    int columns) {
  constexpr int Width = 4;
  constexpr int bytes = 320;
  constexpr int lanes = 128 / Width;
  const int part = Parts == 1 ? 0 : blockIdx.y;
  const int lane = threadIdx.x % lanes;
  const int row = blockIdx.x * Width * Tile + threadIdx.x / lanes;
  if (row >= rows) {
    return;
  }
  const int element = lane * Width;
  float partial[Tile][Width] = {};
  for (int block = (columns / 256) * part / Parts;
       block < (columns / 256) * (part + 1) / Parts;
       ++block) {
    float first_inputs[Width];
    float second_inputs[Width];
#pragma unroll
    for (int word = 0; word < Width / 4; ++word) {
      const float4 first = __ldg(reinterpret_cast<const float4*>(
          input + block * 256 + element + word * 4));
      const float4 second = __ldg(reinterpret_cast<const float4*>(
          input + block * 256 + element + 128 + word * 4));
      first_inputs[word * 4] = first.x;
      first_inputs[word * 4 + 1] = first.y;
      first_inputs[word * 4 + 2] = first.z;
      first_inputs[word * 4 + 3] = first.w;
      second_inputs[word * 4] = second.x;
      second_inputs[word * 4 + 1] = second.y;
      second_inputs[word * 4 + 2] = second.z;
      second_inputs[word * 4 + 3] = second.w;
    }
#pragma unroll
    for (int tile = 0; tile < Tile; ++tile) {
      if (row + tile * Width < rows) {
        const auto* weight = packed +
            (int64_t(row + tile * Width) * (columns / 256) + block) * bytes;
        const float first_scale =
            __ldg(reinterpret_cast<const float*>(weight + 256) + element / 16);
        const float second_scale = __ldg(
            reinterpret_cast<const float*>(weight + 256) + element / 16 + 8);
#pragma unroll
        for (int word = 0; word < Width / 4; ++word) {
          const uint32_t first = __ldg(
              reinterpret_cast<const uint32_t*>(weight + element + word * 4));
          const uint32_t second = __ldg(reinterpret_cast<const uint32_t*>(
              weight + element + 128 + word * 4));
#if defined(EXECUTORCH_MUSA_GGUF_QUAD_ARITHMETIC)
          const FloatQuad first_values = __builtin_convertvector(
              __builtin_bit_cast(SignedByteQuad, first), FloatQuad);
          const FloatQuad second_values = __builtin_convertvector(
              __builtin_bit_cast(SignedByteQuad, second), FloatQuad);
          const int component = word * 4;
          const FloatQuad first_input = {
              first_inputs[component],
              first_inputs[component + 1],
              first_inputs[component + 2],
              first_inputs[component + 3]};
          const FloatQuad second_input = {
              second_inputs[component],
              second_inputs[component + 1],
              second_inputs[component + 2],
              second_inputs[component + 3]};
          FloatQuad accum = {
              partial[tile][component],
              partial[tile][component + 1],
              partial[tile][component + 2],
              partial[tile][component + 3]};
          accum =
              fma_decode_quad(first_scale * first_values, first_input, accum);
          accum = fma_decode_quad(
              second_scale * second_values, second_input, accum);
#pragma unroll
          for (int byte = 0; byte < 4; ++byte) {
            partial[tile][component + byte] = accum[byte];
          }
#else
#pragma unroll
          for (int byte = 0; byte < 4; ++byte) {
            const int component = word * 4 + byte;
            const int first_value = static_cast<int8_t>(first >> (byte * 8));
            const int second_value = static_cast<int8_t>(second >> (byte * 8));
            const float first_weight =
                first_scale * static_cast<float>(first_value);
            const float second_weight =
                second_scale * static_cast<float>(second_value);
            partial[tile][component] = fmaf(
                first_weight,
                first_inputs[component],
                partial[tile][component]);
            partial[tile][component] = fmaf(
                second_weight,
                second_inputs[component],
                partial[tile][component]);
          }
#endif
        }
      }
    }
  }
#pragma unroll
  for (int tile = 0; tile < Tile; ++tile) {
#pragma unroll
    for (int component = 0; component < Width; ++component) {
      for (int stride = lanes / 2; stride > 0; stride /= 2) {
        partial[tile][component] += __shfl_down_sync(
            0xffffffff, partial[tile][component], stride, lanes);
      }
    }
    if (lane == 0 && row + tile * Width < rows) {
#pragma unroll
      for (int stride = Width / 2; stride > 0; stride /= 2) {
#pragma unroll
        for (int component = 0; component < stride; ++component) {
          partial[tile][component] += partial[tile][component + stride];
        }
      }
      output[part * rows + row + tile * Width] = partial[tile][0];
    }
  }
}

#if defined(EXECUTORCH_MUSA_GGUF_SPLIT_K)
__global__ void
reduce_split_k_kernel(float* output, const float* workspace, int rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    float result = workspace[row];
#pragma unroll
    for (int part = 1; part < 4; ++part) {
      result += workspace[part * rows + row];
    }
    output[row] = result;
  }
}
#endif

template <int QuantType>
musaError_t launch_linear_vector_decode(
    float* output,
    const float* input,
    const uint8_t* packed,
    int rows,
    int columns,
    musaStream_t stream) {
  if constexpr (QuantType == 14) {
    if (rows >= 4096 && columns == 2560) {
      linear_q6_tiled_decode_kernel<<<(rows + 7) / 8, 128, 0, stream>>>(
          output, input, packed, rows, columns);
      return musaGetLastError();
    }
    linear_vector_decode_kernel<QuantType, 4>
        <<<(rows + 3) / 4, 128, 0, stream>>>(
            output, input, packed, rows, columns);
  } else if (rows >= 128 && reinterpret_cast<uintptr_t>(packed) % 16 == 0) {
    linear_vector_decode_kernel<QuantType, 16>
        <<<(rows + 15) / 16, 128, 0, stream>>>(
            output, input, packed, rows, columns);
  } else {
    linear_vector_decode_kernel<QuantType, 4>
        <<<(rows + 3) / 4, 128, 0, stream>>>(
            output, input, packed, rows, columns);
  }
  return musaGetLastError();
}

template <int QuantType>
__global__ void embedding_kernel(float* output, const int64_t* indices,
                                 const uint8_t* packed, int* invalid,
                                 int64_t rows, int64_t columns, int64_t count) {
  constexpr int bytes = QuantType == 114 ? 320
      : QuantType == 12                  ? 144
      : QuantType == 13                  ? 176
                                         : 210;
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

#if defined(EXECUTORCH_MUSA_GGUF_SPLIT_K)
extern "C" musaError_t executorch_musa_gguf_parts_launch(
    float* output,
    const float* input,
    const uint8_t* packed,
    int64_t rows,
    int64_t columns,
    int64_t batches,
    int64_t quant_type,
    musaStream_t stream) {
  if (batches != 1 || rows < 128 ||
      rows > std::numeric_limits<int>::max() / 4 || columns < 4096 ||
      columns > std::numeric_limits<int>::max() || columns % 256 != 0 ||
      (quant_type != 12 && quant_type != 114) ||
      reinterpret_cast<uintptr_t>(input) % 16 != 0 ||
      reinterpret_cast<uintptr_t>(packed) % 16 != 0) {
    return musaErrorInvalidValue;
  }
  if (quant_type == 12) {
    linear_vector_decode_kernel<12, 16, 4>
        <<<dim3((rows + 15) / 16, 4), 128, 0, stream>>>(
            output, input, packed, rows, columns);
  } else {
    linear_predecoded_q6_decode_kernel<2, 4>
        <<<dim3((rows + 7) / 8, 4), 128, 0, stream>>>(
            output, input, packed, rows, columns);
  }
  return musaGetLastError();
}

extern "C" musaError_t executorch_musa_gguf_split_k_launch(
    float* output,
    const float* input,
    const uint8_t* packed,
    float* workspace,
    int64_t rows,
    int64_t columns,
    int64_t batches,
    int64_t quant_type,
    musaStream_t stream) {
  const auto status = executorch_musa_gguf_parts_launch(
      workspace, input, packed, rows, columns, batches, quant_type, stream);
  if (status != musaSuccess) {
    return status;
  }
  reduce_split_k_kernel<<<(rows + 127) / 128, 128, 0, stream>>>(
      output, workspace, rows);
  return musaGetLastError();
}
#endif

extern "C" musaError_t executorch_musa_gguf_linear_launch(
    float* output,
    const float* input,
    const uint8_t* packed,
    int64_t rows,
    int64_t columns,
    int64_t batches,
    int64_t quant_type,
    musaStream_t stream) {
  const auto grid = static_cast<unsigned int>(rows * batches);
  if (quant_type == 114) {
#if defined(EXECUTORCH_MUSA_GGUF_GROUPED_Q6_HEAD)
    if (batches == 1 && rows == 248320 && columns == 2560 &&
        reinterpret_cast<uintptr_t>(input) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(packed) % 16 == 0) {
      grouped_predecoded_q6_head_kernel<<<rows / 32, 128, 0, stream>>>(
          output, input, packed);
      return musaGetLastError();
    }
#endif
    if (batches == 1 && columns <= std::numeric_limits<int>::max() &&
        reinterpret_cast<uintptr_t>(input) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(packed) % 16 == 0) {
      if (rows >= 4096 && columns == 2560) {
        linear_predecoded_q6_decode_kernel<4>
            <<<(rows + 15) / 16, 128, 0, stream>>>(
                output,
                input,
                packed,
                static_cast<int>(rows),
                static_cast<int>(columns));
      } else {
        linear_predecoded_q6_decode_kernel<2>
            <<<(rows + 7) / 8, 128, 0, stream>>>(
                output,
                input,
                packed,
                static_cast<int>(rows),
                static_cast<int>(columns));
      }
    } else {
      linear_kernel<114>
          <<<grid, 128, 0, stream>>>(output, input, packed, rows, columns);
    }
    return musaGetLastError();
  }
  if (batches == 1 && columns <= std::numeric_limits<int>::max()) {
    if (reinterpret_cast<uintptr_t>(input) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(packed) % (quant_type == 14 ? 2 : 4) == 0) {
      if (quant_type == 12) {
        return launch_linear_vector_decode<12>(
            output,
            input,
            packed,
            static_cast<int>(rows),
            static_cast<int>(columns),
            stream);
      } else if (quant_type == 13) {
        return launch_linear_vector_decode<13>(
            output,
            input,
            packed,
            static_cast<int>(rows),
            static_cast<int>(columns),
            stream);
      } else if (quant_type == 14) {
        return launch_linear_vector_decode<14>(
            output,
            input,
            packed,
            static_cast<int>(rows),
            static_cast<int>(columns),
            stream);
      }
    }
    const auto decode_grid = static_cast<unsigned int>((rows + 3) / 4);
    if (quant_type == 12) {
      linear_decode_kernel<12><<<decode_grid, 128, 0, stream>>>(
          output,
          input,
          packed,
          static_cast<int>(rows),
          static_cast<int>(columns));
    } else if (quant_type == 13) {
      linear_decode_kernel<13><<<decode_grid, 128, 0, stream>>>(
          output,
          input,
          packed,
          static_cast<int>(rows),
          static_cast<int>(columns));
    } else if (quant_type == 14) {
      linear_decode_kernel<14><<<decode_grid, 128, 0, stream>>>(
          output,
          input,
          packed,
          static_cast<int>(rows),
          static_cast<int>(columns));
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
  if (quant_type == 114) {
    embedding_kernel<114><<<grid, 256, 0, stream>>>(
        output, indices, packed, invalid, rows, columns, count);
  } else if (quant_type == 12) {
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
