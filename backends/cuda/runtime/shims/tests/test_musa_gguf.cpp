#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/cuda/runtime/shims/musa_gguf.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/platform/platform.h>

namespace {

namespace slim = executorch::backends::aoti::slim;
using executorch::backends::cuda::aoti_torch_musa_gguf_embedding_out;
using executorch::backends::cuda::aoti_torch_musa_gguf_linear_out;
using executorch::runtime::Error;
using slim::SlimTensor;
using slim::c10::ScalarType;

SlimTensor tensor(std::initializer_list<int64_t> shape,
                  ScalarType dtype = ScalarType::Float) {
  return slim::empty(shape, dtype, slim::c10::Device(slim::c10::kMUSA, 0));
}

template <typename Value>
void write(SlimTensor& destination, const std::vector<Value>& values) {
  ASSERT_EQ(destination.nbytes(), values.size() * sizeof(Value));
  ASSERT_EQ(cudaMemcpy(destination.data_ptr(), values.data(),
                       destination.nbytes(), cudaMemcpyHostToDevice),
            cudaSuccess);
}

std::vector<float> read(SlimTensor& source) {
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float> values(source.numel());
  EXPECT_EQ(cudaMemcpy(values.data(), source.data_ptr(), source.nbytes(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  return values;
}

float expected(int64_t quant_type, int row, int column) {
  const int element = column % 256;
  const float base = std::ldexp(1.0f, row);
  if (quant_type == 14) {
    const int levels[] = {-31, -15, 2, 18};
    return base * (element / 16 - 8) * levels[(element % 128) / 32];
  }
  const int group = element / 32;
  return base * (group % 2 == 0 ? 1 : quant_type == 12 ? 2 : 18) - 0.5f;
}

SlimTensor weight(int64_t quant_type, int64_t columns, int rows = 3) {
  const int64_t bytes = quant_type == 12 ? 144 : quant_type == 13 ? 176 : 210;
  auto owner = tensor({rows * (columns / 256) * bytes + 2}, ScalarType::Byte);
  auto packed = owner.as_strided({rows, (columns / 256) * bytes},
                                 {(columns / 256) * bytes, 1}, 1);
  std::vector<uint8_t> data(packed.numel(), 0);
  for (int row = 0; row < rows; ++row) {
    for (int64_t index = 0; index < columns / 256; ++index) {
      auto* block = data.data() + (row * (columns / 256) + index) * bytes;
      const auto bits = 0x3c00 + row * 0x400;
      if (quant_type == 14) {
        std::fill(block, block + 128, 0x21);
        std::fill(block + 128, block + 192, 0xe4);
        for (int group = 0; group < 16; ++group) {
          block[192 + group] = static_cast<uint8_t>(group - 8);
        }
        block[208] = bits & 255;
        block[209] = bits >> 8;
      } else {
        block[0] = bits & 255;
        block[1] = bits >> 8;
        block[3] = 0x38;
        std::fill(block + 4, block + 12, 1);
        std::fill(block + 12, block + 16, 0x11);
        if (quant_type == 13) {
          std::fill(block + 16, block + 48, 0xaa);
        }
        std::fill(block + (quant_type == 12 ? 16 : 48), block + bytes, 0x21);
      }
    }
  }
  write(packed, data);
  return packed;
}

class MusaGgufTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "MUSA device is not available";
    }
  }
};

TEST_F(MusaGgufTest, EmbeddingPreservesAllBitPlanesAndOffsetRows) {
  for (int quant_type : {12, 13, 14}) {
    for (int columns : {256, 2560, 4096, 9216}) {
      auto packed = weight(quant_type, columns);
      auto indices = tensor({2, 2}, ScalarType::Long);
      write<int64_t>(indices, {2, 0, 1, 2});
      auto output = tensor({2, 2, columns});
      ASSERT_EQ(aoti_torch_musa_gguf_embedding_out(&output, &indices, &packed,
                                                   quant_type),
                Error::Ok);
      const auto actual = read(output);
      const int rows[] = {2, 0, 1, 2};
      for (int index = 0; index < 4; ++index) {
        for (int column = 0; column < columns; ++column) {
          ASSERT_EQ(actual[index * columns + column],
                    expected(quant_type, rows[index], column));
        }
      }
    }
  }
}

TEST_F(MusaGgufTest, LinearBatchesWidthsAndExactRepresentableSums) {
  for (int quant_type : {12, 13, 14}) {
    for (int columns : {256, 2560, 4096, 9216, 12288}) {
      auto packed = weight(quant_type, columns);
      auto input = tensor({2, 1, columns});
      std::vector<float> values(2 * columns, 1);
      std::fill(values.begin() + columns, values.end(), 2);
      write(input, values);
      auto output = tensor({2, 1, 3});
      ASSERT_EQ(
          aoti_torch_musa_gguf_linear_out(&output, &input, &packed, quant_type),
          Error::Ok);
      const auto actual = read(output);
      for (int row = 0; row < 3; ++row) {
        float sum = 0;
        for (int column = 0; column < columns; ++column) {
          sum += expected(quant_type, row, column);
        }
        EXPECT_EQ(actual[row], sum);
        EXPECT_EQ(actual[3 + row], 2 * sum);
      }
    }
  }
}

TEST_F(MusaGgufTest, DecodePreservesReductionOrderAndPartialRowTiles) {
  for (int quant_type : {12, 13, 14}) {
    for (int columns : {256, 768, 2560, 4096, 9216, 12288}) {
      for (int rows : {1, 3, 4, 5, 9}) {
        auto packed = weight(quant_type, columns, rows);
        auto input = tensor({1, columns});
        std::vector<float> values(columns);
        for (int column = 0; column < columns; ++column) {
          values[column] = std::sin(static_cast<float>(column * 13));
        }
        write(input, values);
        auto output = tensor({1, rows});
        ASSERT_EQ(
            aoti_torch_musa_gguf_linear_out(&output, &input, &packed, quant_type),
            Error::Ok);
        const auto actual = read(output);
        for (int row = 0; row < rows; ++row) {
          std::vector<float> partial(128, 0);
          for (int lane = 0; lane < 128; ++lane) {
            for (int column = lane; column < columns; column += 128) {
              partial[lane] = std::fma(
                  expected(quant_type, row, column), values[column], partial[lane]);
            }
          }
          for (int stride = 64; stride > 0; stride /= 2) {
            for (int lane = 0; lane < stride; ++lane) {
              partial[lane] += partial[lane + stride];
            }
          }
          EXPECT_EQ(actual[row], partial[0]);
        }
      }
    }
  }
}

TEST_F(MusaGgufTest, RejectsOutOfRangeIndicesWithoutReadingWeightOutOfBounds) {
  auto packed = weight(14, 256);
  auto indices = tensor({1}, ScalarType::Long);
  auto output = tensor({1, 256});
  for (int64_t invalid : {-1, 3}) {
    write<int64_t>(indices, {invalid});
    EXPECT_EQ(
        aoti_torch_musa_gguf_embedding_out(&output, &indices, &packed, 14),
        Error::InvalidArgument);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  }
  write<int64_t>(indices, {0});
  EXPECT_EQ(aoti_torch_musa_gguf_embedding_out(&output, &indices, &packed, 14),
            Error::Ok);
}

TEST_F(MusaGgufTest, VectorDecodeMatchesGenericForOffsetsTailsAndBitPlanes) {
  for (int quant_type : {12, 13, 14}) {
    const int block_bytes = quant_type == 12 ? 144
        : quant_type == 13                   ? 176
                                             : 210;
    for (int columns : {256, 768, 2560, 4096, 9216, 12288}) {
      for (int rows : {1, 5, 17, 127, 128, 129}) {
        SCOPED_TRACE(quant_type);
        SCOPED_TRACE(columns);
        SCOPED_TRACE(rows);
        const int row_bytes = columns / 256 * block_bytes;
        std::vector<uint8_t> data(rows * row_bytes);
        uint32_t random_state = 0x73841a9d;
        for (auto& byte : data) {
          random_state ^= random_state << 13;
          random_state ^= random_state >> 17;
          random_state ^= random_state << 5;
          byte = static_cast<uint8_t>(random_state);
        }
        for (size_t block = 0; block < data.size(); block += block_bytes) {
          const auto scale = block + (quant_type == 14 ? 208 : 0);
          data[scale + 1] = (data[scale + 1] & 0x83) | 0x38;
          if (quant_type != 14) {
            data[block + 3] = (data[block + 3] & 0x83) | 0x34;
          }
        }
        auto reference_packed = tensor({rows, row_bytes}, ScalarType::Byte);
        write(reference_packed, data);
        auto batch_input = tensor({2, columns});
        std::vector<float> values(2 * columns);
        for (int column = 0; column < columns; ++column) {
          values[column] = std::sin(static_cast<float>(column * 13)) * 3;
          values[columns + column] = values[column];
        }
        write(batch_input, values);
        auto reference_output = tensor({2, rows});
        ASSERT_EQ(
            aoti_torch_musa_gguf_linear_out(
                &reference_output, &batch_input, &reference_packed, quant_type),
            Error::Ok);
        const auto expected_values = read(reference_output);
        values.resize(columns);
        for (int packed_offset : {0, 1, 2, 4, 16}) {
          auto owner = tensor({rows * row_bytes + 16}, ScalarType::Byte);
          auto packed = owner.as_strided(
              {rows, row_bytes}, {row_bytes, 1}, packed_offset);
          write(packed, data);
          for (int input_offset : {0, 1, 4}) {
            SCOPED_TRACE(packed_offset);
            SCOPED_TRACE(input_offset);
            auto input_owner = tensor({columns + 4});
            auto input = input_owner.as_strided(
                {1, columns}, {columns, 1}, input_offset);
            write(input, values);
            auto output_owner = tensor({rows + 1});
            auto output = output_owner.as_strided({1, rows}, {rows, 1}, 1);
            ASSERT_EQ(
                aoti_torch_musa_gguf_linear_out(
                    &output, &input, &packed, quant_type),
                Error::Ok);
            const auto actual = read(output);
            for (int row = 0; row < rows; ++row) {
              ASSERT_TRUE(std::isfinite(actual[row]));
              ASSERT_EQ(
                  std::memcmp(
                      &actual[row], &expected_values[row], sizeof(float)),
                  0);
            }
          }
        }
      }
    }
  }
}

TEST_F(MusaGgufTest, RejectsShapeTypeStridesAliasesAndStorageViolations) {
  auto packed = weight(12, 256);
  auto input = tensor({1, 256});
  auto output = tensor({1, 3});
  EXPECT_EQ(aoti_torch_musa_gguf_linear_out(nullptr, &input, &packed, 12),
            Error::InvalidArgument);
  EXPECT_EQ(aoti_torch_musa_gguf_linear_out(&output, &input, &packed, 11),
            Error::InvalidArgument);
  auto alias = input.as_strided({1, 3}, {3, 1}, 0);
  EXPECT_EQ(aoti_torch_musa_gguf_linear_out(&alias, &input, &packed, 12),
            Error::InvalidArgument);
  auto short_input = input.as_strided({1, 255}, {255, 1}, 0);
  EXPECT_EQ(aoti_torch_musa_gguf_linear_out(&output, &short_input, &packed, 12),
            Error::InvalidArgument);
  auto bad_storage = input.as_strided({1, 256}, {256, 1}, 1);
  EXPECT_EQ(aoti_torch_musa_gguf_linear_out(&output, &bad_storage, &packed, 12),
            Error::InvalidArgument);
  auto broadcast = input.as_strided({1, 256}, {0, 0}, 0);
  EXPECT_EQ(aoti_torch_musa_gguf_linear_out(&output, &broadcast, &packed, 12),
            Error::InvalidArgument);
  auto half_input = tensor({1, 256}, ScalarType::Half);
  EXPECT_EQ(aoti_torch_musa_gguf_linear_out(&output, &half_input, &packed, 12),
            Error::InvalidArgument);
  auto wrong_output = tensor({1, 4});
  EXPECT_EQ(aoti_torch_musa_gguf_linear_out(&wrong_output, &input, &packed, 12),
            Error::InvalidArgument);
  auto incomplete = packed.as_strided({3, 143}, {143, 1}, 0);
  EXPECT_EQ(aoti_torch_musa_gguf_linear_out(&output, &input, &incomplete, 12),
            Error::InvalidArgument);
  input.reset();
  EXPECT_EQ(aoti_torch_musa_gguf_linear_out(&output, &input, &packed, 12),
            Error::InvalidArgument);
}

TEST_F(MusaGgufTest, UsesCallingStreamOnDeviceZero) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  {
    auto guard = executorch::backends::cuda::CUDAStreamGuard::create(stream, 0);
    ASSERT_TRUE(guard.ok());
    auto packed = weight(12, 256);
    auto input = tensor({1, 256});
    auto output = tensor({1, 3});
    ASSERT_EQ(cudaMemsetAsync(input.data_ptr(), 0, input.nbytes(), stream),
              cudaSuccess);
    ASSERT_EQ(aoti_torch_musa_gguf_linear_out(&output, &input, &packed, 12),
              Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (float value : read(output)) {
      EXPECT_EQ(value, 0);
    }
    int device = -1;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    EXPECT_EQ(device, 0);
  }
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(MusaGgufTest, VectorDecodeUsesCallingStreamWithPartialRowTile) {
  cudaStream_t stream;
  ASSERT_EQ(musaStreamCreateWithFlags(&stream, musaStreamNonBlocking), musaSuccess);
  {
    auto guard = executorch::backends::cuda::CUDAStreamGuard::create(stream, 0);
    ASSERT_TRUE(guard.ok());
    auto packed = tensor({129, 144}, ScalarType::Byte);
    std::vector<uint8_t> data(129 * 144, 0);
    for (int row = 0; row < 129; ++row) {
      auto* block = data.data() + row * 144;
      block[1] = 0x3c;
      std::fill(block + 4, block + 12, 1);
      std::fill(block + 12, block + 144, 0x11);
    }
    write(packed, data);
    auto input = tensor({1, 256});
    write(input, std::vector<float>(256, 1));
    auto output = tensor({1, 129});
    ASSERT_EQ(cudaMemsetAsync(input.data_ptr(), 0, input.nbytes(), stream), cudaSuccess);
    ASSERT_EQ(aoti_torch_musa_gguf_linear_out(&output, &input, &packed, 12), Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (float value : read(output)) {
      EXPECT_EQ(value, 0);
    }
  }
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}
