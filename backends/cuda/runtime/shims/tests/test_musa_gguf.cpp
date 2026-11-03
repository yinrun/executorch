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

#if defined(EXECUTORCH_MUSA_GGUF_SPLIT_K)
std::vector<float> split_reference(
    const std::vector<uint8_t>& data,
    const std::vector<float>& values,
    int quant_type,
    int rows,
    int columns) {
  const int bytes = quant_type == 12 ? 144 : 210;
  std::vector<float> result(rows, 0);
  for (int part = 0; part < 4; ++part) {
    const int first = (columns / 256) * part / 4;
    const int last = (columns / 256) * (part + 1) / 4;
    const int part_columns = (last - first) * 256;
    auto packed = tensor({rows, (last - first) * bytes}, ScalarType::Byte);
    std::vector<uint8_t> part_data(packed.numel());
    for (int row = 0; row < rows; ++row) {
      std::copy_n(
          data.begin() + (row * (columns / 256) + first) * bytes,
          (last - first) * bytes,
          part_data.begin() + row * (last - first) * bytes);
    }
    write(packed, part_data);
    auto input = tensor({2, part_columns});
    std::vector<float> part_values(2 * part_columns);
    for (int batch = 0; batch < 2; ++batch) {
      std::copy_n(
          values.begin() + first * 256,
          part_columns,
          part_values.begin() + batch * part_columns);
    }
    write(input, part_values);
    auto output = tensor({2, rows});
    EXPECT_EQ(
        aoti_torch_musa_gguf_linear_out(&output, &input, &packed, quant_type),
        Error::Ok);
    const auto partial = read(output);
    for (int row = 0; row < rows; ++row) {
      if (part == 0) {
        result[row] = partial[row];
      } else {
        result[row] += partial[row];
      }
    }
  }
  return result;
}
#endif

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
#if defined(EXECUTORCH_MUSA_GGUF_SPLIT_K)
        const auto split_values =
            quant_type == 12 && rows >= 128 && columns >= 4096
            ? split_reference(data, values, quant_type, rows, columns)
            : std::vector<float>{};
#endif
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
            const auto* expected_output = expected_values.data();
#if defined(EXECUTORCH_MUSA_GGUF_SPLIT_K)
            if (!split_values.empty() && packed_offset % 16 == 0 &&
                input_offset % 4 == 0) {
              expected_output = split_values.data();
            }
#endif
            for (int row = 0; row < rows; ++row) {
              ASSERT_TRUE(std::isfinite(actual[row]));
              ASSERT_EQ(
                  std::memcmp(
                      &actual[row], &expected_output[row], sizeof(float)),
                  0);
            }
          }
        }
      }
    }
  }
}

TEST_F(MusaGgufTest, TiledQ6DecodeMatchesGenericAtDispatchAndRowBoundaries) {
  for (int columns : {256, 2560}) {
    for (int rows : {4095, 4096, 4097}) {
      const int row_bytes = columns / 256 * 210;
      std::vector<uint8_t> data(rows * row_bytes);
      uint32_t random_state = 0x34591a7d;
      for (auto& byte : data) {
        random_state ^= random_state << 13;
        random_state ^= random_state >> 17;
        random_state ^= random_state << 5;
        byte = static_cast<uint8_t>(random_state);
      }
      for (size_t block = 0; block < data.size(); block += 210) {
        data[block + 209] = (data[block + 209] & 0x83) | 0x38;
      }
      auto reference_packed = tensor({rows, row_bytes}, ScalarType::Byte);
      write(reference_packed, data);
      std::vector<float> values(2 * columns);
      for (int column = 0; column < columns; ++column) {
        values[column] = std::sin(static_cast<float>(column * 17)) * 3;
        values[columns + column] = values[column];
      }
      auto reference_input = tensor({2, columns});
      auto reference_output = tensor({2, rows});
      write(reference_input, values);
      ASSERT_EQ(
          aoti_torch_musa_gguf_linear_out(
              &reference_output, &reference_input, &reference_packed, 14),
          Error::Ok);
      const auto expected = read(reference_output);
      auto input = tensor({1, columns});
      values.resize(columns);
      write(input, values);
      for (int offset : {0, 1, 2}) {
        SCOPED_TRACE(columns);
        SCOPED_TRACE(rows);
        SCOPED_TRACE(offset);
        auto packed_owner = tensor({rows * row_bytes + 2}, ScalarType::Byte);
        auto packed = packed_owner.as_strided(
            {rows, row_bytes}, {row_bytes, 1}, offset);
        write(packed, data);
        auto output_owner = tensor({rows + 2});
        write(output_owner, std::vector<float>(rows + 2, 12345));
        auto output = output_owner.as_strided({1, rows}, {rows, 1}, 1);
        ASSERT_EQ(
            aoti_torch_musa_gguf_linear_out(&output, &input, &packed, 14),
            Error::Ok);
        const auto actual = read(output_owner);
        ASSERT_EQ(actual.front(), 12345);
        ASSERT_EQ(actual.back(), 12345);
        for (int row = 0; row < rows; ++row) {
          ASSERT_TRUE(std::isfinite(actual[row + 1]));
          ASSERT_EQ(
              std::memcmp(&actual[row + 1], &expected[row], sizeof(float)), 0);
        }
      }
    }
  }
}

TEST_F(MusaGgufTest, PredecodedQ6MatchesOriginalBitsBatchesOffsetsAndRowTails) {
  for (int columns : {256, 2560, 9216}) {
    const auto row_counts = columns == 2560
        ? std::vector<int>{1, 7, 8, 9, 15, 16, 17, 129, 4095, 4096, 4097}
        : std::vector<int>{1, 7, 8, 9, 15, 16, 17, 129};
    for (int rows : row_counts) {
      SCOPED_TRACE(columns);
      SCOPED_TRACE(rows);
      const int blocks = rows * (columns / 256);
      std::vector<uint8_t> original(blocks * 210);
      std::vector<uint8_t> prepared(blocks * 320);
      uint32_t random_state = 0x514831d9;
      for (auto& byte : original) {
        random_state ^= random_state << 13;
        random_state ^= random_state >> 17;
        random_state ^= random_state << 5;
        byte = static_cast<uint8_t>(random_state);
      }
      for (int block = 0; block < blocks; ++block) {
        auto* source = original.data() + block * 210;
        auto* target = prepared.data() + block * 320;
        source[208] = 0;
        source[209] = block % 2 == 0 ? 0x38 : 0xb8;
        for (int element = 0; element < 256; ++element) {
          const int half = element / 128;
          const int group = element % 128 / 32;
          const int lane = element % 32;
          const int low =
              (source[half * 64 + group % 2 * 32 + lane] >> (group / 2 * 4)) &
              15;
          const int high = (source[128 + half * 32 + lane] >> (group * 2)) & 3;
          target[element] = static_cast<uint8_t>((low | (high << 4)) - 32);
        }
        for (int group = 0; group < 16; ++group) {
          const float scale = (block % 2 == 0 ? 0.5f : -0.5f) *
              static_cast<int8_t>(source[192 + group]);
          std::memcpy(target + 256 + group * 4, &scale, sizeof(float));
        }
      }
      auto original_tensor =
          tensor({rows, columns / 256 * 210}, ScalarType::Byte);
      write(original_tensor, original);
      std::vector<float> values(2 * columns);
      for (int column = 0; column < columns; ++column) {
        values[column] = std::sin(static_cast<float>(column * 13)) * 3;
        values[columns + column] = values[column];
      }
      auto reference_input = tensor({2, columns});
      auto reference_output = tensor({2, rows});
      write(reference_input, values);
      ASSERT_EQ(
          aoti_torch_musa_gguf_linear_out(
              &reference_output, &reference_input, &original_tensor, 14),
          Error::Ok);
      const auto expected_values = read(reference_output);
#if defined(EXECUTORCH_MUSA_GGUF_SPLIT_K)
      const auto split_values = rows >= 128 && columns >= 4096
          ? split_reference(original, values, 14, rows, columns)
          : std::vector<float>{};
#endif
      for (int packed_offset : {0, 1, 4, 16}) {
        auto owner = tensor({blocks * 320 + 16}, ScalarType::Byte);
        auto packed = owner.as_strided(
            {rows, columns / 256 * 320},
            {columns / 256 * 320, 1},
            packed_offset);
        write(packed, prepared);
        for (int batches : {1, 2}) {
          for (int input_offset : {0, 1}) {
            auto input_owner = tensor({batches * columns + 1});
            auto input = input_owner.as_strided(
                {batches, columns}, {columns, 1}, input_offset);
            write(
                input,
                std::vector<float>(
                    values.begin(), values.begin() + batches * columns));
            auto output_owner = tensor({batches * rows + 2});
            write(output_owner, std::vector<float>(batches * rows + 2, 12345));
            auto output =
                output_owner.as_strided({batches, rows}, {rows, 1}, 1);
            ASSERT_EQ(
                aoti_torch_musa_gguf_linear_out(&output, &input, &packed, 114),
                Error::Ok);
            const auto actual = read(output_owner);
            ASSERT_EQ(actual.front(), 12345);
            ASSERT_EQ(actual.back(), 12345);
            const auto* expected_output = expected_values.data();
#if defined(EXECUTORCH_MUSA_GGUF_SPLIT_K)
            if (!split_values.empty() && batches == 1 &&
                packed_offset % 16 == 0 && input_offset == 0) {
              expected_output = split_values.data();
            }
#endif
            ASSERT_EQ(
                std::memcmp(
                    actual.data() + 1,
                    expected_output,
                    batches * rows * sizeof(float)),
                0);
          }
        }
        auto indices = tensor({2}, ScalarType::Long);
        write<int64_t>(indices, {rows - 1, 0});
        auto original_embedding = tensor({2, columns});
        auto prepared_embedding = tensor({2, columns});
        ASSERT_EQ(
            aoti_torch_musa_gguf_embedding_out(
                &original_embedding, &indices, &original_tensor, 14),
            Error::Ok);
        ASSERT_EQ(
            aoti_torch_musa_gguf_embedding_out(
                &prepared_embedding, &indices, &packed, 114),
            Error::Ok);
        const auto expected_embedding = read(original_embedding);
        const auto actual_embedding = read(prepared_embedding);
        ASSERT_EQ(
            std::memcmp(
                actual_embedding.data(),
                expected_embedding.data(),
                expected_embedding.size() * sizeof(float)),
            0);
      }
    }
  }
}

TEST_F(MusaGgufTest, PredecodedQ6FullHeadAccuracyAndCallingStream) {
  constexpr int rows = 248320;
  constexpr int columns = 2560;
  constexpr int patterns = 17;
  constexpr int row_bytes = columns / 256 * 320;
  std::vector<uint8_t> templates(patterns * row_bytes);
  std::vector<float> values(columns);
  for (int column = 0; column < columns; ++column) {
    values[column] = std::sin(static_cast<float>(column * 13)) * 3;
  }
  std::vector<double> expected_values(patterns, 0);
  for (int pattern = 0; pattern < patterns; ++pattern) {
    for (int block = 0; block < columns / 256; ++block) {
      auto* target = templates.data() + pattern * row_bytes + block * 320;
      for (int group = 0; group < 16; ++group) {
        const float scale = (pattern + group + block) % 2 == 0
            ? (group + 1) / 256.0f
            : -(group + 1) / 256.0f;
        std::memcpy(target + 256 + group * 4, &scale, sizeof(float));
        for (int lane = 0; lane < 16; ++lane) {
          const int level = (pattern * 11 + group * 7 + lane + block) % 64 - 32;
          target[group * 16 + lane] = static_cast<uint8_t>(level);
          expected_values[pattern] += static_cast<double>(scale) * level *
              values[block * 256 + group * 16 + lane];
        }
      }
    }
  }
  std::vector<uint8_t> data(static_cast<size_t>(rows) * row_bytes);
  for (int row = 0; row < rows; ++row) {
    std::memcpy(
        data.data() + static_cast<size_t>(row) * row_bytes,
        templates.data() + (row % patterns) * row_bytes,
        row_bytes);
  }
  auto packed = tensor({rows, row_bytes}, ScalarType::Byte);
  auto input = tensor({1, columns});
  auto owner = tensor({rows + 2});
  auto output = owner.as_strided({1, rows}, {rows, 1}, 1);
  write(packed, data);
  write(input, values);
  write(owner, std::vector<float>(rows + 2, 12345));
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  {
    auto guard = executorch::backends::cuda::CUDAStreamGuard::create(stream, 0);
    ASSERT_TRUE(guard.ok());
    ASSERT_EQ(
        aoti_torch_musa_gguf_linear_out(&output, &input, &packed, 114),
        Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    const auto actual = read(owner);
    ASSERT_EQ(actual.front(), 12345);
    ASSERT_EQ(actual.back(), 12345);
    for (int row = 0; row < rows; ++row) {
      ASSERT_TRUE(std::isfinite(actual[row + 1]));
      const double reference = expected_values[row % patterns];
      ASSERT_NEAR(
          actual[row + 1], reference, 0.001 + 0.001 * std::abs(reference));
    }
    ASSERT_EQ(
        cudaMemsetAsync(input.data_ptr(), 0, input.nbytes(), stream),
        cudaSuccess);
    ASSERT_EQ(
        aoti_torch_musa_gguf_linear_out(&output, &input, &packed, 114),
        Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (float value : read(output)) {
      ASSERT_EQ(value, 0);
    }
  }
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(MusaGgufTest, PredecodedQ6UsesCallingStreamAndObservesUpdates) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  {
    auto guard = executorch::backends::cuda::CUDAStreamGuard::create(stream, 0);
    ASSERT_TRUE(guard.ok());
    auto packed = tensor({9, 320}, ScalarType::Byte);
    std::vector<uint8_t> data(9 * 320, 1);
    const float scale = 1;
    for (int row = 0; row < 9; ++row) {
      for (int group = 0; group < 16; ++group) {
        std::memcpy(
            data.data() + row * 320 + 256 + group * 4, &scale, sizeof(float));
      }
    }
    write(packed, data);
    auto input = tensor({1, 256});
    write(input, std::vector<float>(256, 1));
    auto output = tensor({1, 9});
    ASSERT_EQ(
        aoti_torch_musa_gguf_linear_out(&output, &input, &packed, 114),
        Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (float value : read(output)) {
      EXPECT_EQ(value, 256);
    }
    ASSERT_EQ(
        cudaMemsetAsync(input.data_ptr(), 0, input.nbytes(), stream),
        cudaSuccess);
    ASSERT_EQ(
        aoti_torch_musa_gguf_linear_out(&output, &input, &packed, 114),
        Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (float value : read(output)) {
      EXPECT_EQ(value, 0);
    }
    write(input, std::vector<float>(256, 1));
    ASSERT_EQ(
        cudaMemsetAsync(packed.data_ptr(), 0, packed.nbytes(), stream),
        cudaSuccess);
    ASSERT_EQ(
        aoti_torch_musa_gguf_linear_out(&output, &input, &packed, 114),
        Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (float value : read(output)) {
      EXPECT_EQ(value, 0);
    }
  }
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
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

TEST_F(MusaGgufTest, LongKProjectionsCoverPartitionsTailsOffsetsAndFallbacks) {
  struct Shape {
    int rows;
    int columns;
    int batches;
    int offset;
  };
  for (int quant_type : {12, 13, 114}) {
    for (const Shape shape :
         {Shape{129, 2560, 1, 0},
          {129, 4096, 1, 0},
          {257, 4352, 1, 0},
          {129, 9216, 1, 0},
          {127, 4096, 1, 0},
          {129, 4096, 2, 0},
          {129, 4096, 1, 1}}) {
      SCOPED_TRACE(
          ::testing::Message()
          << quant_type << ":" << shape.rows << ":" << shape.columns << ":"
          << shape.batches << ":" << shape.offset);
      const int bytes = quant_type == 114 ? 320 : quant_type == 12 ? 144 : 176;
      const int row_bytes = shape.columns / 256 * bytes;
      auto owner =
          tensor({shape.rows * row_bytes + shape.offset}, ScalarType::Byte);
      auto packed = owner.as_strided(
          {shape.rows, row_bytes}, {row_bytes, 1}, shape.offset);
      auto input_owner = tensor({shape.batches * shape.columns + shape.offset});
      auto input = input_owner.as_strided(
          {shape.batches, shape.columns}, {shape.columns, 1}, shape.offset);
      std::vector<float> values(input.numel());
      for (size_t index = 0; index < values.size(); ++index) {
        values[index] = std::sin(static_cast<float>(index) * 0.13f);
      }
      write(input, values);
      std::vector<uint8_t> data(packed.numel(), 0);
      std::vector<double> reference(shape.batches * shape.rows, 0);
      for (int row = 0; row < shape.rows; ++row) {
        for (int block_index = 0; block_index < shape.columns / 256;
             ++block_index) {
          auto* block = data.data() + row * row_bytes + block_index * bytes;
          if (quant_type != 114) {
            block[1] = 0x30;
            block[3] = 0x2c;
            std::fill(block + 4, block + 12, 1);
            std::fill(block + 12, block + 16, 0x11);
          }
          for (int element = 0; element < 256; ++element) {
            const int code = row * 7 + block_index * 3 + element;
            float weight_value;
            if (quant_type == 114) {
              const int level = code % 64 - 32;
              const float scale =
                  static_cast<float>(
                      (row + block_index + element / 16) % 7 - 3) /
                  64;
              block[element] = static_cast<uint8_t>(level);
              std::memcpy(
                  block + 256 + element / 16 * sizeof(float),
                  &scale,
                  sizeof(float));
              weight_value = scale * level;
            } else {
              const int level = code % (quant_type == 12 ? 16 : 32);
              const int group = element / 32;
              const int lane = element % 32;
              const int offset = quant_type == 12 ? 16 : 48;
              block[offset + group / 2 * 32 + lane] |= (level & 15)
                  << (group % 2 * 4);
              if (quant_type == 13) {
                block[16 + lane] |= (level >> 4) << group;
              }
              weight_value = 0.125f * level - 0.0625f;
            }
            for (int batch = 0; batch < shape.batches; ++batch) {
              reference[batch * shape.rows + row] +=
                  static_cast<double>(weight_value) *
                  values[batch * shape.columns + block_index * 256 + element];
            }
          }
        }
      }
      write(packed, data);
      auto output = tensor({shape.batches, shape.rows});
      ASSERT_EQ(
          aoti_torch_musa_gguf_linear_out(&output, &input, &packed, quant_type),
          Error::Ok);
      const auto actual = read(output);
      for (size_t index = 0; index < actual.size(); ++index) {
        ASSERT_TRUE(std::isfinite(actual[index]));
        ASSERT_NEAR(
            actual[index],
            reference[index],
            0.001 + 0.001 * std::abs(reference[index]));
      }
    }
  }
}

TEST_F(MusaGgufTest, LongKWorkspaceLifetimeAndCallingStream) {
  cudaStream_t stream;
  ASSERT_EQ(
      musaStreamCreateWithFlags(&stream, musaStreamNonBlocking), musaSuccess);
  {
    auto guard = executorch::backends::cuda::CUDAStreamGuard::create(stream, 0);
    ASSERT_TRUE(guard.ok());
    for (int quant_type : {12, 114}) {
      const int bytes = quant_type == 12 ? 144 : 320;
      auto packed = tensor({129, 16 * bytes}, ScalarType::Byte);
      std::vector<uint8_t> data(packed.numel(), 0);
      for (int index = 0; index < 129 * 16; ++index) {
        auto* block = data.data() + index * bytes;
        if (quant_type == 12) {
          block[1] = 0x3c;
          std::fill(block + 4, block + 12, 1);
          std::fill(block + 12, block + 144, 0x11);
        } else {
          std::fill(block, block + 256, 1);
          const float scale = 1;
          for (int group = 0; group < 16; ++group) {
            std::memcpy(
                block + 256 + group * sizeof(float), &scale, sizeof(float));
          }
        }
      }
      write(packed, data);
      auto positive = tensor({1, 4096});
      auto zero = tensor({1, 4096});
      write(positive, std::vector<float>(4096, 1));
      ASSERT_EQ(
          cudaMemsetAsync(zero.data_ptr(), 0, zero.nbytes(), stream),
          cudaSuccess);
      std::vector<SlimTensor> outputs;
      for (int iteration = 0; iteration < 8; ++iteration) {
        outputs.push_back(tensor({1, 129}));
        auto& input = iteration % 2 == 0 ? positive : zero;
        ASSERT_EQ(
            aoti_torch_musa_gguf_linear_out(
                &outputs.back(), &input, &packed, quant_type),
            Error::Ok);
      }
      ASSERT_EQ(
          cudaMemsetAsync(packed.data_ptr(), 0, packed.nbytes(), stream),
          cudaSuccess);
      outputs.push_back(tensor({1, 129}));
      ASSERT_EQ(
          aoti_torch_musa_gguf_linear_out(
              &outputs.back(), &positive, &packed, quant_type),
          Error::Ok);
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      for (size_t iteration = 0; iteration < outputs.size(); ++iteration) {
        for (float value : read(outputs[iteration])) {
          EXPECT_EQ(value, iteration < 8 && iteration % 2 == 0 ? 4096 : 0);
        }
      }
    }
  }
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
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
    ASSERT_EQ(aoti_torch_musa_gguf_linear_out(&output, &input, &packed, 12), Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (float value : read(output)) {
      EXPECT_EQ(value, 256);
    }
    ASSERT_EQ(cudaMemsetAsync(input.data_ptr(), 0, input.nbytes(), stream), cudaSuccess);
    ASSERT_EQ(aoti_torch_musa_gguf_linear_out(&output, &input, &packed, 12), Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (float value : read(output)) {
      EXPECT_EQ(value, 0);
    }
    write(input, std::vector<float>(256, 1));
    ASSERT_EQ(cudaMemsetAsync(packed.data_ptr(), 0, packed.nbytes(), stream), cudaSuccess);
    ASSERT_EQ(aoti_torch_musa_gguf_linear_out(&output, &input, &packed, 12), Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (float value : read(output)) {
      EXPECT_EQ(value, 0);
    }
  }
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}
