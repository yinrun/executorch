#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/platform/platform.h>

namespace executorch::backends::cuda {
extern "C" {
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_musa_convolution(
    SlimTensor*,
    SlimTensor*,
    SlimTensor* const*,
    const int64_t*,
    int64_t,
    const int64_t*,
    int64_t,
    const int64_t*,
    int64_t,
    int32_t,
    const int64_t*,
    int64_t,
    int64_t,
    SlimTensor**);
}
}

namespace {

using executorch::backends::aoti::slim::c10::DeviceType;
using executorch::backends::aoti::slim::c10::ScalarType;
using executorch::backends::cuda::SlimTensor;
using executorch::backends::cuda::aoti_torch_empty_strided;
using executorch::backends::cuda::aoti_torch_musa_convolution;
using executorch::runtime::Error;

uint16_t float_to_half_bits(float value) {
  if (value == 0.0f) {
    return 0;
  }
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t exponent = (bits >> 23) & 0xff;
  const uint32_t mantissa = bits & 0x7fffff;
  return static_cast<uint16_t>(((exponent - 112) << 10) | (mantissa >> 13));
}

float half_bits_to_float(uint16_t value) {
  const uint32_t exponent = (value >> 10) & 0x1f;
  const uint32_t mantissa = value & 0x3ff;
  uint32_t bits = ((exponent + 112) << 23) | (mantissa << 13);
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t float_to_bfloat16_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

float bfloat16_bits_to_float(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

size_t itemsize(ScalarType dtype) {
  return dtype == ScalarType::Float ? sizeof(float) : sizeof(uint16_t);
}

std::vector<uint8_t> encode_values(
    const std::vector<float>& values,
    ScalarType dtype) {
  std::vector<uint8_t> encoded(values.size() * itemsize(dtype));
  for (size_t index = 0; index < values.size(); ++index) {
    if (dtype == ScalarType::Float) {
      std::memcpy(
          encoded.data() + index * sizeof(float),
          &values[index],
          sizeof(float));
      continue;
    }
    const uint16_t bits = dtype == ScalarType::Half
        ? float_to_half_bits(values[index])
        : float_to_bfloat16_bits(values[index]);
    std::memcpy(
        encoded.data() + index * sizeof(uint16_t), &bits, sizeof(bits));
  }
  return encoded;
}

std::vector<float> decode_values(
    const std::vector<uint8_t>& encoded,
    ScalarType dtype) {
  const size_t element_count = encoded.size() / itemsize(dtype);
  std::vector<float> values(element_count);
  for (size_t index = 0; index < element_count; ++index) {
    if (dtype == ScalarType::Float) {
      std::memcpy(
          &values[index],
          encoded.data() + index * sizeof(float),
          sizeof(float));
      continue;
    }
    uint16_t bits;
    std::memcpy(
        &bits, encoded.data() + index * sizeof(uint16_t), sizeof(bits));
    values[index] = dtype == ScalarType::Half
        ? half_bits_to_float(bits)
        : bfloat16_bits_to_float(bits);
  }
  return values;
}

SlimTensor* empty_tensor(
    ScalarType dtype,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides) {
  SlimTensor* tensor = nullptr;
  const Error error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      strides.data(),
      static_cast<int32_t>(dtype),
      static_cast<int32_t>(DeviceType::CUDA),
      0,
      &tensor);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor, nullptr);
  return tensor;
}

void copy_to_device(SlimTensor* tensor, const std::vector<uint8_t>& values) {
  EXPECT_EQ(
      cudaMemcpy(
          tensor->data_ptr(),
          values.data(),
          values.size(),
          cudaMemcpyHostToDevice),
      cudaSuccess);
}

std::vector<float> copy_from_device(SlimTensor* tensor, ScalarType dtype) {
  std::vector<uint8_t> encoded(tensor->nbytes());
  EXPECT_EQ(
      cudaMemcpy(
          encoded.data(),
          tensor->data_ptr(),
          encoded.size(),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  return decode_values(encoded, dtype);
}

void expect_values(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], 0.05f) << "index " << index;
  }
}

class MusaConvolutionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "MUSA device is not available";
    }
  }
};

TEST_F(MusaConvolutionTest, QueuedWorkspaceAcrossStreamsAndThreads) {
  std::array<std::thread, 2> workers;
  for (size_t worker_index = 0; worker_index < workers.size(); ++worker_index) {
    workers[worker_index] = std::thread([worker_index] {
      ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
      std::array<cudaStream_t, 2> streams{};
      for (auto& stream : streams) {
        ASSERT_EQ(
            musaStreamCreateWithFlags(&stream, musaStreamNonBlocking),
            musaSuccess);
      }
      for (auto dtype :
           {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
        const int64_t batch = worker_index + 1;
        const int64_t height = 17 + worker_index * 2;
        const int64_t width = 19 + worker_index * 2;
        const float input_value = 0.25f * (worker_index + 1);
        ASSERT_EQ(
            executorch::backends::cuda::setCurrentCUDAStream(streams[0], 0),
            Error::Ok);
        auto input = std::unique_ptr<SlimTensor>(empty_tensor(
            dtype,
            {batch, 32, height, width},
            {32 * height * width, height * width, width, 1}));
        auto weight = std::unique_ptr<SlimTensor>(
            empty_tensor(dtype, {32, 32, 3, 3}, {288, 9, 3, 1}));
        auto bias = std::unique_ptr<SlimTensor>(empty_tensor(dtype, {32}, {1}));
        copy_to_device(
            input.get(),
            encode_values(
                std::vector<float>(input->numel(), input_value), dtype));
        copy_to_device(
            weight.get(),
            encode_values(std::vector<float>(weight->numel(), 0.125f), dtype));
        copy_to_device(
            bias.get(), encode_values(std::vector<float>(32, 0.5f), dtype));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const int64_t stride[] = {1, 1};
        const int64_t padding[] = {1, 1};
        const int64_t dilation[] = {1, 1};
        const int64_t output_padding[] = {0, 0};
        SlimTensor* bias_handle = bias.get();
        std::vector<std::unique_ptr<SlimTensor>> outputs;
        for (size_t repeat = 0; repeat < 12; ++repeat) {
          ASSERT_EQ(
              executorch::backends::cuda::setCurrentCUDAStream(
                  streams[(repeat / 3) % streams.size()], 0),
              Error::Ok);
          SlimTensor* output = nullptr;
          ASSERT_EQ(
              aoti_torch_musa_convolution(
                  input.get(),
                  weight.get(),
                  &bias_handle,
                  stride,
                  2,
                  padding,
                  2,
                  dilation,
                  2,
                  false,
                  output_padding,
                  2,
                  1,
                  &output),
              Error::Ok);
          ASSERT_NE(output, nullptr);
          outputs.emplace_back(output);
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for (const auto& output : outputs) {
          const auto actual = copy_from_device(output.get(), dtype);
          for (size_t index = 0; index < actual.size(); ++index) {
            const int64_t row = (index / width) % height;
            const int64_t column = index % width;
            const int64_t rows = row == 0 || row == height - 1 ? 2 : 3;
            const int64_t columns = column == 0 || column == width - 1 ? 2 : 3;
            const float expected =
                rows * columns * 32 * input_value * 0.125f + 0.5f;
            ASSERT_NEAR(actual[index], expected, 0.05f) << "index " << index;
          }
        }
      }
      executorch::backends::cuda::clearCurrentCUDAStream(0);
      for (auto stream : streams) {
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
}

TEST_F(MusaConvolutionTest, WorkspaceGrowthAndStreamRecreation) {
  std::thread worker([] {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    for (auto dtype :
         {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
      auto weight = std::unique_ptr<SlimTensor>(
          empty_tensor(dtype, {32, 32, 3, 3}, {288, 9, 3, 1}));
      auto bias = std::unique_ptr<SlimTensor>(empty_tensor(dtype, {32}, {1}));
      copy_to_device(
          weight.get(),
          encode_values(std::vector<float>(weight->numel(), 0.125f), dtype));
      copy_to_device(
          bias.get(), encode_values(std::vector<float>(32, 0.5f), dtype));
      const int64_t stride[] = {1, 1};
      const int64_t padding[] = {1, 1};
      const int64_t dilation[] = {1, 1};
      const int64_t output_padding[] = {0, 0};
      const std::array<int64_t, 8> dimensions{17, 33, 25, 49, 17, 41, 25, 17};
      SlimTensor* bias_handle = bias.get();
      std::vector<std::unique_ptr<SlimTensor>> inputs;
      std::vector<std::unique_ptr<SlimTensor>> outputs;
      for (bool recreate : {false, true}) {
        cudaStream_t stream = nullptr;
        for (size_t iteration = 0; iteration < dimensions.size(); ++iteration) {
          if (stream == nullptr) {
            ASSERT_EQ(
                musaStreamCreateWithFlags(&stream, musaStreamNonBlocking),
                musaSuccess);
          }
          ASSERT_EQ(
              executorch::backends::cuda::setCurrentCUDAStream(stream, 0),
              Error::Ok);
          const int64_t dimension = dimensions[iteration];
          auto input = std::unique_ptr<SlimTensor>(empty_tensor(
              dtype,
              {1, 32, dimension, dimension},
              {32 * dimension * dimension,
               dimension * dimension,
               dimension,
               1}));
          copy_to_device(
              input.get(),
              encode_values(std::vector<float>(input->numel(), 0.25f), dtype));
          SlimTensor* output = nullptr;
          ASSERT_EQ(
              aoti_torch_musa_convolution(
                  input.get(),
                  weight.get(),
                  &bias_handle,
                  stride,
                  2,
                  padding,
                  2,
                  dilation,
                  2,
                  false,
                  output_padding,
                  2,
                  1,
                  &output),
              Error::Ok);
          ASSERT_NE(output, nullptr);
          inputs.push_back(std::move(input));
          outputs.emplace_back(output);
          if (recreate && iteration % 2 == 1) {
            executorch::backends::cuda::clearCurrentCUDAStream(0);
            ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
            stream = nullptr;
          }
        }
        if (stream != nullptr) {
          executorch::backends::cuda::clearCurrentCUDAStream(0);
          ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        }
      }
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      for (const auto& output : outputs) {
        const int64_t dimension = output->size(2);
        const auto actual = copy_from_device(output.get(), dtype);
        for (size_t index = 0; index < actual.size(); ++index) {
          const int64_t row = (index / dimension) % dimension;
          const int64_t column = index % dimension;
          const int64_t rows = row == 0 || row == dimension - 1 ? 2 : 3;
          const int64_t columns =
              column == 0 || column == dimension - 1 ? 2 : 3;
          ASSERT_NEAR(actual[index], rows * columns + 0.5f, 0.05f)
              << "index " << index;
        }
      }
    }
  });
  worker.join();
}

void run_convolution(ScalarType dtype) {
  SlimTensor* input = empty_tensor(dtype, {1, 1, 3, 3}, {9, 9, 3, 1});
  SlimTensor* weight = empty_tensor(dtype, {1, 1, 2, 2}, {4, 4, 2, 1});
  SlimTensor* bias = empty_tensor(dtype, {1}, {1});
  copy_to_device(input, encode_values({1, 2, 3, 4, 5, 6, 7, 8, 9}, dtype));
  copy_to_device(weight, encode_values({1, 2, 3, 4}, dtype));
  copy_to_device(bias, encode_values({1}, dtype));

  const int64_t stride[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t dilation[] = {1, 1};
  const int64_t output_padding[] = {0, 0};
  SlimTensor* output = nullptr;
  EXPECT_EQ(
      aoti_torch_musa_convolution(
          input,
          weight,
          &bias,
          stride,
          2,
          padding,
          2,
          dilation,
          2,
          false,
          output_padding,
          2,
          1,
          &output),
      Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_values(copy_from_device(output, dtype), {38, 48, 68, 78});

  delete output;
  delete input;
  delete weight;
  delete bias;
}

void run_convolution_1d() {
  SlimTensor* input = empty_tensor(ScalarType::Float, {1, 1, 4}, {4, 4, 1});
  SlimTensor* weight = empty_tensor(ScalarType::Float, {1, 1, 3}, {3, 3, 1});
  SlimTensor* bias = empty_tensor(ScalarType::Float, {1}, {1});
  copy_to_device(input, encode_values({1, 2, 3, 4}, ScalarType::Float));
  copy_to_device(weight, encode_values({1, 2, 3}, ScalarType::Float));
  copy_to_device(bias, encode_values({1}, ScalarType::Float));

  const int64_t stride[] = {1};
  const int64_t padding[] = {1};
  const int64_t dilation[] = {1};
  const int64_t output_padding[] = {0};
  SlimTensor* output = nullptr;
  EXPECT_EQ(
      aoti_torch_musa_convolution(
          input,
          weight,
          &bias,
          stride,
          1,
          padding,
          1,
          dilation,
          1,
          false,
          output_padding,
          1,
          1,
          &output),
      Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_values(copy_from_device(output, ScalarType::Float), {9, 15, 21, 12});

  delete output;
  delete input;
  delete weight;
  delete bias;
}

void run_convolution_3d(ScalarType dtype) {
  SlimTensor* input = empty_tensor(dtype, {1, 1, 3, 3, 3}, {27, 27, 9, 3, 1});
  SlimTensor* weight = empty_tensor(dtype, {1, 1, 2, 2, 2}, {8, 8, 4, 2, 1});
  SlimTensor* bias = empty_tensor(dtype, {1}, {1});
  copy_to_device(
      input,
      encode_values(
          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
          dtype));
  copy_to_device(weight, encode_values({1, 1, 1, 1, 1, 1, 1, 1}, dtype));
  copy_to_device(bias, encode_values({1}, dtype));

  const int64_t stride[] = {1, 1, 1};
  const int64_t padding[] = {0, 0, 0};
  const int64_t dilation[] = {1, 1, 1};
  const int64_t output_padding[] = {0, 0, 0};
  SlimTensor* output = nullptr;
  EXPECT_EQ(
      aoti_torch_musa_convolution(
          input,
          weight,
          &bias,
          stride,
          3,
          padding,
          3,
          dilation,
          3,
          false,
          output_padding,
          3,
          1,
          &output),
      Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_values(
      copy_from_device(output, dtype), {61, 69, 85, 93, 133, 141, 157, 165});

  delete output;
  delete input;
  delete weight;
  delete bias;
}

void run_convolution_transposed_2d(ScalarType dtype) {
  SlimTensor* input = empty_tensor(dtype, {1, 1, 2, 2}, {4, 4, 2, 1});
  SlimTensor* weight = empty_tensor(dtype, {1, 1, 2, 2}, {4, 4, 2, 1});
  SlimTensor* bias = empty_tensor(dtype, {1}, {1});
  copy_to_device(input, encode_values({1, 2, 3, 4}, dtype));
  copy_to_device(weight, encode_values({1, 1, 1, 1}, dtype));
  copy_to_device(bias, encode_values({1}, dtype));

  const int64_t stride[] = {2, 2};
  const int64_t padding[] = {0, 0};
  const int64_t dilation[] = {1, 1};
  const int64_t output_padding[] = {0, 0};
  SlimTensor* output = nullptr;
  EXPECT_EQ(
      aoti_torch_musa_convolution(
          input,
          weight,
          &bias,
          stride,
          2,
          padding,
          2,
          dilation,
          2,
          true,
          output_padding,
          2,
          1,
          &output),
      Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_values(
      copy_from_device(output, dtype),
      {2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 5, 5});

  delete output;
  delete input;
  delete weight;
  delete bias;
}

void run_convolution_transposed_1d(ScalarType dtype, bool dilated) {
  SlimTensor* input = empty_tensor(dtype, {1, 1, 3}, {3, 3, 1});
  SlimTensor* weight = empty_tensor(dtype, {1, 1, 3}, {3, 3, 1});
  copy_to_device(input, encode_values({1, 2, 3}, dtype));
  copy_to_device(weight, encode_values({1, 1, 1}, dtype));

  const int64_t stride[] = {dilated ? 1 : 2};
  const int64_t padding[] = {dilated ? 2 : 1};
  const int64_t dilation[] = {dilated ? 2 : 1};
  const int64_t output_padding[] = {1};
  SlimTensor* output = nullptr;
  EXPECT_EQ(
      aoti_torch_musa_convolution(
          input,
          weight,
          nullptr,
          stride,
          1,
          padding,
          1,
          dilation,
          1,
          true,
          output_padding,
          1,
          1,
          &output),
      Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_values(
      copy_from_device(output, dtype),
      dilated ? std::vector<float>{4, 2, 4, 2}
              : std::vector<float>{1, 3, 2, 5, 3, 3});

  delete output;
  delete input;
  delete weight;
}

void run_convolution_transposed_3d(ScalarType dtype) {
  SlimTensor* input = empty_tensor(dtype, {1, 1, 2, 2, 2}, {8, 8, 4, 2, 1});
  SlimTensor* weight = empty_tensor(dtype, {1, 1, 2, 2, 2}, {8, 8, 4, 2, 1});
  const std::vector<float> input_values{1, 2, 3, 4, 5, 6, 7, 8};
  copy_to_device(input, encode_values(input_values, dtype));
  copy_to_device(weight, encode_values(std::vector<float>(8, 1), dtype));

  const int64_t stride[] = {2, 2, 2};
  const int64_t padding[] = {0, 0, 0};
  const int64_t dilation[] = {1, 1, 1};
  const int64_t output_padding[] = {0, 0, 0};
  SlimTensor* output = nullptr;
  EXPECT_EQ(
      aoti_torch_musa_convolution(
          input,
          weight,
          nullptr,
          stride,
          3,
          padding,
          3,
          dilation,
          3,
          true,
          output_padding,
          3,
          1,
          &output),
      Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float> expected;
  for (size_t depth = 0; depth < 4; ++depth) {
    for (size_t height = 0; height < 4; ++height) {
      for (size_t width = 0; width < 4; ++width) {
        expected.push_back(
            input_values[(depth / 2) * 4 + (height / 2) * 2 + width / 2]);
      }
    }
  }
  expect_values(copy_from_device(output, dtype), expected);

  delete output;
  delete input;
  delete weight;
}

void run_convolution_transposed_grouped() {
  SlimTensor* input = empty_tensor(ScalarType::Float, {1, 2, 2, 2}, {8, 4, 2, 1});
  SlimTensor* weight = empty_tensor(ScalarType::Float, {2, 1, 2, 2}, {4, 4, 2, 1});
  copy_to_device(
      input, encode_values({1, 2, 3, 4, 5, 6, 7, 8}, ScalarType::Float));
  copy_to_device(
      weight, encode_values({1, 1, 1, 1, 2, 2, 2, 2}, ScalarType::Float));

  const int64_t stride[] = {2, 2};
  const int64_t padding[] = {0, 0};
  const int64_t dilation[] = {1, 1};
  const int64_t output_padding[] = {0, 0};
  SlimTensor* output = nullptr;
  EXPECT_EQ(
      aoti_torch_musa_convolution(
          input,
          weight,
          nullptr,
          stride,
          2,
          padding,
          2,
          dilation,
          2,
          true,
          output_padding,
          2,
          2,
          &output),
      Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_values(
      copy_from_device(output, ScalarType::Float),
      {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4,
       10, 10, 12, 12, 10, 10, 12, 12, 14, 14, 16, 16, 14, 14, 16, 16});

  delete output;
  delete input;
  delete weight;
}

void run_convolution_depthwise(ScalarType dtype) {
  SlimTensor* input = empty_tensor(dtype, {1, 2, 3, 3}, {18, 9, 3, 1});
  SlimTensor* weight = empty_tensor(dtype, {2, 1, 2, 2}, {4, 4, 2, 1});
  copy_to_device(
      input,
      encode_values(
          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
          dtype));
  copy_to_device(weight, encode_values({1, 0, 0, 1, 2, 0, 0, 2}, dtype));

  const int64_t stride[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t dilation[] = {1, 1};
  const int64_t output_padding[] = {0, 0};
  SlimTensor* output = nullptr;
  EXPECT_EQ(
      aoti_torch_musa_convolution(
          input,
          weight,
          nullptr,
          stride,
          2,
          padding,
          2,
          dilation,
          2,
          false,
          output_padding,
          2,
          2,
          &output),
      Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_values(
      copy_from_device(output, dtype), {6, 8, 12, 14, 48, 52, 60, 64});

  delete output;
  delete input;
  delete weight;
}

void run_convolution_noncontiguous_input(ScalarType dtype) {
  SlimTensor* input = empty_tensor(dtype, {1, 1, 3, 3}, {9, 9, 1, 3});
  SlimTensor* weight = empty_tensor(dtype, {1, 1, 2, 2}, {4, 4, 2, 1});
  SlimTensor* bias = empty_tensor(dtype, {1}, {1});
  copy_to_device(input, encode_values({1, 4, 7, 2, 5, 8, 3, 6, 9}, dtype));
  copy_to_device(weight, encode_values({1, 2, 3, 4}, dtype));
  copy_to_device(bias, encode_values({1}, dtype));

  const int64_t stride[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t dilation[] = {1, 1};
  const int64_t output_padding[] = {0, 0};
  SlimTensor* output = nullptr;
  EXPECT_EQ(
      aoti_torch_musa_convolution(
          input,
          weight,
          &bias,
          stride,
          2,
          padding,
          2,
          dilation,
          2,
          false,
          output_padding,
          2,
          1,
          &output),
      Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_values(copy_from_device(output, dtype), {38, 48, 68, 78});

  delete output;
  delete input;
  delete weight;
  delete bias;
}

void run_convolution_channels_last(ScalarType dtype) {
  SlimTensor* input = empty_tensor(dtype, {1, 2, 3, 3}, {18, 1, 6, 2});
  SlimTensor* weight = empty_tensor(dtype, {1, 2, 2, 2}, {8, 4, 2, 1});
  copy_to_device(
      input,
      encode_values(
          {1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9, 1},
          dtype));
  copy_to_device(weight, encode_values({1, 2, 3, 4, 1, 1, 1, 1}, dtype));

  const int64_t stride[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t dilation[] = {1, 1};
  const int64_t output_padding[] = {0, 0};
  SlimTensor* output = nullptr;
  EXPECT_EQ(
      aoti_torch_musa_convolution(
          input,
          weight,
          nullptr,
          stride,
          2,
          padding,
          2,
          dilation,
          2,
          false,
          output_padding,
          2,
          1,
          &output),
      Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_TRUE(output->is_contiguous());
  expect_values(copy_from_device(output, dtype), {41, 51, 71, 81});

  delete output;
  delete input;
  delete weight;
}

TEST_F(MusaConvolutionTest, ForwardDtypes) {
  for (ScalarType dtype : {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    run_convolution(dtype);
  }
}

TEST_F(MusaConvolutionTest, BFloat16BiasBeforeOutputRounding) {
  const auto dtype = ScalarType::BFloat16;
  SlimTensor* input = empty_tensor(dtype, {1, 2, 1, 1}, {2, 1, 1, 1});
  SlimTensor* weight = empty_tensor(dtype, {2, 2, 1, 1}, {2, 1, 1, 1});
  SlimTensor* bias = empty_tensor(dtype, {2}, {1});
  copy_to_device(input, encode_values({1.0f, 0.00390625f}, dtype));
  copy_to_device(weight, encode_values({1.0f, 1.0f, 1.0f, 1.0f}, dtype));
  copy_to_device(bias, encode_values({-1.0f, -1.0f}, dtype));
  const int64_t stride[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t dilation[] = {1, 1};
  SlimTensor* output = nullptr;
  ASSERT_EQ(
      aoti_torch_musa_convolution(
          input,
          weight,
          &bias,
          stride,
          2,
          padding,
          2,
          dilation,
          2,
          false,
          padding,
          2,
          1,
          &output),
      Error::Ok);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const auto values = copy_from_device(output, dtype);
  ASSERT_EQ(values.size(), 2);
  EXPECT_FLOAT_EQ(values[0], 0.00390625f);
  EXPECT_FLOAT_EQ(values[1], 0.00390625f);
  delete output;
  delete bias;
  delete weight;
  delete input;
}

TEST_F(MusaConvolutionTest, ForwardNd) {
  run_convolution_1d();
}

TEST_F(MusaConvolutionTest, Forward3dDtypes) {
  for (ScalarType dtype :
       {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    run_convolution_3d(dtype);
  }
}

TEST_F(MusaConvolutionTest, Transposed2dDtypes) {
  for (ScalarType dtype : {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    run_convolution_transposed_2d(dtype);
  }
}

TEST_F(MusaConvolutionTest, Transposed1dDtypes) {
  for (ScalarType dtype : {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    run_convolution_transposed_1d(dtype, false);
    run_convolution_transposed_1d(dtype, true);
  }
}

TEST_F(MusaConvolutionTest, Transposed3dDtypes) {
  for (ScalarType dtype : {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    run_convolution_transposed_3d(dtype);
  }
}

TEST_F(MusaConvolutionTest, TransposedGrouped) {
  run_convolution_transposed_grouped();
}

TEST_F(MusaConvolutionTest, DepthwiseDtypes) {
  for (ScalarType dtype : {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    run_convolution_depthwise(dtype);
  }
}

TEST_F(MusaConvolutionTest, NonContiguousInput) {
  for (ScalarType dtype : {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    run_convolution_noncontiguous_input(dtype);
  }
}

TEST_F(MusaConvolutionTest, ChannelsLastInput) {
  for (ScalarType dtype : {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    run_convolution_channels_last(dtype);
  }
}

}
