#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/platform/platform.h>

namespace executorch::backends::cuda {
extern "C" {
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_musa_convolution(
    SlimTensor*,
    SlimTensor*,
    SlimTensor*,
    const int64_t*,
    int64_t,
    const int64_t*,
    int64_t,
    const int64_t*,
    int64_t,
    bool,
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
          bias,
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
          bias,
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
          bias,
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

TEST_F(MusaConvolutionTest, ForwardNd) {
  run_convolution_1d();
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
