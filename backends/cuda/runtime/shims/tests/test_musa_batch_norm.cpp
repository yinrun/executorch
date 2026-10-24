#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/platform/platform.h>

namespace executorch::backends::cuda {
extern "C" {
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_musa_batch_norm(
    SlimTensor*,
    SlimTensor*,
    SlimTensor*,
    SlimTensor*,
    SlimTensor*,
    double,
    SlimTensor**);
}
}

namespace {

using executorch::backends::aoti::slim::c10::DeviceType;
using executorch::backends::aoti::slim::c10::ScalarType;
using executorch::backends::aoti::slim::SlimTensor;
using executorch::backends::cuda::aoti_torch_empty_strided;
using executorch::backends::cuda::aoti_torch_musa_batch_norm;
using executorch::runtime::Error;

SlimTensor* make_tensor(
    const std::vector<int64_t>& sizes,
    const std::vector<float>& values) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int64_t index = static_cast<int64_t>(sizes.size()) - 2; index >= 0;
       --index) {
    strides[index] = strides[index + 1] * sizes[index + 1];
  }
  SlimTensor* tensor = nullptr;
  EXPECT_EQ(
      aoti_torch_empty_strided(
          sizes.size(),
          sizes.data(),
          strides.data(),
          static_cast<int32_t>(ScalarType::Float),
          static_cast<int32_t>(DeviceType::CUDA),
          0,
          &tensor),
      Error::Ok);
  EXPECT_NE(tensor, nullptr);
  if (tensor == nullptr) {
    return nullptr;
  }
  EXPECT_EQ(
      cudaMemcpy(
          tensor->data_ptr(),
          values.data(),
          values.size() * sizeof(float),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  return tensor;
}

SlimTensor* make_tensor_with_strides(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    const std::vector<float>& values) {
  int64_t storage_size = 1;
  for (size_t dimension = 0; dimension < sizes.size(); ++dimension) {
    storage_size += (sizes[dimension] - 1) * strides[dimension];
  }
  std::vector<float> storage(storage_size);
  for (int64_t linear_index = 0; linear_index < static_cast<int64_t>(values.size());
       ++linear_index) {
    int64_t remaining = linear_index;
    int64_t storage_offset = 0;
    for (int64_t dimension = static_cast<int64_t>(sizes.size()) - 1;
         dimension >= 0;
         --dimension) {
      const int64_t index = remaining % sizes[dimension];
      remaining /= sizes[dimension];
      storage_offset += index * strides[dimension];
    }
    storage[storage_offset] = values[linear_index];
  }

  SlimTensor* tensor = nullptr;
  EXPECT_EQ(
      aoti_torch_empty_strided(
          sizes.size(),
          sizes.data(),
          strides.data(),
          static_cast<int32_t>(ScalarType::Float),
          static_cast<int32_t>(DeviceType::CUDA),
          0,
          &tensor),
      Error::Ok);
  EXPECT_NE(tensor, nullptr);
  if (tensor == nullptr) {
    return nullptr;
  }
  EXPECT_EQ(
      cudaMemcpy(
          tensor->data_ptr(),
          storage.data(),
          storage.size() * sizeof(float),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  return tensor;
}

std::vector<float> read_tensor(SlimTensor* tensor) {
  std::vector<float> values(tensor->numel());
  EXPECT_EQ(
      cudaMemcpy(
          values.data(),
          tensor->data_ptr(),
          values.size() * sizeof(float),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  return values;
}

void expect_values(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], 1e-4f) << "index " << index;
  }
}

class MusaBatchNormTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "MUSA device is not available";
    }
  }
};

TEST_F(MusaBatchNormTest, SupportsFourDimensionalAffineInference) {
  SlimTensor* input = make_tensor(
      {1, 2, 1, 2},
      {1.0f, 3.0f, 2.0f, 6.0f});
  SlimTensor* weight = make_tensor({2}, {2.0f, 0.5f});
  SlimTensor* bias = make_tensor({2}, {1.0f, -1.0f});
  SlimTensor* running_mean = make_tensor({2}, {1.0f, 2.0f});
  SlimTensor* running_var = make_tensor({2}, {4.0f, 9.0f});
  SlimTensor* output = nullptr;

  EXPECT_EQ(
      aoti_torch_musa_batch_norm(
          input,
          weight,
          bias,
          running_mean,
          running_var,
          1e-5,
          &output),
      Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_values(
      read_tensor(output),
      {1.0f, 2.9999976f, -1.0f, -0.3333334f});

  delete output;
  delete running_var;
  delete running_mean;
  delete bias;
  delete weight;
  delete input;
}

TEST_F(MusaBatchNormTest, SupportsTwoDimensionalNonAffineInference) {
  SlimTensor* input = make_tensor({2, 2}, {1.0f, 4.0f, 2.0f, 8.0f});
  SlimTensor* weight = make_tensor({2}, {1.0f, 1.0f});
  SlimTensor* bias = make_tensor({2}, {0.0f, 0.0f});
  SlimTensor* running_mean = make_tensor({2}, {1.0f, 4.0f});
  SlimTensor* running_var = make_tensor({2}, {1.0f, 4.0f});
  SlimTensor* output = nullptr;

  EXPECT_EQ(
      aoti_torch_musa_batch_norm(
          input,
          weight,
          bias,
          running_mean,
          running_var,
          1e-5,
          &output),
      Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_values(read_tensor(output), {0.0f, 0.0f, 1.0f, 2.0f});

  delete output;
  delete running_var;
  delete running_mean;
  delete bias;
  delete weight;
  delete input;
}

TEST_F(MusaBatchNormTest, SupportsNonContiguousInput) {
  SlimTensor* input = make_tensor_with_strides(
      {1, 2, 1, 2}, {4, 1, 2, 2}, {1.0f, 3.0f, 2.0f, 6.0f});
  SlimTensor* weight = make_tensor({2}, {2.0f, 0.5f});
  SlimTensor* bias = make_tensor({2}, {1.0f, -1.0f});
  SlimTensor* running_mean = make_tensor({2}, {1.0f, 2.0f});
  SlimTensor* running_var = make_tensor({2}, {4.0f, 9.0f});
  SlimTensor* output = nullptr;

  ASSERT_NE(input, nullptr);
  EXPECT_FALSE(input->is_contiguous());
  EXPECT_EQ(
      aoti_torch_musa_batch_norm(
          input,
          weight,
          bias,
          running_mean,
          running_var,
          1e-5,
          &output),
      Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_values(
      read_tensor(output),
      {1.0f, 2.9999976f, -1.0f, -0.3333334f});

  delete output;
  delete running_var;
  delete running_mean;
  delete bias;
  delete weight;
  delete input;
}

TEST_F(MusaBatchNormTest, SupportsPaddedBatchAndStridedParameters) {
  SlimTensor* input = make_tensor_with_strides(
      {2, 2, 1, 2}, {8, 2, 2, 1},
      {1.0f, 3.0f, 2.0f, 6.0f, 3.0f, 1.0f, 6.0f, 2.0f});
  SlimTensor* weight = make_tensor_with_strides({2}, {2}, {2.0f, 0.5f});
  SlimTensor* bias = make_tensor_with_strides({2}, {2}, {1.0f, -1.0f});
  SlimTensor* running_mean = make_tensor_with_strides({2}, {2}, {1.0f, 2.0f});
  SlimTensor* running_var = make_tensor_with_strides({2}, {2}, {4.0f, 9.0f});
  SlimTensor* output = nullptr;

  for (auto* tensor : {input, weight, bias, running_mean, running_var}) {
    ASSERT_NE(tensor, nullptr);
    EXPECT_FALSE(tensor->is_contiguous());
  }
  EXPECT_EQ(
      aoti_torch_musa_batch_norm(
          input, weight, bias, running_mean, running_var, 1e-5, &output),
      Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(output->is_contiguous());
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_values(
      read_tensor(output),
      {1.0f, 2.9999976f, -1.0f, -0.3333334f,
       2.9999976f, 1.0f, -0.3333334f, -1.0f});

  delete output;
  delete running_var;
  delete running_mean;
  delete bias;
  delete weight;
  delete input;
}

} // namespace
