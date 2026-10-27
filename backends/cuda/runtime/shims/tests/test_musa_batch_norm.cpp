#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/musa_batch_norm.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/platform/platform.h>

namespace {

namespace slim = executorch::backends::aoti::slim;
using executorch::backends::aoti::slim::SlimTensor;
using executorch::backends::aoti::slim::c10::DeviceType;
using executorch::backends::aoti::slim::c10::ScalarType;
using executorch::backends::cuda::aoti_torch__alloc_from_pool;
using executorch::backends::cuda::aoti_torch_empty_strided;
using executorch::backends::cuda::aoti_torch_musa_batch_norm;
using executorch::backends::cuda::aoti_torch_musa_batch_norm_out;
using executorch::backends::cuda::CUDAStreamGuard;
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
  cudaStream_t stream = nullptr;

  void SetUp() override {
    et_pal_init();
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "MUSA device is not available";
    }
  }

  void TearDown() override {
    if (stream != nullptr) {
      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
  }
};

TEST_F(MusaBatchNormTest, OutMatchesAllocatingAcrossRanksAndDtypes) {
  ASSERT_EQ(
      musaStreamCreateWithFlags(&stream, musaStreamNonBlocking), musaSuccess);
  auto guard = CUDAStreamGuard::create(stream, 0);
  ASSERT_TRUE(guard.ok());
  std::unique_ptr<SlimTensor> weight(make_tensor({2}, {2.0f, 0.5f}));
  std::unique_ptr<SlimTensor> bias(make_tensor({2}, {1.0f, -1.0f}));
  std::unique_ptr<SlimTensor> mean(make_tensor({2}, {1.0f, 2.0f}));
  std::unique_ptr<SlimTensor> variance(make_tensor({2}, {4.0f, 9.0f}));
  for (auto dtype :
       {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    for (const std::vector<int64_t>& sizes :
         {std::vector<int64_t>{2, 2},
          {1, 2, 2},
          {1, 2, 1, 2},
          {1, 2, 1, 1, 2}}) {
      std::vector<int64_t> strides(sizes.size(), 1);
      for (size_t dimension = sizes.size() - 1; dimension > 0; --dimension) {
        strides[dimension - 1] = strides[dimension] * sizes[dimension];
      }
      auto pool = slim::empty(
          {128}, ScalarType::Byte, slim::c10::Device(slim::c10::kMUSA, 0));
      SlimTensor* raw_input = nullptr;
      SlimTensor* raw_output = nullptr;
      ASSERT_EQ(
          aoti_torch__alloc_from_pool(
              &pool,
              0,
              static_cast<int32_t>(dtype),
              sizes.size(),
              sizes.data(),
              strides.data(),
              &raw_input),
          Error::Ok);
      ASSERT_EQ(
          aoti_torch__alloc_from_pool(
              &pool,
              32,
              static_cast<int32_t>(dtype),
              sizes.size(),
              sizes.data(),
              strides.data(),
              &raw_output),
          Error::Ok);
      std::unique_ptr<SlimTensor> input(raw_input);
      std::unique_ptr<SlimTensor> output(raw_output);
      std::vector<uint8_t> initial(128, 0xA5);
      const float one = 1.0f;
      const uint16_t low_precision_one =
          dtype == ScalarType::Half ? 0x3C00 : 0x3F80;
      for (size_t index = 0; index < input->numel(); ++index) {
        std::memcpy(
            initial.data() + index * input->itemsize(),
            dtype == ScalarType::Float
                ? static_cast<const void*>(&one)
                : static_cast<const void*>(&low_precision_one),
            input->itemsize());
      }
      ASSERT_EQ(
          cudaMemcpy(
              pool.data_ptr(),
              initial.data(),
              initial.size(),
              cudaMemcpyHostToDevice),
          cudaSuccess);
      SlimTensor* raw_reference = nullptr;
      ASSERT_EQ(
          aoti_torch_musa_batch_norm(
              input.get(),
              weight.get(),
              bias.get(),
              mean.get(),
              variance.get(),
              1e-5,
              &raw_reference),
          Error::Ok);
      std::unique_ptr<SlimTensor> reference(raw_reference);
      const auto pointer = output->data_ptr();
      for (int repeat = 0; repeat < 3; ++repeat) {
        ASSERT_EQ(
            aoti_torch_musa_batch_norm_out(
                output.get(),
                input.get(),
                weight.get(),
                bias.get(),
                mean.get(),
                variance.get(),
                1e-5),
            Error::Ok);
      }
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      EXPECT_EQ(output->data_ptr(), pointer);
      EXPECT_EQ(output->storage(), pool.storage());
      EXPECT_EQ(output->storage_offset(), 32 / output->itemsize());
      auto host = slim::empty({128}, ScalarType::Byte);
      host.copy_(pool);
      auto reference_host = slim::empty(
          {static_cast<int64_t>(reference->nbytes())}, ScalarType::Byte);
      SlimTensor reference_bytes(
          reference->storage(),
          {static_cast<int64_t>(reference->nbytes())},
          {1},
          ScalarType::Byte);
      reference_host.copy_(reference_bytes);
      auto* actual = static_cast<const uint8_t*>(host.data_ptr());
      EXPECT_EQ(
          std::memcmp(
              actual + 32, reference_host.data_ptr(), reference->nbytes()),
          0);
      for (size_t index = 0; index < initial.size(); ++index) {
        if (index < 32 || index >= 32 + output->nbytes()) {
          EXPECT_EQ(actual[index], initial[index]);
        }
      }
    }
  }
}

TEST_F(MusaBatchNormTest, OutSupportsNonContiguousInputs) {
  std::unique_ptr<SlimTensor> input(make_tensor_with_strides(
      {1, 2, 1, 2}, {4, 1, 2, 2}, {1.0f, 3.0f, 2.0f, 6.0f}));
  std::unique_ptr<SlimTensor> weight(
      make_tensor_with_strides({2}, {2}, {2.0f, 0.5f}));
  std::unique_ptr<SlimTensor> bias(make_tensor({2}, {1.0f, -1.0f}));
  std::unique_ptr<SlimTensor> mean(make_tensor({2}, {1.0f, 2.0f}));
  std::unique_ptr<SlimTensor> variance(make_tensor({2}, {4.0f, 9.0f}));
  std::unique_ptr<SlimTensor> output(make_tensor({1, 2, 1, 2}, {0, 0, 0, 0}));
  ASSERT_EQ(
      aoti_torch_musa_batch_norm_out(
          output.get(),
          input.get(),
          weight.get(),
          bias.get(),
          mean.get(),
          variance.get(),
          1e-5),
      Error::Ok);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_values(
      read_tensor(output.get()), {1.0f, 2.9999976f, -1.0f, -0.3333334f});
}

TEST_F(MusaBatchNormTest, OutSupportsDisjointStridedPoolViews) {
  auto pool = slim::empty(
      {128}, ScalarType::Byte, slim::c10::Device(slim::c10::kMUSA, 0));
  const int64_t sizes[] = {1, 2, 1, 2};
  const int64_t input_strides[] = {4, 1, 2, 2};
  const int64_t output_strides[] = {4, 2, 2, 1};
  SlimTensor input(
      pool.storage(),
      slim::IntArrayRef(sizes, 4),
      slim::IntArrayRef(input_strides, 4),
      ScalarType::Float);
  SlimTensor output(
      pool.storage(),
      slim::IntArrayRef(sizes, 4),
      slim::IntArrayRef(output_strides, 4),
      ScalarType::Float,
      8);
  const std::vector<float> physical = {1, 2, 3, 6};
  ASSERT_EQ(
      cudaMemcpy(
          pool.data_ptr(),
          physical.data(),
          physical.size() * sizeof(float),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  std::unique_ptr<SlimTensor> weight(make_tensor({2}, {2, 0.5}));
  std::unique_ptr<SlimTensor> bias(make_tensor({2}, {1, -1}));
  std::unique_ptr<SlimTensor> mean(make_tensor({2}, {1, 2}));
  std::unique_ptr<SlimTensor> variance(make_tensor({2}, {4, 9}));
  ASSERT_EQ(
      aoti_torch_musa_batch_norm_out(
          &output,
          &input,
          weight.get(),
          bias.get(),
          mean.get(),
          variance.get(),
          1e-5),
      Error::Ok);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_values(read_tensor(&output), {1.0f, 2.9999976f, -1.0f, -0.3333334f});
}

TEST_F(MusaBatchNormTest, OutRejectsInvalidDestinations) {
  std::unique_ptr<SlimTensor> input(make_tensor({1, 2, 1, 2}, {1, 3, 2, 6}));
  std::unique_ptr<SlimTensor> weight(make_tensor({2}, {2, 0.5}));
  std::unique_ptr<SlimTensor> bias(make_tensor({2}, {1, -1}));
  std::unique_ptr<SlimTensor> mean(make_tensor({2}, {1, 2}));
  std::unique_ptr<SlimTensor> variance(make_tensor({2}, {4, 9}));
  std::unique_ptr<SlimTensor> wrong_shape(make_tensor({2, 2}, {0, 0, 0, 0}));
  std::unique_ptr<SlimTensor> noncontiguous(
      make_tensor_with_strides({1, 2, 1, 2}, {4, 1, 2, 2}, {0, 0, 0, 0}));
  auto wrong_dtype =
      slim::empty({1, 2, 1, 2}, ScalarType::Half, input->device());
  auto host = slim::empty({1, 2, 1, 2}, ScalarType::Float);
  SlimTensor undefined;
  auto too_small = input->as_strided({1, 2, 1, 2}, {4, 2, 2, 1}, 1);
  for (auto* output :
       {static_cast<SlimTensor*>(nullptr),
        &undefined,
        input.get(),
        wrong_shape.get(),
        noncontiguous.get(),
        &wrong_dtype,
        &host,
        &too_small}) {
    EXPECT_EQ(
        aoti_torch_musa_batch_norm_out(
            output,
            input.get(),
            weight.get(),
            bias.get(),
            mean.get(),
            variance.get(),
            1e-5),
        Error::InvalidArgument);
  }
}

TEST_F(MusaBatchNormTest, OutRejectsOverlappingParameters) {
  std::unique_ptr<SlimTensor> input(make_tensor({1, 2, 1, 2}, {1, 3, 2, 6}));
  std::unique_ptr<SlimTensor> output(make_tensor({1, 2, 1, 2}, {2, 0.5, 0, 0}));
  auto weight = output->as_strided({2}, {1}, 0);
  std::unique_ptr<SlimTensor> bias(make_tensor({2}, {1, -1}));
  std::unique_ptr<SlimTensor> mean(make_tensor({2}, {1, 2}));
  std::unique_ptr<SlimTensor> variance(make_tensor({2}, {4, 9}));
  EXPECT_EQ(
      aoti_torch_musa_batch_norm_out(
          output.get(),
          input.get(),
          &weight,
          bias.get(),
          mean.get(),
          variance.get(),
          1e-5),
      Error::InvalidArgument);
  expect_values(read_tensor(output.get()), {2, 0.5, 0, 0});
}

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
