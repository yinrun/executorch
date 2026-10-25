#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/musa_activation.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/core/portable_type/bfloat16.h>
#include <executorch/runtime/core/portable_type/half.h>
#include <executorch/runtime/platform/platform.h>

namespace {

using executorch::backends::aoti::slim::SlimTensor;
using executorch::backends::aoti::slim::c10::DeviceType;
using executorch::backends::aoti::slim::c10::ScalarType;
using executorch::backends::cuda::aoti_torch_empty_strided;
using executorch::backends::cuda::aoti_torch_musa_hardsigmoid;
using executorch::backends::cuda::aoti_torch_musa_hardswish;
using executorch::runtime::Error;
using executorch::runtime::etensor::BFloat16;
using executorch::runtime::etensor::Half;

template <typename Element>
void check_values(
    ScalarType dtype,
    int64_t stride,
    int64_t offset,
    bool hard_sigmoid = false) {
  const std::vector<float> values = {-4, -3, -2, -1, 0, 1, 2, 3, 4};
  const std::vector<float> expected = hard_sigmoid
      ? std::vector<
            float>{0, 0, 1.0f / 6, 1.0f / 3, 0.5f, 2.0f / 3, 5.0f / 6, 1, 1}
      : std::vector<float>{
            0, 0, -1.0f / 3, -1.0f / 3, 0, 2.0f / 3, 5.0f / 3, 3, 4};
  const int64_t storage_size = offset + (values.size() - 1) * stride + 1;
  const int64_t storage_stride = 1;
  std::vector<Element> storage(storage_size, Element(0.0f));
  for (size_t index = 0; index < values.size(); ++index) {
    storage[offset + index * stride] = Element(values[index]);
  }
  SlimTensor* raw_input = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1,
          &storage_size,
          &storage_stride,
          static_cast<int32_t>(dtype),
          static_cast<int32_t>(DeviceType::CUDA),
          0,
          &raw_input),
      Error::Ok);
  std::unique_ptr<SlimTensor> owner(raw_input);
  ASSERT_EQ(
      cudaMemcpy(
          owner->data_ptr(),
          storage.data(),
          storage.size() * sizeof(Element),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  auto input = owner->as_strided(
      {static_cast<int64_t>(values.size())}, {stride}, offset);
  SlimTensor* raw_output = nullptr;
  const auto activation =
      hard_sigmoid ? aoti_torch_musa_hardsigmoid : aoti_torch_musa_hardswish;
  ASSERT_EQ(activation(&input, &raw_output), Error::Ok);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::unique_ptr<SlimTensor> output(raw_output);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(output->is_contiguous());
  std::vector<Element> actual(values.size());
  ASSERT_EQ(
      cudaMemcpy(
          actual.data(),
          output->data_ptr(),
          actual.size() * sizeof(Element),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  const float tolerance = dtype == ScalarType::Float ? 1e-6f : 0.01f;
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(static_cast<float>(actual[index]), expected[index], tolerance);
  }
  std::vector<Element> input_after(storage.size());
  ASSERT_EQ(
      cudaMemcpy(
          input_after.data(),
          owner->data_ptr(),
          input_after.size() * sizeof(Element),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (size_t index = 0; index < storage.size(); ++index) {
    EXPECT_FLOAT_EQ(
        static_cast<float>(input_after[index]),
        static_cast<float>(storage[index]));
  }
}

class MusaActivationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "MUSA device is not available";
    }
  }
};

TEST_F(MusaActivationTest, HardswishDtypesAndStridedViews) {
  for (int64_t stride : {1, 2}) {
    check_values<float>(ScalarType::Float, stride, 1);
    check_values<Half>(ScalarType::Half, stride, 1);
    check_values<BFloat16>(ScalarType::BFloat16, stride, 1);
  }
}

TEST_F(MusaActivationTest, EmptyAndScalarInputs) {
  for (const std::vector<int64_t>& sizes :
       {std::vector<int64_t>{0}, std::vector<int64_t>{}}) {
    std::vector<int64_t> strides(sizes.size(), 1);
    SlimTensor* raw_input = nullptr;
    ASSERT_EQ(
        aoti_torch_empty_strided(
            sizes.size(),
            sizes.data(),
            strides.data(),
            static_cast<int32_t>(ScalarType::Float),
            static_cast<int32_t>(DeviceType::CUDA),
            0,
            &raw_input),
        Error::Ok);
    std::unique_ptr<SlimTensor> input(raw_input);
    if (sizes.empty()) {
      const float value = 3.0f;
      ASSERT_EQ(
          cudaMemcpy(
              input->data_ptr(), &value, sizeof(value), cudaMemcpyHostToDevice),
          cudaSuccess);
    }
    SlimTensor* raw_output = nullptr;
    ASSERT_EQ(aoti_torch_musa_hardswish(input.get(), &raw_output), Error::Ok);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::unique_ptr<SlimTensor> output(raw_output);
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->dim(), sizes.size());
    EXPECT_EQ(output->numel(), input->numel());
    if (sizes.empty()) {
      float actual = 0;
      ASSERT_EQ(
          cudaMemcpy(
              &actual,
              output->data_ptr(),
              sizeof(actual),
              cudaMemcpyDeviceToHost),
          cudaSuccess);
      EXPECT_FLOAT_EQ(actual, 3.0f);
    }
  }
}

TEST_F(MusaActivationTest, HardsigmoidDtypesAndStridedViews) {
  for (int64_t stride : {1, 2}) {
    check_values<float>(ScalarType::Float, stride, 1, true);
    check_values<Half>(ScalarType::Half, stride, 1, true);
    check_values<BFloat16>(ScalarType::BFloat16, stride, 1, true);
  }
}

TEST_F(MusaActivationTest, RejectsNullInput) {
  SlimTensor* output = nullptr;
  EXPECT_EQ(
      aoti_torch_musa_hardswish(nullptr, &output), Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_musa_hardsigmoid(nullptr, &output), Error::InvalidArgument);
}

}
