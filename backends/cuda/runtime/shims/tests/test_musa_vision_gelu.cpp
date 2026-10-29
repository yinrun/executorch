#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/cuda/runtime/shims/musa_vision_gelu.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/platform/platform.h>

namespace {

namespace slim = executorch::backends::aoti::slim;
using executorch::backends::cuda::aoti_torch_musa_vision_gelu_out;
using executorch::backends::cuda::CUDAStreamGuard;
using executorch::runtime::Error;
using slim::SlimTensor;
using slim::c10::ScalarType;

SlimTensor tensor(
    std::initializer_list<int64_t> shape,
    ScalarType dtype = ScalarType::Float) {
  return slim::empty(shape, dtype, slim::c10::Device(slim::c10::kMUSA, 0));
}

std::vector<float> read(const SlimTensor& source) {
  std::vector<float> values(source.numel());
  EXPECT_EQ(
      cudaMemcpy(values.data(), source.data_ptr(), source.nbytes(), cudaMemcpyDeviceToHost),
      cudaSuccess);
  return values;
}

class MusaVisionGeluTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "MUSA device is not available";
    }
  }
};

TEST_F(MusaVisionGeluTest, BothModesAllRowsAgainstDoubleFormula) {
  for (int64_t rows : {200, 300, 800, 1200}) {
    auto input = tensor({rows, 4096});
    auto output = tensor({rows, 4096});
    std::vector<float> values(input.numel());
    for (size_t index = 0; index < values.size(); ++index) {
      values[index] = (static_cast<int>(index % 201) - 100) * 0.1f;
    }
    ASSERT_EQ(
        cudaMemcpy(input.data_ptr(), values.data(), input.nbytes(), cudaMemcpyHostToDevice),
        cudaSuccess);
    for (int64_t mode : {0, 1}) {
      ASSERT_EQ(aoti_torch_musa_vision_gelu_out(&output, &input, mode), Error::Ok);
      const auto actual = read(output);
      for (size_t index = 0; index < values.size(); ++index) {
        const double value = values[index];
        const double factor = mode
            ? std::tanh(std::sqrt(2.0 / std::acos(-1.0)) *
                        (value + 0.044715 * value * value * value))
            : std::erf(value / std::sqrt(2.0));
        const double expected = 0.5 * value * (1.0 + factor);
        ASSERT_NEAR(actual[index], expected, 2e-6);
      }
    }
    EXPECT_EQ(read(input), values);
  }
}

TEST_F(MusaVisionGeluTest, OffsetDisjointStorageOnExplicitStream) {
  const int64_t elements = 200 * 4096;
  auto pool = tensor({2 * elements + 9});
  auto input = pool.as_strided({200, 4096}, {4096, 1}, 3);
  auto output = pool.as_strided({200, 4096}, {4096, 1}, elements + 7);
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  {
    auto guard = CUDAStreamGuard::create(stream, 0);
    ASSERT_TRUE(guard.ok());
    ASSERT_EQ(cudaMemsetAsync(input.data_ptr(), 0, input.nbytes(), stream), cudaSuccess);
    for (int64_t mode : {0, 1}) {
      ASSERT_EQ(aoti_torch_musa_vision_gelu_out(&output, &input, mode), Error::Ok);
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      EXPECT_EQ(read(output), std::vector<float>(elements, 0));
    }
    EXPECT_EQ(read(input), std::vector<float>(elements, 0));
  }
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(MusaVisionGeluTest, RejectsAliasingIncludingPartialOverlap) {
  auto pool = tensor({200 * 4096 + 3});
  auto input = pool.as_strided({200, 4096}, {4096, 1}, 0);
  auto output = pool.as_strided({200, 4096}, {4096, 1}, 1);
  EXPECT_EQ(aoti_torch_musa_vision_gelu_out(&input, &input, 1), Error::InvalidArgument);
  EXPECT_EQ(aoti_torch_musa_vision_gelu_out(&output, &input, 0), Error::InvalidArgument);
}

TEST_F(MusaVisionGeluTest, RejectsInvalidContract) {
  auto input = tensor({200, 4096});
  auto output = tensor({200, 4096});
  auto wrong_width = tensor({200, 4095});
  auto wrong_rows = tensor({201, 4096});
  auto wrong_dtype = tensor({200, 4096}, ScalarType::BFloat16);
  auto wrong_grid = tensor({300, 4096});
  auto strided = input.as_strided({200, 4096}, {1, 200}, 0);
  for (auto* invalid : {&wrong_width, &wrong_rows, &wrong_dtype, &wrong_grid, &strided}) {
    EXPECT_EQ(aoti_torch_musa_vision_gelu_out(&output, invalid, 1), Error::InvalidArgument);
    EXPECT_EQ(aoti_torch_musa_vision_gelu_out(invalid, &input, 0), Error::InvalidArgument);
  }
  EXPECT_EQ(aoti_torch_musa_vision_gelu_out(&output, nullptr, 0), Error::InvalidArgument);
  EXPECT_EQ(aoti_torch_musa_vision_gelu_out(nullptr, &input, 1), Error::InvalidArgument);
  for (int64_t mode : {-1, 2}) {
    EXPECT_EQ(aoti_torch_musa_vision_gelu_out(&output, &input, mode), Error::InvalidArgument);
  }
}

}
