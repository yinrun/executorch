#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/cuda/runtime/shims/musa_vision_layer_norm.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/platform/platform.h>

namespace {

namespace slim = executorch::backends::aoti::slim;
using executorch::backends::cuda::aoti_torch_musa_vision_layer_norm_out;
using executorch::backends::cuda::CUDAStreamGuard;
using executorch::runtime::Error;
using slim::SlimTensor;
using slim::c10::ScalarType;

SlimTensor tensor(
    std::initializer_list<int64_t> shape,
    ScalarType dtype = ScalarType::Float) {
  return slim::empty(shape, dtype, slim::c10::Device(slim::c10::kMUSA, 0));
}

void upload(SlimTensor& destination, const std::vector<float>& values) {
  ASSERT_EQ(destination.nbytes(), values.size() * sizeof(float));
  ASSERT_EQ(
      cudaMemcpy(
          destination.data_ptr(), values.data(), destination.nbytes(), cudaMemcpyHostToDevice),
      cudaSuccess);
}

std::vector<float> read(const SlimTensor& source) {
  std::vector<float> values(source.numel());
  EXPECT_EQ(
      cudaMemcpy(values.data(), source.data_ptr(), source.nbytes(), cudaMemcpyDeviceToHost),
      cudaSuccess);
  return values;
}

class MusaVisionLayerNormTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "MUSA device is not available";
    }
  }
};

TEST_F(MusaVisionLayerNormTest, BothGridsAffineAndInputPreservation) {
  for (int64_t rows : {800, 1200}) {
    auto input = tensor({rows, 1024});
    auto output = tensor({rows, 1024});
    auto weight = tensor({1024});
    auto bias = tensor({1024});
    std::vector<float> values(rows * 1024);
    std::vector<float> gamma(1024), beta(1024);
    for (int64_t column = 0; column < 1024; ++column) {
      gamma[column] = 0.25f + static_cast<float>(column % 7) * 0.125f;
      beta[column] = static_cast<float>(column % 5) - 2;
    }
    for (int64_t index = 0; index < rows * 1024; ++index) {
      values[index] = static_cast<float>(index % 17) - 8;
    }
    upload(input, values);
    upload(weight, gamma);
    upload(bias, beta);
    ASSERT_EQ(
        aoti_torch_musa_vision_layer_norm_out(&output, &input, &weight, &bias, 1e-6),
        Error::Ok);
    const auto actual = read(output);
    for (int64_t row = 0; row < rows; ++row) {
      double mean = 0;
      double variance = 0;
      for (int64_t column = 0; column < 1024; ++column) {
        mean += values[row * 1024 + column] / 1024.0;
      }
      for (int64_t column = 0; column < 1024; ++column) {
        const double difference = values[row * 1024 + column] - mean;
        variance += difference * difference / 1024.0;
      }
      for (int64_t column = 0; column < 1024; ++column) {
        const double expected = (values[row * 1024 + column] - mean) /
                std::sqrt(variance + 1e-6) * gamma[column] + beta[column];
        ASSERT_NEAR(actual[row * 1024 + column], expected, 2e-6);
      }
    }
    EXPECT_EQ(read(input), values);
    EXPECT_EQ(read(weight), gamma);
    EXPECT_EQ(read(bias), beta);
  }
}

TEST_F(MusaVisionLayerNormTest, OffsetDisjointStorageOnExplicitStream) {
  auto pool = tensor({2 * 800 * 1024 + 7});
  auto input = pool.as_strided({800, 1024}, {1024, 1}, 3);
  auto output = pool.as_strided({800, 1024}, {1024, 1}, 800 * 1024 + 5);
  auto weight = tensor({1024});
  auto bias = tensor({1024});
  upload(input, std::vector<float>(800 * 1024, 7));
  upload(weight, std::vector<float>(1024, 3));
  upload(bias, std::vector<float>(1024, 0.125f));
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  {
    auto guard = CUDAStreamGuard::create(stream, 0);
    ASSERT_TRUE(guard.ok());
    ASSERT_EQ(
        aoti_torch_musa_vision_layer_norm_out(&output, &input, &weight, &bias, 1e-6),
        Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(read(output), std::vector<float>(800 * 1024, 0.125f));
    EXPECT_EQ(read(input), std::vector<float>(800 * 1024, 7));
  }
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(MusaVisionLayerNormTest, RejectsOutputAliasingAnyInput) {
  auto input = tensor({800, 1024});
  auto output = tensor({800, 1024});
  auto weight = tensor({1024});
  auto bias = tensor({1024});
  EXPECT_EQ(
      aoti_torch_musa_vision_layer_norm_out(&input, &input, &weight, &bias, 1e-6),
      Error::InvalidArgument);
  auto alias = output.as_strided({1024}, {1}, 3);
  EXPECT_EQ(
      aoti_torch_musa_vision_layer_norm_out(&output, &input, &alias, &bias, 1e-6),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_musa_vision_layer_norm_out(&output, &input, &weight, &alias, 1e-6),
      Error::InvalidArgument);
}

TEST_F(MusaVisionLayerNormTest, RejectsInvalidShapeDtypeStrideAndEpsilon) {
  auto input = tensor({800, 1024});
  auto output = tensor({800, 1024});
  auto weight = tensor({1024});
  auto bias = tensor({1024});
  auto wrong_shape = tensor({800, 512});
  auto wrong_weight = tensor({1023});
  auto wrong_dtype = tensor({800, 1024}, ScalarType::BFloat16);
  auto pool = tensor({800, 2048});
  auto strided = pool.as_strided({800, 1024}, {2048, 2}, 0);
  for (auto* invalid : {&wrong_shape, &wrong_dtype, &strided}) {
    EXPECT_EQ(
        aoti_torch_musa_vision_layer_norm_out(&output, invalid, &weight, &bias, 1e-6),
        Error::InvalidArgument);
    EXPECT_EQ(
        aoti_torch_musa_vision_layer_norm_out(invalid, &input, &weight, &bias, 1e-6),
        Error::InvalidArgument);
  }
  EXPECT_EQ(
      aoti_torch_musa_vision_layer_norm_out(&output, &input, &wrong_weight, &bias, 1e-6),
      Error::InvalidArgument);
  for (double epsilon : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_EQ(
        aoti_torch_musa_vision_layer_norm_out(&output, &input, &weight, &bias, epsilon),
        Error::InvalidArgument);
  }
  EXPECT_EQ(
      aoti_torch_musa_vision_layer_norm_out(&output, nullptr, &weight, &bias, 1e-6),
      Error::InvalidArgument);
}

}
