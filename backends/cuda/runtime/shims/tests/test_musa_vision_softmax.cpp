#include <gtest/gtest.h>

#include <vector>

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/cuda/runtime/shims/musa_vision_softmax.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/platform/platform.h>

namespace {

namespace slim = executorch::backends::aoti::slim;
using executorch::backends::cuda::aoti_torch_musa_vision_softmax_out;
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
      cudaMemcpy(
          values.data(), source.data_ptr(), source.nbytes(), cudaMemcpyDeviceToHost),
      cudaSuccess);
  return values;
}

class MusaVisionSoftmaxTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "MUSA device is not available";
    }
  }
};

TEST_F(MusaVisionSoftmaxTest, BothGridsUniformAndInputPreservation) {
  for (int64_t length : {800, 1200}) {
    auto input = tensor({16, length, length});
    auto output = tensor({16, length, length});
    ASSERT_EQ(
        cudaMemsetAsync(input.data_ptr(), 0, input.nbytes(), nullptr), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    for (int repeat = 0; repeat < 2; ++repeat) {
      ASSERT_EQ(aoti_torch_musa_vision_softmax_out(&output, &input), Error::Ok);
      for (auto probability : read(output)) {
        ASSERT_NEAR(probability, 1.0 / length, 1e-9);
      }
    }
    EXPECT_EQ(read(input), std::vector<float>(input.numel(), 0));
  }
}

TEST_F(MusaVisionSoftmaxTest, OffsetDisjointStorageOnExplicitStream) {
  const int64_t elements = 16 * 800 * 800;
  auto pool = tensor({2 * elements + 9});
  auto input = pool.as_strided({16, 800, 800}, {640000, 800, 1}, 3);
  auto output = pool.as_strided({16, 800, 800}, {640000, 800, 1}, elements + 7);
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  {
    auto guard = CUDAStreamGuard::create(stream, 0);
    ASSERT_TRUE(guard.ok());
    ASSERT_EQ(cudaMemsetAsync(input.data_ptr(), 0, input.nbytes(), stream), cudaSuccess);
    ASSERT_EQ(aoti_torch_musa_vision_softmax_out(&output, &input), Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (auto probability : read(output)) {
      ASSERT_NEAR(probability, 1.0 / 800, 1e-9);
    }
    EXPECT_EQ(read(input), std::vector<float>(elements, 0));
  }
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(MusaVisionSoftmaxTest, RejectsAliasingIncludingPartialOverlap) {
  const int64_t elements = 16 * 800 * 800;
  auto pool = tensor({elements + 3});
  auto input = pool.as_strided({16, 800, 800}, {640000, 800, 1}, 0);
  auto output = pool.as_strided({16, 800, 800}, {640000, 800, 1}, 1);
  EXPECT_EQ(
      aoti_torch_musa_vision_softmax_out(&input, &input), Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_musa_vision_softmax_out(&output, &input), Error::InvalidArgument);
}

TEST_F(MusaVisionSoftmaxTest, RejectsInvalidContract) {
  auto input = tensor({16, 800, 800});
  auto output = tensor({16, 800, 800});
  auto wrong_shape = tensor({16, 800, 799});
  auto wrong_heads = tensor({1, 800, 800});
  auto wrong_dtype = tensor({16, 800, 800}, ScalarType::BFloat16);
  auto wrong_grid = tensor({16, 1200, 1200});
  auto strided = input.as_strided({16, 800, 800}, {640000, 1, 800}, 0);
  for (auto* invalid : {
           &wrong_shape, &wrong_heads, &wrong_dtype, &wrong_grid, &strided}) {
    EXPECT_EQ(
        aoti_torch_musa_vision_softmax_out(&output, invalid), Error::InvalidArgument);
    EXPECT_EQ(
        aoti_torch_musa_vision_softmax_out(invalid, &input), Error::InvalidArgument);
  }
  EXPECT_EQ(
      aoti_torch_musa_vision_softmax_out(&output, nullptr), Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_musa_vision_softmax_out(nullptr, &input), Error::InvalidArgument);
}

}
