#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/cuda/runtime/shims/musa_reduce.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/core/portable_type/bfloat16.h>
#include <executorch/runtime/platform/platform.h>

namespace {

namespace slim = executorch::backends::aoti::slim;
using executorch::backends::cuda::aoti_torch_musa_bf16_sum_out;
using executorch::backends::cuda::aoti_torch_musa_reduce_out;
using executorch::backends::cuda::CUDAStreamGuard;
using executorch::runtime::Error;
using executorch::runtime::etensor::BFloat16;
using slim::SlimTensor;
using slim::c10::ScalarType;

SlimTensor tensor(
    const std::vector<int64_t>& sizes,
    ScalarType dtype = ScalarType::Float) {
  return slim::empty(
      slim::IntArrayRef(sizes.data(), sizes.size()),
      dtype,
      slim::c10::Device(slim::c10::kMUSA, 0));
}

std::vector<uint8_t> storage_bytes(const SlimTensor& value) {
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint8_t> result(value.storage()->nbytes());
  EXPECT_EQ(
      cudaMemcpy(
          result.data(),
          value.storage()->data(),
          result.size(),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  return result;
}

template <typename Element>
std::vector<Element> read(const SlimTensor& value) {
  const auto storage = storage_bytes(value);
  std::vector<Element> result(value.numel());
  std::memcpy(
      result.data(),
      storage.data() + value.storage_offset() * sizeof(Element),
      result.size() * sizeof(Element));
  return result;
}

Error run(
    SlimTensor& output,
    SlimTensor& input,
    int64_t axis,
    bool keepdim,
    bool mean) {
  return input.dtype() == ScalarType::Float
      ? aoti_torch_musa_reduce_out(&output, &input, axis, keepdim, mean)
      : aoti_torch_musa_bf16_sum_out(&output, &input, axis, keepdim, mean);
}

template <typename Element>
void check_last_axis(
    const std::vector<int64_t>& shape,
    ScalarType dtype,
    bool mean,
    bool keepdim) {
  auto input = tensor(shape, dtype);
  std::vector<Element> values(input.numel());
  for (size_t index = 0; index < values.size(); ++index) {
    values[index] =
        Element(static_cast<float>(static_cast<int>(index % 19) - 9));
  }
  ASSERT_EQ(
      cudaMemcpy(
          input.data_ptr(),
          values.data(),
          values.size() * sizeof(Element),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  const auto original = storage_bytes(input);
  auto reduced = shape;
  if (keepdim) {
    reduced.back() = 1;
  } else {
    reduced.pop_back();
  }
  auto output = tensor(reduced, dtype);
  ASSERT_EQ(run(output, input, -1, keepdim, mean), Error::Ok);
  const auto actual = read<Element>(output);
  for (size_t row = 0; row < actual.size(); ++row) {
    float expected = 0;
    for (int64_t column = 0; column < shape.back(); ++column) {
      expected += static_cast<float>(values[row * shape.back() + column]);
    }
    if (mean) {
      expected /= shape.back();
    }
    EXPECT_EQ(
        static_cast<float>(actual[row]), static_cast<float>(Element(expected)));
  }
  EXPECT_EQ(storage_bytes(input), original);
}

class MusaReduceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "MUSA device is not available";
    }
  }
};

TEST_F(MusaReduceTest, Fp32SumMeanRanksKeepdimAndScalarOutputs) {
  for (const std::vector<int64_t>& shape : std::vector<std::vector<int64_t>>{
           {128}, {3, 128}, {1, 2, 1, 256}, {1, 1, 1, 1, 1, 1, 2, 1024}}) {
    for (bool mean : {false, true}) {
      for (bool keepdim : {false, true}) {
        check_last_axis<float>(shape, ScalarType::Float, mean, keepdim);
      }
    }
  }
}

TEST_F(MusaReduceTest, Bf16SumPreservesOutputTypeAndKeepdim) {
  for (bool keepdim : {false, true}) {
    check_last_axis<BFloat16>(
        {1, 1, 16, 128}, ScalarType::BFloat16, false, keepdim);
  }
}

TEST_F(MusaReduceTest, Fp32GdnNonLastAxisSum) {
  auto input = tensor({1, 16, 128, 128});
  std::vector<float> values(input.numel());
  for (size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<float>(index % 128);
  }
  ASSERT_EQ(
      cudaMemcpy(
          input.data_ptr(),
          values.data(),
          input.nbytes(),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  for (bool keepdim : {false, true}) {
    auto output = tensor(
        keepdim ? std::vector<int64_t>{1, 16, 1, 128}
                : std::vector<int64_t>{1, 16, 128});
    ASSERT_EQ(run(output, input, 2, keepdim, false), Error::Ok);
    const auto actual = read<float>(output);
    for (size_t index = 0; index < actual.size(); ++index) {
      EXPECT_EQ(actual[index], static_cast<float>((index % 128) * 128));
    }
  }
}

TEST_F(MusaReduceTest, TransposeSteppedBroadcastAndOffsetInputs) {
  for (const auto& strides :
       std::vector<std::vector<int64_t>>{{128, 1}, {1, 3}, {256, 2}, {0, 1}}) {
    auto owner = tensor({1024});
    std::vector<float> values(1024);
    for (size_t index = 0; index < values.size(); ++index) {
      values[index] = static_cast<int>(index % 19) - 9;
    }
    ASSERT_EQ(
        cudaMemcpy(
            owner.data_ptr(),
            values.data(),
            owner.nbytes(),
            cudaMemcpyHostToDevice),
        cudaSuccess);
    auto input = owner.as_strided({3, 128}, {strides[0], strides[1]}, 1);
    auto output = tensor({3});
    const auto original = storage_bytes(owner);
    for (bool mean : {false, true}) {
      ASSERT_EQ(run(output, input, 1, false, mean), Error::Ok);
      auto actual = read<float>(output);
      for (int64_t row = 0; row < 3; ++row) {
        float expected = 0;
        for (int64_t column = 0; column < 128; ++column) {
          expected += values[1 + row * strides[0] + column * strides[1]];
        }
        EXPECT_EQ(actual[row], mean ? expected / 128 : expected);
      }
      EXPECT_EQ(storage_bytes(owner), original);
    }
  }
}

TEST_F(MusaReduceTest, Bf16SteppedInputAndDisjointSharedStorage) {
  auto pool = tensor({5000}, ScalarType::BFloat16);
  std::vector<BFloat16> values(5000, BFloat16(1.0f));
  ASSERT_EQ(
      cudaMemcpy(
          pool.data_ptr(),
          values.data(),
          pool.nbytes(),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  auto input = pool.as_strided({1, 1, 16, 128}, {4096, 4096, 256, 2}, 1);
  auto output = pool.as_strided({1, 1, 16}, {16, 16, 1}, 4501);
  ASSERT_EQ(run(output, input, 3, false, false), Error::Ok);
  for (const auto value : read<BFloat16>(output)) {
    EXPECT_EQ(static_cast<float>(value), 128.0f);
  }
  const auto after = read<BFloat16>(pool);
  for (size_t index = 0; index < after.size(); ++index) {
    EXPECT_EQ(
        static_cast<float>(after[index]),
        index >= 4501 && index < 4517 ? 128.0f : 1.0f);
  }
}

TEST_F(MusaReduceTest, RejectsPhysicalAliasesAcrossStorageWrappers) {
  auto input = tensor({3, 128});
  ASSERT_EQ(musaMemset(input.data_ptr(), 0x5A, input.nbytes()), musaSuccess);
  const auto original = storage_bytes(input);
  auto output = input.as_strided({3}, {1}, 1);
  EXPECT_EQ(run(output, input, 1, false, false), Error::InvalidArgument);
  SlimTensor wrapped(
      slim::Storage(new slim::MaybeOwningStorage(
          input.device(), static_cast<float*>(input.data_ptr()) + 1, 12)),
      {3},
      {1},
      ScalarType::Float);
  EXPECT_EQ(run(wrapped, input, 1, false, false), Error::InvalidArgument);
  EXPECT_EQ(storage_bytes(input), original);
}

TEST_F(MusaReduceTest, Fp32DisjointSharedPoolPreservesOtherValues) {
  auto pool = tensor({400});
  std::vector<float> values(400, 1.0f);
  ASSERT_EQ(
      cudaMemcpy(
          pool.data_ptr(),
          values.data(),
          pool.nbytes(),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  auto input = pool.as_strided({3, 128}, {128, 1}, 1);
  auto output = pool.as_strided({3}, {1}, 390);
  ASSERT_EQ(run(output, input, 1, false, false), Error::Ok);
  auto actual = read<float>(pool);
  for (size_t index = 0; index < actual.size(); ++index) {
    EXPECT_EQ(actual[index], index >= 390 && index < 393 ? 128.0f : 1.0f);
  }
}

TEST_F(MusaReduceTest, RejectsInvalidAxesShapesModesAndEmptyInput) {
  auto input = tensor({3, 128});
  auto output = tensor({3});
  for (int64_t axis :
       {int64_t{-3}, int64_t{2}, std::numeric_limits<int64_t>::min()}) {
    EXPECT_EQ(run(output, input, axis, false, false), Error::InvalidArgument);
  }
  auto unsupported = tensor({3, 64});
  EXPECT_EQ(run(output, unsupported, 1, false, false), Error::NotSupported);
  auto empty = tensor({0, 128});
  EXPECT_EQ(run(output, empty, 1, false, false), Error::NotSupported);
  auto scalar = tensor({});
  EXPECT_EQ(run(output, scalar, 0, false, false), Error::NotSupported);
  auto bf16 = tensor({1, 1, 16, 128}, ScalarType::BFloat16);
  auto bf16_out = tensor({1, 1, 16}, ScalarType::BFloat16);
  EXPECT_EQ(run(bf16_out, bf16, 3, false, true), Error::NotSupported);
  EXPECT_EQ(
      aoti_torch_musa_reduce_out(&bf16_out, &bf16, 3, false, false),
      Error::InvalidArgument);
  auto wrong_shape = tensor({3, 1});
  EXPECT_EQ(run(wrong_shape, input, 1, false, false), Error::InvalidArgument);
  auto stepped = input.as_strided({3}, {2}, 390);
  EXPECT_EQ(run(stepped, input, 1, false, false), Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_musa_reduce_out(nullptr, &input, 1, false, false),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_musa_reduce_out(&output, nullptr, 1, false, false),
      Error::InvalidArgument);
}

TEST_F(MusaReduceTest, RejectsStorageBoundsNegativeStridesAndUnbackedData) {
  auto input = tensor({3, 128});
  auto output = tensor({3});
  auto beyond = input.as_strided({3, 128}, {128, 1}, 1);
  EXPECT_EQ(run(output, beyond, 1, false, false), Error::InvalidArgument);
  auto negative = input.as_strided({3, 128}, {-128, 1}, 256);
  EXPECT_EQ(run(output, negative, 1, false, false), Error::InvalidArgument);
  SlimTensor unbacked(
      slim::Storage(new slim::MaybeOwningStorage(
          input.device(), nullptr, input.nbytes())),
      input.sizes(),
      input.strides(),
      ScalarType::Float);
  EXPECT_EQ(run(output, unbacked, 1, false, false), Error::InvalidArgument);
  auto backing = tensor({400});
  SlimTensor misaligned(
      slim::Storage(new slim::MaybeOwningStorage(
          backing.device(),
          static_cast<char*>(backing.data_ptr()) + 1,
          backing.nbytes() - 1)),
      input.sizes(),
      input.strides(),
      ScalarType::Float);
  EXPECT_EQ(run(output, misaligned, 1, false, false), Error::InvalidArgument);
  auto overflow =
      input.as_strided({std::numeric_limits<int64_t>::max(), 128}, {0, 1}, 0);
  EXPECT_EQ(run(output, overflow, 1, false, false), Error::InvalidArgument);
}

TEST_F(MusaReduceTest, NonfiniteValuesPropagate) {
  auto input = tensor({3, 128});
  auto output = tensor({3});
  std::vector<float> values(input.numel(), 1.0f);
  values[0] = std::numeric_limits<float>::infinity();
  values[128] = -std::numeric_limits<float>::infinity();
  values[256] = std::numeric_limits<float>::quiet_NaN();
  ASSERT_EQ(
      cudaMemcpy(
          input.data_ptr(),
          values.data(),
          input.nbytes(),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  ASSERT_EQ(run(output, input, 1, false, false), Error::Ok);
  auto actual = read<float>(output);
  EXPECT_EQ(actual[0], std::numeric_limits<float>::infinity());
  EXPECT_EQ(actual[1], -std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isnan(actual[2]));
}

TEST_F(MusaReduceTest, UsesCurrentNonDefaultStream) {
  auto input = tensor({3, 128});
  auto output = tensor({3});
  std::vector<float> values(input.numel(), 1.0f);
  cudaStream_t stream = nullptr;
  ASSERT_EQ(
      musaStreamCreateWithFlags(&stream, musaStreamNonBlocking), musaSuccess);
  {
    auto guard = CUDAStreamGuard::create(stream, 0);
    ASSERT_TRUE(guard.ok());
    ASSERT_EQ(
        cudaMemcpyAsync(
            input.data_ptr(),
            values.data(),
            input.nbytes(),
            cudaMemcpyHostToDevice,
            stream),
        cudaSuccess);
    ASSERT_EQ(run(output, input, 1, false, false), Error::Ok);
    for (const auto value : read<float>(output)) {
      EXPECT_EQ(value, 128.0f);
    }
  }
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(MusaReduceTest, RestoresCallingDevice) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) {
    GTEST_SKIP() << "requires two MUSA devices";
  }
  auto input = tensor({3, 128});
  auto output = tensor({3});
  ASSERT_EQ(musaMemset(input.data_ptr(), 0, input.nbytes()), musaSuccess);
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  EXPECT_EQ(run(output, input, 1, false, false), Error::Ok);
  int current = -1;
  EXPECT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, 1);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
}

}
