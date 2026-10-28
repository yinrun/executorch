#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/cuda/runtime/shims/musa_causal_attention.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/core/portable_type/bfloat16.h>
#include <executorch/runtime/platform/platform.h>

namespace {

namespace slim = executorch::backends::aoti::slim;
using executorch::backends::cuda::aoti_torch_musa_causal_attention_out;
using executorch::backends::cuda::CUDAStreamGuard;
using executorch::runtime::Error;
using executorch::runtime::etensor::BFloat16;
using slim::SlimTensor;
using slim::c10::ScalarType;

SlimTensor tensor(
    std::initializer_list<int64_t> sizes,
    ScalarType dtype = ScalarType::BFloat16) {
  return slim::empty(
      slim::IntArrayRef(sizes.begin(), sizes.size()),
      dtype,
      slim::c10::Device(slim::c10::kMUSA, 0));
}

std::vector<uint8_t> bytes(const SlimTensor& input) {
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint8_t> result(input.nbytes());
  EXPECT_EQ(
      cudaMemcpy(
          result.data(),
          input.data_ptr(),
          result.size(),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  return result;
}

struct Inputs {
  SlimTensor query = tensor({1, 8, 1, 256});
  SlimTensor output = tensor({1, 8, 1, 256});
  SlimTensor key;
  SlimTensor value;
  SlimTensor position = tensor({1}, ScalarType::Long);
  SlimTensor mask;
  int64_t capacity;

  explicit Inputs(int64_t size = 64)
      : key(tensor({1, 8, size, 256})),
        value(tensor({1, 8, size, 256})),
        mask(tensor({1, size}, ScalarType::Bool)),
        capacity(size) {
    EXPECT_EQ(musaMemset(query.data_ptr(), 0, query.nbytes()), musaSuccess);
    set_length(size);
  }

  void set_length(int64_t length) {
    const int64_t index = length - 1;
    std::vector<uint8_t> mask_values(capacity);
    std::vector<BFloat16> key_values(8 * capacity * 256);
    std::vector<BFloat16> value_values(key_values.size());
    for (int64_t column = 0; column < capacity; ++column) {
      mask_values[column] = column < length;
      for (int64_t head = 0; head < 8; ++head) {
        for (int64_t channel = 0; channel < 256; ++channel) {
          const auto offset = (head * capacity + column) * 256 + channel;
          const float future = std::numeric_limits<float>::quiet_NaN();
          key_values[offset] = BFloat16(column < length ? 0.0f : future);
          value_values[offset] =
              BFloat16(column < length ? (column == 0 ? 1.0f : 0.0f) : future);
        }
      }
    }
    EXPECT_EQ(
        cudaMemcpy(
            position.data_ptr(), &index, sizeof(index), cudaMemcpyHostToDevice),
        cudaSuccess);
    EXPECT_EQ(
        cudaMemcpy(
            mask.data_ptr(),
            mask_values.data(),
            mask.nbytes(),
            cudaMemcpyHostToDevice),
        cudaSuccess);
    EXPECT_EQ(
        cudaMemcpy(
            key.data_ptr(),
            key_values.data(),
            key.nbytes(),
            cudaMemcpyHostToDevice),
        cudaSuccess);
    EXPECT_EQ(
        cudaMemcpy(
            value.data_ptr(),
            value_values.data(),
            value.nbytes(),
            cudaMemcpyHostToDevice),
        cudaSuccess);
  }

  Error run(SlimTensor* destination = nullptr) {
    return aoti_torch_musa_causal_attention_out(
        destination == nullptr ? &output : destination,
        &query,
        &key,
        &value,
        &position,
        &mask);
  }

  void expect_result(int64_t length, SlimTensor* destination = nullptr) {
    auto* result = destination == nullptr ? &output : destination;
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<uint8_t> storage(result->storage()->nbytes());
    ASSERT_EQ(
        cudaMemcpy(
            storage.data(),
            result->storage()->data(),
            storage.size(),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    std::vector<BFloat16> actual(2048);
    std::memcpy(
        actual.data(),
        storage.data() + result->storage_offset() * result->itemsize(),
        result->nbytes());
    const auto expected = BFloat16(1.0f / static_cast<float>(length));
    for (size_t index = 0; index < actual.size(); ++index) {
      SCOPED_TRACE(index);
      EXPECT_EQ(
          static_cast<float>(actual[index]), static_cast<float>(expected));
    }
  }
};

class MusaCausalAttentionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "MUSA device is not available";
    }
  }
};

TEST_F(
    MusaCausalAttentionTest,
    ValidLengthsAndPoisonedFutureDoNotMutateInputs) {
  for (int64_t capacity : {64, 128, 256}) {
    Inputs inputs(capacity);
    const auto address = inputs.output.data_ptr();
    for (int64_t length :
         {int64_t{1}, int64_t{2}, int64_t{3}, capacity - 1, capacity}) {
      inputs.set_length(length);
      std::vector<std::vector<uint8_t>> before;
      for (auto* input :
           {&inputs.query,
            &inputs.key,
            &inputs.value,
            &inputs.position,
            &inputs.mask}) {
        before.push_back(bytes(*input));
      }
      ASSERT_EQ(inputs.run(), Error::Ok);
      inputs.expect_result(length);
      EXPECT_EQ(inputs.output.data_ptr(), address);
      size_t index = 0;
      for (auto* input :
           {&inputs.query,
            &inputs.key,
            &inputs.value,
            &inputs.position,
            &inputs.mask}) {
        EXPECT_EQ(bytes(*input), before[index++]);
      }
    }
  }
}

TEST_F(MusaCausalAttentionTest, InvalidPositionsAndMasksLeaveOutputUntouched) {
  Inputs inputs;
  ASSERT_EQ(
      musaMemset(inputs.output.data_ptr(), 0x5A, inputs.output.nbytes()),
      musaSuccess);
  const auto original = bytes(inputs.output);
  for (int64_t index :
       {int64_t{-1}, int64_t{64}, std::numeric_limits<int64_t>::max()}) {
    ASSERT_EQ(
        cudaMemcpy(
            inputs.position.data_ptr(),
            &index,
            sizeof(index),
            cudaMemcpyHostToDevice),
        cudaSuccess);
    EXPECT_EQ(inputs.run(), Error::InvalidArgument);
    EXPECT_EQ(bytes(inputs.output), original);
  }
  inputs.set_length(32);
  for (int64_t column : {0, 31, 32, 63}) {
    inputs.set_length(32);
    uint8_t value = column < 32 ? 0 : 1;
    ASSERT_EQ(
        cudaMemcpy(
            static_cast<uint8_t*>(inputs.mask.data_ptr()) + column,
            &value,
            1,
            cudaMemcpyHostToDevice),
        cudaSuccess);
    EXPECT_EQ(inputs.run(), Error::InvalidArgument);
    EXPECT_EQ(bytes(inputs.output), original);
  }
  for (int64_t length : {32, 1, 64}) {
    inputs.set_length(length);
    ASSERT_EQ(inputs.run(), Error::Ok);
    inputs.expect_result(length);
  }
}

TEST_F(
    MusaCausalAttentionTest,
    OutputAliasesIncludingSeparateWrappersAreRejected) {
  Inputs inputs;
  EXPECT_EQ(inputs.run(&inputs.query), Error::InvalidArgument);
  auto same_storage =
      inputs.query.as_strided({1, 8, 1, 256}, {2048, 256, 256, 1}, 0);
  EXPECT_EQ(inputs.run(&same_storage), Error::InvalidArgument);
  SlimTensor separate(
      slim::Storage(new slim::MaybeOwningStorage(
          inputs.query.device(),
          inputs.query.data_ptr(),
          inputs.query.nbytes())),
      inputs.query.sizes(),
      inputs.query.strides(),
      ScalarType::BFloat16);
  EXPECT_EQ(inputs.run(&separate), Error::InvalidArgument);
  auto partial = inputs.key.as_strided({1, 8, 1, 256}, {2048, 256, 256, 1}, 1);
  EXPECT_EQ(inputs.run(&partial), Error::InvalidArgument);
}

TEST_F(MusaCausalAttentionTest, DisjointSharedStorageAndOffsetsAreSupported) {
  Inputs inputs;
  auto storage = tensor({4100});
  inputs.query = storage.as_strided({1, 8, 1, 256}, {2048, 256, 256, 1}, 2);
  auto output = storage.as_strided({1, 8, 1, 256}, {2048, 256, 256, 1}, 2052);
  ASSERT_EQ(
      musaMemset(inputs.query.data_ptr(), 0, inputs.query.nbytes()),
      musaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(inputs.run(&output), Error::Ok);
  inputs.expect_result(64, &output);
}

TEST_F(MusaCausalAttentionTest, MalformedMetadataAndStorageAreRejected) {
  Inputs inputs;
  auto wrong_dtype = tensor({1, 8, 1, 256}, ScalarType::Float);
  auto wrong_shape = tensor({1, 8, 2, 256});
  EXPECT_EQ(inputs.run(&wrong_dtype), Error::NotSupported);
  EXPECT_EQ(inputs.run(&wrong_shape), Error::NotSupported);
  auto out_of_bounds =
      inputs.output.as_strided({1, 8, 1, 256}, {2048, 256, 256, 1}, 1);
  EXPECT_EQ(inputs.run(&out_of_bounds), Error::InvalidArgument);
  auto wrong_stride_storage = tensor({4096});
  auto wrong_stride =
      wrong_stride_storage.as_strided({1, 8, 1, 256}, {4096, 512, 512, 2}, 0);
  EXPECT_EQ(inputs.run(&wrong_stride), Error::NotSupported);
  SlimTensor undefined;
  EXPECT_EQ(inputs.run(&undefined), Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_musa_causal_attention_out(
          nullptr,
          &inputs.query,
          &inputs.key,
          &inputs.value,
          &inputs.position,
          &inputs.mask),
      Error::InvalidArgument);
  SlimTensor null_data(
      slim::Storage(
          new slim::MaybeOwningStorage(inputs.query.device(), nullptr, 4096)),
      inputs.query.sizes(),
      inputs.query.strides(),
      ScalarType::BFloat16);
  EXPECT_EQ(inputs.run(&null_data), Error::InvalidArgument);
  SlimTensor misaligned(
      slim::Storage(new slim::MaybeOwningStorage(
          inputs.query.device(),
          static_cast<char*>(inputs.output.data_ptr()) + 1,
          4096)),
      inputs.query.sizes(),
      inputs.query.strides(),
      ScalarType::BFloat16);
  EXPECT_EQ(inputs.run(&misaligned), Error::InvalidArgument);
}

TEST_F(MusaCausalAttentionTest, UnalignedControlViewsPreservePositionAndMask) {
  Inputs inputs;
  auto position_storage = tensor({3}, ScalarType::Long);
  auto mask_storage = tensor({80}, ScalarType::Bool);
  std::array<int64_t, 3> positions = {-7, 31, -9};
  std::array<uint8_t, 80> masks{};
  for (size_t column = 3; column < 35; ++column) {
    masks[column] = 1;
  }
  ASSERT_EQ(
      cudaMemcpy(
          position_storage.data_ptr(),
          positions.data(),
          sizeof(positions),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(
          mask_storage.data_ptr(),
          masks.data(),
          masks.size(),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  inputs.position = position_storage.as_strided({1}, {1}, 1);
  inputs.mask = mask_storage.as_strided({1, 64}, {64, 1}, 3);
  ASSERT_EQ(inputs.run(), Error::Ok);
  inputs.expect_result(32);
  positions[1] = -1;
  ASSERT_EQ(
      cudaMemcpy(
          position_storage.data_ptr(),
          positions.data(),
          sizeof(positions),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  EXPECT_EQ(inputs.run(), Error::InvalidArgument);
}

TEST_F(MusaCausalAttentionTest, InvalidControlDtypesAndGeometryAreRejected) {
  Inputs inputs;
  auto wrong_position = tensor({1}, ScalarType::Int);
  auto wrong_mask = tensor({1, 64}, ScalarType::Byte);
  EXPECT_EQ(
      aoti_torch_musa_causal_attention_out(
          &inputs.output,
          &inputs.query,
          &inputs.key,
          &inputs.value,
          &wrong_position,
          &inputs.mask),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_musa_causal_attention_out(
          &inputs.output,
          &inputs.query,
          &inputs.key,
          &inputs.value,
          &inputs.position,
          &wrong_mask),
      Error::InvalidArgument);
  auto unsupported_key = tensor({1, 8, 32, 256});
  EXPECT_EQ(
      aoti_torch_musa_causal_attention_out(
          &inputs.output,
          &inputs.query,
          &unsupported_key,
          &unsupported_key,
          &inputs.position,
          &inputs.mask),
      Error::NotSupported);
}

TEST_F(MusaCausalAttentionTest, ReadsControlUpdatesOnNonDefaultStream) {
  Inputs inputs;
  cudaStream_t stream = nullptr;
  ASSERT_EQ(
      musaStreamCreateWithFlags(&stream, musaStreamNonBlocking), musaSuccess);
  {
    auto guard = CUDAStreamGuard::create(stream, 0);
    ASSERT_TRUE(guard.ok());
    const int64_t index = 0;
    std::array<uint8_t, 64> mask{};
    mask[0] = 1;
    ASSERT_EQ(
        cudaMemcpyAsync(
            inputs.position.data_ptr(),
            &index,
            sizeof(index),
            cudaMemcpyHostToDevice,
            stream),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            inputs.mask.data_ptr(),
            mask.data(),
            mask.size(),
            cudaMemcpyHostToDevice,
            stream),
        cudaSuccess);
    ASSERT_EQ(inputs.run(), Error::Ok);
    inputs.expect_result(1);
  }
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(MusaCausalAttentionTest, RestoresCallingDevice) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) {
    GTEST_SKIP() << "requires two MUSA devices";
  }
  Inputs inputs;
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  EXPECT_EQ(inputs.run(), Error::Ok);
  int current = -1;
  EXPECT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, 1);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  inputs.expect_result(64);
}

}
