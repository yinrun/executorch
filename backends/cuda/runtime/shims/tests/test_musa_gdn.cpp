#include <gtest/gtest.h>

#include <vector>

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/cuda/runtime/shims/musa_gdn.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/platform/platform.h>

namespace {
namespace slim = executorch::backends::aoti::slim;
using executorch::backends::cuda::aoti_torch_musa_gdn_recurrence_out;
using executorch::runtime::Error;
using slim::SlimTensor;
using slim::c10::ScalarType;

SlimTensor tensor(std::initializer_list<int64_t> shape) {
  return slim::empty(
      shape, ScalarType::Float, slim::c10::Device(slim::c10::kMUSA, 0));
}

void fill(SlimTensor& target, float value) {
  std::vector<float> values(target.numel(), value);
  ASSERT_EQ(
      cudaMemcpy(
          target.data_ptr(),
          values.data(),
          target.nbytes(),
          cudaMemcpyHostToDevice),
      cudaSuccess);
}

std::vector<float> read(SlimTensor& target) {
  std::vector<float> values(target.numel());
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(
      cudaMemcpy(
          values.data(),
          target.data_ptr(),
          target.nbytes(),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  return values;
}

class MusaGdnTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "MUSA device is not available";
    }
  }
};

TEST_F(MusaGdnTest, ExactRepresentableUpdatesOffsetsAndCallingStream) {
  for (int heads : {16, 32}) {
    auto query = tensor({1, heads, 1, 128});
    auto key = tensor({1, heads, 1, 128});
    auto value = tensor({1, heads, 1, 128});
    auto decay = tensor({1, heads, 1});
    auto beta = tensor({1, heads, 1});
    auto state = tensor({1, heads, 128, 128});
    const int state_size = heads * 128 * 128;
    const int output_size = state_size + heads * 128;
    auto owner = tensor({output_size + 2});
    auto output = owner.as_strided({output_size}, {1}, 1);
    fill(query, 1.0f / 32);
    fill(key, 1.0f / 16);
    fill(value, 0.5f);
    fill(decay, 0.5f);
    fill(beta, 0.75f);
    fill(state, 1.0f / 64);
    fill(owner, 12345);
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    {
      auto guard =
          executorch::backends::cuda::CUDAStreamGuard::create(stream, 0);
      ASSERT_TRUE(guard.ok());
      ASSERT_EQ(
          aoti_torch_musa_gdn_recurrence_out(
              &output, &query, &key, &value, &decay, &beta, &state),
          Error::Ok);
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      const auto actual = read(owner);
      ASSERT_EQ(actual.front(), 12345);
      ASSERT_EQ(actual.back(), 12345);
      const float updated = 1.0f / 128 + (0.5f - 1.0f / 16) * 0.75f / 16;
      for (int index = 0; index < output_size; ++index) {
        ASSERT_EQ(
            actual[index + 1], index < state_size ? updated : updated * 4);
      }
      for (float element : read(state)) {
        ASSERT_EQ(element, 1.0f / 64);
      }
      ASSERT_EQ(
          cudaMemsetAsync(state.data_ptr(), 0, state.nbytes(), stream),
          cudaSuccess);
      ASSERT_EQ(
          cudaMemsetAsync(value.data_ptr(), 0, value.nbytes(), stream),
          cudaSuccess);
      ASSERT_EQ(
          aoti_torch_musa_gdn_recurrence_out(
              &output, &query, &key, &value, &decay, &beta, &state),
          Error::Ok);
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      for (float element : read(output)) {
        ASSERT_EQ(element, 0);
      }
    }
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  }
}

TEST_F(MusaGdnTest, PreservesDistinctExportedFp32ReductionOrders) {
  for (int heads : {16, 32}) {
    auto query = tensor({1, heads, 1, 128});
    auto key = tensor({1, heads, 1, 128});
    auto value = tensor({1, heads, 1, 128});
    auto decay = tensor({1, heads, 1});
    auto beta = tensor({1, heads, 1});
    auto state = tensor({1, heads, 128, 128});
    const int state_size = heads * 128 * 128;
    auto output = tensor({state_size + heads * 128});
    std::vector<float> initial(state_size, 0);
    for (int head = 0; head < heads; ++head) {
      for (int column = 0; column < 128; ++column) {
        const int base = head * 128 * 128 + column;
        initial[base] = 100000000.0f;
        if (column % 3 == 0) {
          initial[base + 32 * 128] = 1;
          initial[base + 64 * 128] = -100000000.0f;
          initial[base + 96 * 128] = 1;
        } else if (column % 3 == 1) {
          initial[base + 128] = -100000000.0f;
          initial[base + 2 * 128] = 1;
          initial[base + 3 * 128] = 1;
        } else {
          initial[base + 4 * 128] = -100000000.0f;
          initial[base + 2 * 128] = 1;
          initial[base + 6 * 128] = 1;
        }
      }
    }
    ASSERT_EQ(
        cudaMemcpy(
            state.data_ptr(),
            initial.data(),
            state.nbytes(),
            cudaMemcpyHostToDevice),
        cudaSuccess);
    fill(query, 0);
    fill(key, 1);
    fill(value, 0);
    fill(decay, 1);
    fill(beta, 1);
    ASSERT_EQ(
        aoti_torch_musa_gdn_recurrence_out(
            &output, &query, &key, &value, &decay, &beta, &state),
        Error::Ok);
    const auto updated = read(output);
    for (int index = 0; index < state_size; ++index) {
      const float memory = (index % 128) % 3 == 0 ? 1.0f : 0.0f;
      ASSERT_EQ(updated[index], initial[index] - memory);
    }
    for (int index = state_size; index < output.numel(); ++index) {
      ASSERT_EQ(updated[index], 0);
    }
    fill(query, 1);
    fill(beta, 0);
    ASSERT_EQ(
        aoti_torch_musa_gdn_recurrence_out(
            &output, &query, &key, &value, &decay, &beta, &state),
        Error::Ok);
    const auto projected = read(output);
    for (int index = 0; index < state_size; ++index) {
      ASSERT_EQ(projected[index], initial[index]);
    }
    for (int index = 0; index < heads * 128; ++index) {
      const int pattern = (index % 128) % 3;
      ASSERT_EQ(
          projected[state_size + index],
          pattern == 0       ? 1
              : pattern == 1 ? 0
                             : 2);
    }
    ASSERT_EQ(read(state), initial);
  }
}

TEST_F(MusaGdnTest, RejectsBadShapesStorageAndAliasing) {
  auto query = tensor({1, 32, 1, 128});
  auto decay = tensor({1, 32, 1});
  auto owner = tensor({32 * (128 * 128 + 128)});
  auto state = owner.as_strided(
      {1, 32, 128, 128}, {32 * 128 * 128, 128 * 128, 128, 1}, 0);
  auto output = tensor({32 * (128 * 128 + 128)});
  EXPECT_EQ(
      aoti_torch_musa_gdn_recurrence_out(
          nullptr, &query, &query, &query, &decay, &decay, &state),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_musa_gdn_recurrence_out(
          &owner, &query, &query, &query, &decay, &decay, &state),
      Error::InvalidArgument);
  auto short_query =
      query.as_strided({1, 32, 1, 127}, {32 * 127, 127, 127, 1}, 0);
  EXPECT_EQ(
      aoti_torch_musa_gdn_recurrence_out(
          &output, &short_query, &query, &query, &decay, &decay, &state),
      Error::InvalidArgument);
  auto invalid_storage =
      query.as_strided({1, 32, 1, 128}, {4096, 128, 128, 1}, 1);
  EXPECT_EQ(
      aoti_torch_musa_gdn_recurrence_out(
          &output, &invalid_storage, &query, &query, &decay, &decay, &state),
      Error::InvalidArgument);
  auto bad_decay = tensor({32});
  EXPECT_EQ(
      aoti_torch_musa_gdn_recurrence_out(
          &output, &query, &query, &query, &bad_decay, &decay, &state),
      Error::InvalidArgument);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
}
