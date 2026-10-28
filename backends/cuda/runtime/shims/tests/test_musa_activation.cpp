#include <gtest/gtest.h>

#include <cstring>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/musa_activation.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/core/portable_type/bfloat16.h>
#include <executorch/runtime/core/portable_type/half.h>
#include <executorch/runtime/platform/platform.h>

namespace {

namespace slim = executorch::backends::aoti::slim;
using executorch::backends::aoti::slim::SlimTensor;
using executorch::backends::aoti::slim::c10::DeviceType;
using executorch::backends::aoti::slim::c10::ScalarType;
using executorch::backends::cuda::aoti_torch__alloc_from_pool;
using executorch::backends::cuda::aoti_torch_empty_strided;
using executorch::backends::cuda::aoti_torch_musa_hardsigmoid;
using executorch::backends::cuda::aoti_torch_musa_hardsigmoid_out;
using executorch::backends::cuda::aoti_torch_musa_hardswish;
using executorch::backends::cuda::aoti_torch_musa_hardswish_out;
using executorch::backends::cuda::aoti_torch_musa_exp_out;
using executorch::backends::cuda::aoti_torch_musa_softplus_out;
using executorch::backends::cuda::CUDAStreamGuard;
using executorch::runtime::Error;
using executorch::runtime::etensor::BFloat16;
using executorch::runtime::etensor::Half;

Error default_softplus_out(SlimTensor* output, SlimTensor* input) {
  return aoti_torch_musa_softplus_out(output, input, 1.0, 20.0);
}

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
  cudaStream_t stream = nullptr;

  void SetUp() override {
    et_pal_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
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

TEST_F(MusaActivationTest, OutReusesDisjointPoolViewsAcrossShapesAndDtypes) {
  ASSERT_EQ(
      musaStreamCreateWithFlags(&stream, musaStreamNonBlocking), musaSuccess);
  auto guard = CUDAStreamGuard::create(stream, 0);
  ASSERT_TRUE(guard.ok());
  for (const bool hard_sigmoid : {false, true}) {
    const auto allocating =
        hard_sigmoid ? aoti_torch_musa_hardsigmoid : aoti_torch_musa_hardswish;
    const auto activation = hard_sigmoid ? aoti_torch_musa_hardsigmoid_out
                                         : aoti_torch_musa_hardswish_out;
    for (const auto dtype :
         {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
      for (const int64_t step : {1, 2}) {
        for (const std::vector<int64_t>& sizes :
             {std::vector<int64_t>{}, {0}, {2, 0, 3}, {9}, {1, 3, 3}}) {
          SCOPED_TRACE(
              ::testing::Message()
              << hard_sigmoid << " " << static_cast<int>(dtype) << " " << step
              << " rank=" << sizes.size());
          std::vector<int64_t> strides(sizes.size(), 1);
          for (int64_t dimension = static_cast<int64_t>(sizes.size()) - 2;
               dimension >= 0;
               --dimension) {
            strides[dimension] = strides[dimension + 1] * sizes[dimension + 1];
          }
          auto input_strides = strides;
          for (auto& stride : input_strides) {
            stride *= step;
          }
          auto pool = slim::empty(
              {256}, ScalarType::Byte, slim::c10::Device(slim::c10::kMUSA, 0));
          SlimTensor* raw_input = nullptr;
          SlimTensor* raw_output = nullptr;
          ASSERT_EQ(
              aoti_torch__alloc_from_pool(
                  &pool,
                  8,
                  static_cast<int32_t>(dtype),
                  sizes.size(),
                  sizes.data(),
                  input_strides.data(),
                  &raw_input),
              Error::Ok);
          ASSERT_EQ(
              aoti_torch__alloc_from_pool(
                  &pool,
                  128,
                  static_cast<int32_t>(dtype),
                  sizes.size(),
                  sizes.data(),
                  strides.data(),
                  &raw_output),
              Error::Ok);
          std::unique_ptr<SlimTensor> input(raw_input);
          std::unique_ptr<SlimTensor> output(raw_output);
          auto* original_pointer = output->data_ptr();
          for (int iteration = 0; iteration < 3; ++iteration) {
            std::vector<uint8_t> initial(256, 0xA5);
            for (size_t index = 0; index < input->numel(); ++index) {
              const float value = static_cast<float>(index) - 4 + iteration;
              const Half half(value);
              const BFloat16 bfloat(value);
              const void* encoded = dtype == ScalarType::Float
                  ? static_cast<const void*>(&value)
                  : dtype == ScalarType::Half
                  ? static_cast<const void*>(&half)
                  : static_cast<const void*>(&bfloat);
              std::memcpy(
                  initial.data() + 8 + index * step * input->itemsize(),
                  encoded,
                  input->itemsize());
            }
            ASSERT_EQ(
                cudaMemcpyAsync(
                    pool.data_ptr(),
                    initial.data(),
                    initial.size(),
                    cudaMemcpyHostToDevice,
                    stream),
                cudaSuccess);
            SlimTensor* raw_reference = nullptr;
            ASSERT_EQ(allocating(input.get(), &raw_reference), Error::Ok);
            std::unique_ptr<SlimTensor> reference(raw_reference);
            ASSERT_EQ(activation(output.get(), input.get()), Error::Ok);
            EXPECT_EQ(output->data_ptr(), original_pointer);
            std::vector<uint8_t> actual(256);
            std::vector<uint8_t> expected(output->nbytes());
            ASSERT_EQ(
                cudaMemcpyAsync(
                    actual.data(),
                    pool.data_ptr(),
                    actual.size(),
                    cudaMemcpyDeviceToHost,
                    stream),
                cudaSuccess);
            if (!expected.empty()) {
              ASSERT_EQ(
                  cudaMemcpyAsync(
                      expected.data(),
                      reference->data_ptr(),
                      expected.size(),
                      cudaMemcpyDeviceToHost,
                      stream),
                  cudaSuccess);
            }
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            for (size_t index = 0; index < actual.size(); ++index) {
              EXPECT_EQ(
                  actual[index],
                  index >= 128 && index < 128 + expected.size()
                      ? expected[index - 128]
                      : initial[index])
                  << index;
            }
          }
        }
      }
    }
  }
}

TEST_F(MusaActivationTest, OutRejectsInvalidViewsWithoutWriting) {
  auto pool = slim::empty(
      {256}, ScalarType::Byte, slim::c10::Device(slim::c10::kMUSA, 0));
  SlimTensor input(pool.storage(), {4}, {1}, ScalarType::Float);
  SlimTensor output(pool.storage(), {4}, {1}, ScalarType::Float, 32);
  const int64_t wrong_sizes[] = {2, 2};
  const int64_t wrong_strides[] = {2, 1};
  SlimTensor wrong_shape(
      pool.storage(),
      slim::IntArrayRef(wrong_sizes, 2),
      slim::IntArrayRef(wrong_strides, 2),
      ScalarType::Float,
      32);
  SlimTensor wrong_dtype(pool.storage(), {4}, {1}, ScalarType::Half, 64);
  SlimTensor noncontiguous(pool.storage(), {4}, {2}, ScalarType::Float, 32);
  SlimTensor insufficient(pool.storage(), {4}, {1}, ScalarType::Float, 63);
  SlimTensor negative_offset(pool.storage(), {4}, {1}, ScalarType::Float, -1);
  SlimTensor strided_overlap(pool.storage(), {4}, {16}, ScalarType::Float);
  SlimTensor partial_overlap(pool.storage(), {4}, {1}, ScalarType::Float, 33);
  SlimTensor invalid_source(pool.storage(), {4}, {64}, ScalarType::Float);
  SlimTensor undefined;
  auto host = slim::empty({4}, ScalarType::Float);
  std::vector<uint8_t> initial(256, 0xA5);
  ASSERT_EQ(
      cudaMemcpy(
          pool.data_ptr(),
          initial.data(),
          initial.size(),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  for (const auto activation :
       {aoti_torch_musa_hardswish_out, aoti_torch_musa_hardsigmoid_out,
        aoti_torch_musa_exp_out, default_softplus_out}) {
    for (auto* destination :
         {static_cast<SlimTensor*>(nullptr),
          &undefined,
          &input,
          &wrong_shape,
          &wrong_dtype,
          &noncontiguous,
          &insufficient,
          &negative_offset,
          &host}) {
      EXPECT_EQ(activation(destination, &input), Error::InvalidArgument);
    }
    for (auto* source :
         {static_cast<SlimTensor*>(nullptr),
          &undefined,
          &output,
          &strided_overlap,
          &partial_overlap,
          &invalid_source,
          &negative_offset,
          &host}) {
      EXPECT_EQ(activation(&output, source), Error::InvalidArgument);
    }
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint8_t> actual(256);
  ASSERT_EQ(
      cudaMemcpy(
          actual.data(),
          pool.data_ptr(),
          actual.size(),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(actual, initial);
}

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

TEST_F(MusaActivationTest, Fp32MathReusesStridedAndBroadcastPoolViews) {
  ASSERT_EQ(
      musaStreamCreateWithFlags(&stream, musaStreamNonBlocking), musaSuccess);
  auto guard = CUDAStreamGuard::create(stream, 0);
  ASSERT_TRUE(guard.ok());
  for (auto operation : {aoti_torch_musa_exp_out, default_softplus_out}) {
    for (int64_t stride : {0, 1, 2}) {
      auto pool = slim::empty(
          {128}, ScalarType::Float, slim::c10::Device(slim::c10::kMUSA, 0));
      SlimTensor input(pool.storage(), {9}, {stride}, ScalarType::Float, 3);
      SlimTensor output(pool.storage(), {9}, {1}, ScalarType::Float, 64);
      for (int iteration = 0; iteration < 3; ++iteration) {
        std::vector<float> initial(128, 17.0f);
        for (int64_t index = 0; index < 9; ++index) {
          initial[3 + index * stride] =
              static_cast<float>(index - 4 + iteration);
        }
        ASSERT_EQ(
            cudaMemcpyAsync(
                pool.data_ptr(),
                initial.data(),
                pool.nbytes(),
                cudaMemcpyHostToDevice,
                stream),
            cudaSuccess);
        ASSERT_EQ(operation(&output, &input), Error::Ok);
        std::vector<float> actual(128);
        ASSERT_EQ(
            cudaMemcpyAsync(
                actual.data(),
                pool.data_ptr(),
                pool.nbytes(),
                cudaMemcpyDeviceToHost,
                stream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        for (size_t index = 0; index < actual.size(); ++index) {
          if (index >= 64 && index < 73) {
            const float value = initial[3 + (index - 64) * stride];
            const float expected = operation == aoti_torch_musa_exp_out
                ? std::exp(value)
                : std::log1p(std::exp(value));
            EXPECT_NEAR(
                actual[index], expected, 1e-6f * std::max(1.0f, expected));
          } else {
            EXPECT_EQ(actual[index], initial[index]);
          }
        }
      }
    }
  }
}

TEST_F(MusaActivationTest, Fp32MathEmptyAndScalar) {
  for (auto operation : {aoti_torch_musa_exp_out, default_softplus_out}) {
    for (const std::vector<int64_t>& sizes :
         {std::vector<int64_t>{}, {0}, {2, 0, 3}}) {
      auto input = slim::empty(
          slim::makeArrayRef(sizes),
          ScalarType::Float,
          slim::c10::Device(slim::c10::kMUSA, 0));
      auto output = slim::empty(
          slim::makeArrayRef(sizes), ScalarType::Float, input.device());
      if (input.numel()) {
        const float value = 0.0f;
        ASSERT_EQ(
            cudaMemcpy(
                input.data_ptr(),
                &value,
                sizeof(value),
                cudaMemcpyHostToDevice),
            cudaSuccess);
      }
      ASSERT_EQ(operation(&output, &input), Error::Ok);
      if (input.numel()) {
        float actual;
        ASSERT_EQ(
            cudaMemcpy(
                &actual,
                output.data_ptr(),
                sizeof(actual),
                cudaMemcpyDeviceToHost),
            cudaSuccess);
        EXPECT_NEAR(
            actual,
            operation == aoti_torch_musa_exp_out ? 1.0f : std::log(2.0f),
            1e-7f);
      }
    }
  }
}

TEST_F(MusaActivationTest, Fp32MathSpecialValues) {
  const std::vector<float> values{
      -std::numeric_limits<float>::infinity(),
      0,
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN()};
  auto input = slim::empty(
      {4}, ScalarType::Float, slim::c10::Device(slim::c10::kMUSA, 0));
  auto output = slim::empty({4}, ScalarType::Float, input.device());
  ASSERT_EQ(
      cudaMemcpy(
          input.data_ptr(),
          values.data(),
          input.nbytes(),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  for (auto operation : {aoti_torch_musa_exp_out, default_softplus_out}) {
    ASSERT_EQ(operation(&output, &input), Error::Ok);
    std::vector<float> actual(4);
    ASSERT_EQ(
        cudaMemcpy(
            actual.data(),
            output.data_ptr(),
            output.nbytes(),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(actual[0], 0.0f);
    EXPECT_TRUE(std::isinf(actual[2]) && actual[2] > 0);
    EXPECT_TRUE(std::isnan(actual[3]));
  }
}

TEST_F(MusaActivationTest, Fp32MathRejectsOtherDtypesAndSoftplusParameters) {
  for (auto dtype : {ScalarType::Half, ScalarType::BFloat16, ScalarType::Int}) {
    auto input =
        slim::empty({4}, dtype, slim::c10::Device(slim::c10::kMUSA, 0));
    auto output = slim::empty({4}, dtype, input.device());
    ASSERT_EQ(
        musaMemset(output.data_ptr(), 0xA5, output.nbytes()), musaSuccess);
    for (auto operation : {aoti_torch_musa_exp_out, default_softplus_out}) {
      EXPECT_EQ(operation(&output, &input), Error::NotSupported);
    }
    std::vector<uint8_t> actual(output.nbytes());
    ASSERT_EQ(
        cudaMemcpy(
            actual.data(),
            output.data_ptr(),
            output.nbytes(),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(actual, std::vector<uint8_t>(output.nbytes(), 0xA5));
  }
  auto input = slim::empty(
      {4}, ScalarType::Float, slim::c10::Device(slim::c10::kMUSA, 0));
  auto output = slim::empty({4}, ScalarType::Float, input.device());
  ASSERT_EQ(musaMemset(output.data_ptr(), 0xA5, output.nbytes()), musaSuccess);
  for (double beta :
       {0.0, -1.0, 2.0, std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_EQ(
        aoti_torch_musa_softplus_out(&output, &input, beta, 20.0),
        Error::NotSupported);
  }
  EXPECT_EQ(
      aoti_torch_musa_softplus_out(&output, &input, 1.0, 10.0),
      Error::NotSupported);
  std::vector<uint8_t> actual(output.nbytes());
  ASSERT_EQ(
      cudaMemcpy(
          actual.data(),
          output.data_ptr(),
          output.nbytes(),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(actual, std::vector<uint8_t>(output.nbytes(), 0xA5));
}

TEST_F(MusaActivationTest, RejectsUnbackedAndIndependentlyWrappedOverlaps) {
  auto input = slim::empty(
      {8}, ScalarType::Float, slim::c10::Device(slim::c10::kMUSA, 0));
  auto output = slim::empty({8}, ScalarType::Float, input.device());
  ASSERT_EQ(musaMemset(output.data_ptr(), 0xA5, output.nbytes()), musaSuccess);
  for (void* address :
       {static_cast<void*>(nullptr),
        static_cast<void*>(static_cast<char*>(input.data_ptr()) + 1)}) {
    SlimTensor invalid(
        slim::Storage(
            new slim::MaybeOwningStorage(input.device(), address, 32)),
        {8},
        {1},
        ScalarType::Float);
    for (auto operation :
         {aoti_torch_musa_exp_out,
          default_softplus_out,
          aoti_torch_musa_hardswish_out,
          aoti_torch_musa_hardsigmoid_out}) {
      EXPECT_EQ(operation(&output, &invalid), Error::InvalidArgument);
      EXPECT_EQ(operation(&invalid, &input), Error::InvalidArgument);
    }
  }
  for (auto operation :
       {aoti_torch_musa_exp_out,
        default_softplus_out,
        aoti_torch_musa_hardswish_out,
        aoti_torch_musa_hardsigmoid_out}) {
    SlimTensor alias(
        slim::Storage(new slim::MaybeOwningStorage(
            output.device(), output.data_ptr(), output.nbytes())),
        {8},
        {1},
        ScalarType::Float);
    EXPECT_EQ(operation(&output, &alias), Error::InvalidArgument);
  }
  std::vector<uint8_t> actual(output.nbytes());
  ASSERT_EQ(
      cudaMemcpy(
          actual.data(),
          output.data_ptr(),
          output.nbytes(),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(actual, std::vector<uint8_t>(output.nbytes(), 0xA5));
}

TEST_F(MusaActivationTest, Fp32MathRestoresCallingDevice) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) {
    GTEST_SKIP() << "requires two MUSA devices";
  }
  auto input = slim::empty(
      {4}, ScalarType::Float, slim::c10::Device(slim::c10::kMUSA, 1));
  auto output = slim::empty({4}, ScalarType::Float, input.device());
  {
    auto guard = executorch::backends::cuda::CUDAGuard::create(1);
    ASSERT_TRUE(guard.ok());
    ASSERT_EQ(musaMemset(input.data_ptr(), 0, input.nbytes()), musaSuccess);
  }
  auto caller = executorch::backends::cuda::CUDAGuard::create(0);
  ASSERT_TRUE(caller.ok());
  for (auto operation : {aoti_torch_musa_exp_out, default_softplus_out}) {
    ASSERT_EQ(operation(&output, &input), Error::Ok);
    int current = -1;
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, 0);
  }
}

}
