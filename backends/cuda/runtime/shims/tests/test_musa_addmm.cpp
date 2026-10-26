#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/platform/platform.h>

namespace executorch::backends::cuda {
extern "C" {
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_musa_addmm_out(
    SlimTensor*,
    SlimTensor*,
    SlimTensor*,
    SlimTensor*,
    double,
    double);
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_musa_mm_out(
    SlimTensor*,
    SlimTensor*,
    SlimTensor*);
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_musa_bmm_out(
    SlimTensor*,
    SlimTensor*,
    SlimTensor*);
}
}

namespace {

using executorch::backends::aoti::slim::c10::DeviceType;
using executorch::backends::aoti::slim::c10::ScalarType;
using executorch::backends::cuda::aoti_torch_empty_strided;
using executorch::backends::cuda::aoti_torch_musa_addmm_out;
using executorch::backends::cuda::aoti_torch_musa_bmm_out;
using executorch::backends::cuda::aoti_torch_musa_mm_out;
using executorch::backends::cuda::clearCurrentCUDAStream;
using executorch::backends::cuda::getCurrentCUDAStream;
using executorch::backends::cuda::setCurrentCUDAStream;
using executorch::backends::cuda::SlimTensor;
using executorch::runtime::Error;

uint16_t float_to_half_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fffffff) == 0) {
    return static_cast<uint16_t>(bits >> 16);
  }
  const uint32_t exponent = (bits >> 23) & 0xff;
  const uint32_t mantissa = bits & 0x7fffff;
  return static_cast<uint16_t>(((exponent - 112) << 10) | (mantissa >> 13));
}

float half_bits_to_float(uint16_t value) {
  if ((value & 0x7fff) == 0) {
    uint32_t bits = static_cast<uint32_t>(value & 0x8000) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
  }
  const uint32_t exponent = (value >> 10) & 0x1f;
  const uint32_t mantissa = value & 0x3ff;
  uint32_t bits = ((exponent + 112) << 23) | (mantissa << 13);
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t float_to_bfloat16_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

float bfloat16_bits_to_float(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

size_t itemsize(ScalarType dtype) {
  return dtype == ScalarType::Float ? sizeof(float) : sizeof(uint16_t);
}

std::vector<uint8_t> encode_values(
    const std::vector<float>& values,
    ScalarType dtype) {
  std::vector<uint8_t> encoded(values.size() * itemsize(dtype));
  for (size_t index = 0; index < values.size(); ++index) {
    if (dtype == ScalarType::Float) {
      std::memcpy(encoded.data() + index * sizeof(float), &values[index], sizeof(float));
      continue;
    }
    const uint16_t bits = dtype == ScalarType::Half
        ? float_to_half_bits(values[index])
        : float_to_bfloat16_bits(values[index]);
    std::memcpy(encoded.data() + index * sizeof(uint16_t), &bits, sizeof(bits));
  }
  return encoded;
}

std::vector<float> decode_values(
    const std::vector<uint8_t>& encoded,
    ScalarType dtype) {
  const size_t element_count = encoded.size() / itemsize(dtype);
  std::vector<float> values(element_count);
  for (size_t index = 0; index < element_count; ++index) {
    if (dtype == ScalarType::Float) {
      std::memcpy(&values[index], encoded.data() + index * sizeof(float), sizeof(float));
      continue;
    }
    uint16_t bits;
    std::memcpy(&bits, encoded.data() + index * sizeof(uint16_t), sizeof(bits));
    values[index] = dtype == ScalarType::Half
        ? half_bits_to_float(bits)
        : bfloat16_bits_to_float(bits);
  }
  return values;
}

SlimTensor* empty_tensor(
    ScalarType dtype,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides) {
  SlimTensor* tensor = nullptr;
  const Error error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      strides.data(),
      static_cast<int32_t>(dtype),
      static_cast<int32_t>(DeviceType::CUDA),
      0,
      &tensor);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor, nullptr);
  return tensor;
}

void copy_to_device(SlimTensor* tensor, const std::vector<uint8_t>& values) {
  auto stream = getCurrentCUDAStream(tensor->device_index());
  ASSERT_TRUE(stream.ok());
  EXPECT_EQ(
      cudaMemcpyAsync(
          tensor->data_ptr(),
          values.data(),
          values.size(),
          cudaMemcpyHostToDevice,
          stream.get()),
      cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
}

std::vector<float> copy_from_device(
    SlimTensor* tensor,
    ScalarType dtype,
    size_t physical_elements = 0) {
  std::vector<uint8_t> encoded(
      physical_elements ? physical_elements * itemsize(dtype) : tensor->nbytes());
  auto stream = getCurrentCUDAStream(tensor->device_index());
  EXPECT_TRUE(stream.ok());
  if (!stream.ok()) {
    return {};
  }
  EXPECT_EQ(
      cudaMemcpyAsync(
          encoded.data(),
          tensor->data_ptr(),
          encoded.size(),
          cudaMemcpyDeviceToHost,
          stream.get()),
      cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  return decode_values(encoded, dtype);
}

void expect_values(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], 0.01f) << "index " << index;
  }
}

class MusaAddmmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "MUSA device is not available";
    }
  }
};

void run_addmm(
    ScalarType dtype,
    bool transposed_mat2,
    bool transposed_mat1 = false,
    bool padded_mat1 = false,
    double beta = 1.0) {
  const std::vector<int64_t> output_sizes = {2, 2};
  const std::vector<int64_t> output_strides = {2, 1};
  const std::vector<int64_t> bias_sizes = {2};
  const std::vector<int64_t> bias_strides = {1};
  const std::vector<int64_t> mat1_sizes = {2, 3};
  const std::vector<int64_t> mat1_strides =
      transposed_mat1 ? std::vector<int64_t>{1, 2}
      : std::vector<int64_t>{padded_mat1 ? 4 : 3, 1};
  const std::vector<int64_t> mat2_sizes = {3, 2};
  const std::vector<int64_t> mat2_strides = transposed_mat2
      ? std::vector<int64_t>{1, 3}
      : std::vector<int64_t>{2, 1};
  SlimTensor* output = empty_tensor(dtype, output_sizes, output_strides);
  SlimTensor* bias = empty_tensor(dtype, bias_sizes, bias_strides);
  SlimTensor* mat1 = empty_tensor(dtype, mat1_sizes, mat1_strides);
  SlimTensor* mat2 = empty_tensor(dtype, mat2_sizes, mat2_strides);
  const std::vector<float> mat1_values =
      transposed_mat1 ? std::vector<float>{1, 4, 2, 5, 3, 6}
      : (padded_mat1 ? std::vector<float>{1, 2, 3, 0, 4, 5, 6}
                     : std::vector<float>{1, 2, 3, 4, 5, 6});
  const std::vector<float> mat2_values = transposed_mat2
      ? std::vector<float>{7, 9, 11, 8, 10, 12}
      : std::vector<float>{7, 8, 9, 10, 11, 12};
  copy_to_device(mat1, encode_values(mat1_values, dtype));
  copy_to_device(mat2, encode_values(mat2_values, dtype));
  copy_to_device(bias, encode_values({1, 2}, dtype));
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(
      aoti_torch_musa_addmm_out(output, bias, mat1, mat2, beta, 1.0),
      Error::Ok);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SCOPED_TRACE(::testing::Message()
               << "dtype=" << static_cast<int>(dtype)
               << " transposed_mat2=" << transposed_mat2
               << " transposed_mat1=" << transposed_mat1
               << " padded_mat1=" << padded_mat1 << " beta=" << beta);
  const auto actual = copy_from_device(output, dtype);
  const float last_expected = beta == 0.0 ? 154.0f : 156.0f;
  if (padded_mat1 && actual.size() == 4 &&
      (!std::isfinite(actual[3]) ||
       std::abs(actual[3] - last_expected) > 0.01f)) {
    const auto repeated = copy_from_device(output, dtype);
    const auto input = copy_from_device(mat1, dtype, 7);
    const auto weights = copy_from_device(mat2, dtype);
    const auto offsets = copy_from_device(bias, dtype);
    ADD_FAILURE() << "output=" << actual[0] << "," << actual[3]
                  << " repeated=" << repeated[0] << "," << repeated[3]
                  << " input=" << input[0] << "," << input[4] << ","
                  << input[5] << "," << input[6]
                  << " weights=" << weights[0]
                  << " bias=" << offsets[0]
                  << " pointers=" << output->data_ptr() << ","
                  << mat1->data_ptr() << "," << mat2->data_ptr() << ","
                  << bias->data_ptr();
  }
  expect_values(
      actual, beta == 0.0 ? std::vector<float>{58, 64, 139, 154}
                          : std::vector<float>{59, 66, 140, 156});

  delete output;
  delete bias;
  delete mat1;
  delete mat2;
}

void run_mm(ScalarType dtype, bool transposed_mat1 = false) {
  const std::vector<int64_t> output_sizes = {2, 2};
  const std::vector<int64_t> matrix_sizes = {2, 3};
  const std::vector<int64_t> mat2_sizes = {3, 2};
  SlimTensor* output = empty_tensor(dtype, output_sizes, {2, 1});
  SlimTensor* mat1 = empty_tensor(
      dtype,
      matrix_sizes,
      transposed_mat1 ? std::vector<int64_t>{1, 2}
                      : std::vector<int64_t>{3, 1});
  SlimTensor* mat2 = empty_tensor(dtype, mat2_sizes, {2, 1});
  copy_to_device(
      mat1,
      encode_values(
          transposed_mat1 ? std::vector<float>{1, 4, 2, 5, 3, 6}
                          : std::vector<float>{1, 2, 3, 4, 5, 6},
          dtype));
  copy_to_device(mat2, encode_values({7, 8, 9, 10, 11, 12}, dtype));

  EXPECT_EQ(aoti_torch_musa_mm_out(output, mat1, mat2), Error::Ok);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_values(copy_from_device(output, dtype), {58, 64, 139, 154});

  delete output;
  delete mat1;
  delete mat2;
}

void run_bmm(
    ScalarType dtype,
    bool transposed_mat2,
    bool padded_mat1 = false,
    bool padded_mat2 = false,
    bool transposed_mat1 = false) {
  const std::vector<int64_t> output_sizes = {2, 2, 2};
  const std::vector<int64_t> mat1_sizes = {2, 2, 3};
  const std::vector<int64_t> mat2_sizes = {2, 3, 2};
  SlimTensor* output = empty_tensor(dtype, output_sizes, {4, 2, 1});
  SlimTensor* mat1 = empty_tensor(
      dtype,
      mat1_sizes,
      transposed_mat1
          ? std::vector<int64_t>{padded_mat1 ? 7 : 6, 1, 2}
          : std::vector<int64_t>{padded_mat1 ? 7 : 6, 3, 1});
  SlimTensor* mat2 = empty_tensor(
      dtype,
      mat2_sizes,
      transposed_mat2
          ? std::vector<int64_t>{padded_mat2 ? 7 : 6, 1, 3}
          : std::vector<int64_t>{padded_mat2 ? 7 : 6, 2, 1});
  std::vector<float> mat1_values = transposed_mat1
      ? std::vector<float>{1, 4, 2, 5, 3, 6, 2, 5, 3, 6, 4, 7}
      : std::vector<float>{1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7};
  if (padded_mat1) {
    mat1_values.insert(mat1_values.begin() + 6, 0);
  }
  copy_to_device(mat1, encode_values(mat1_values, dtype));
  std::vector<float> mat2_values = transposed_mat2
      ? std::vector<float>{7, 9, 11, 8, 10, 12, 1, 3, 5, 2, 4, 6}
      : std::vector<float>{7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6};
  if (padded_mat2) {
    mat2_values.insert(mat2_values.begin() + 6, 0);
  }
  copy_to_device(
      mat2,
      encode_values(mat2_values, dtype));

  EXPECT_EQ(aoti_torch_musa_bmm_out(output, mat1, mat2), Error::Ok);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_values(
      copy_from_device(output, dtype),
      {58, 64, 139, 154, 31, 40, 58, 76});

  delete output;
  delete mat1;
  delete mat2;
}

TEST_F(MusaAddmmTest, ImmutableTensorReadbackTail) {
  for (int64_t elements : {1000, 1001, 1999}) {
    auto tensor = std::unique_ptr<SlimTensor>(
        empty_tensor(ScalarType::Float, {elements}, {1}));
    copy_to_device(
        tensor.get(),
        encode_values(std::vector<float>(elements, 2.5f), ScalarType::Float));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int repeat = 0; repeat < 5000; ++repeat) {
      const auto actual = copy_from_device(tensor.get(), ScalarType::Float);
      for (size_t index = 0; index < actual.size(); ++index) {
        ASSERT_EQ(actual[index], 2.5f)
            << "elements=" << elements << " repeat=" << repeat
            << " index=" << index;
      }
    }
  }
}

TEST_F(MusaAddmmTest, SynchronousPageableAndPinnedReadbackTail) {
  auto stream = getCurrentCUDAStream(0);
  ASSERT_TRUE(stream.ok());
  for (int64_t elements : {1000, 1001, 1999}) {
    auto tensor = std::unique_ptr<SlimTensor>(
        empty_tensor(ScalarType::Float, {elements}, {1}));
    const std::vector<float> expected(elements, 2.5f);
    copy_to_device(tensor.get(), encode_values(expected, ScalarType::Float));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    void* pinned = nullptr;
    ASSERT_EQ(cudaMallocHost(&pinned, tensor->nbytes()), cudaSuccess);
    musaPointerAttributes attributes{};
    ASSERT_EQ(musaPointerGetAttributes(&attributes, pinned), musaSuccess);
    EXPECT_EQ(attributes.type, musaMemoryTypeHost);
    for (int repeat = 0; repeat < 1000; ++repeat) {
      std::vector<float> pageable(elements, 0.0f);
      EXPECT_EQ(
          cudaMemcpy(
              pageable.data(),
              tensor->data_ptr(),
              tensor->nbytes(),
              cudaMemcpyDeviceToHost),
          cudaSuccess);
      EXPECT_EQ(pageable, expected)
          << "repeat=" << repeat << " elements=" << elements;
      std::memset(pinned, 0, tensor->nbytes());
      EXPECT_EQ(
          cudaMemcpyAsync(
              pinned,
              tensor->data_ptr(),
              tensor->nbytes(),
              cudaMemcpyDeviceToHost,
              *stream),
          cudaSuccess);
      EXPECT_EQ(cudaStreamSynchronize(*stream), cudaSuccess);
      EXPECT_EQ(std::memcmp(pinned, expected.data(), tensor->nbytes()), 0)
          << "repeat=" << repeat << " elements=" << elements;
    }
    EXPECT_EQ(cudaFreeHost(pinned), cudaSuccess);
  }
}

TEST_F(MusaAddmmTest, BlasQueuedCallsAcrossStreams) {
  std::thread worker([] {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(setCurrentCUDAStream(nullptr, 0), Error::Ok);
    std::array<cudaStream_t, 2> streams{};
    for (auto& stream : streams) {
      ASSERT_EQ(
          musaStreamCreateWithFlags(&stream, musaStreamNonBlocking),
          musaSuccess);
    }
    for (auto dtype :
         {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
      auto mat1 =
          std::unique_ptr<SlimTensor>(empty_tensor(dtype, {2, 3}, {3, 1}));
      auto mat2 =
          std::unique_ptr<SlimTensor>(empty_tensor(dtype, {3, 2}, {2, 1}));
      auto bias = std::unique_ptr<SlimTensor>(empty_tensor(dtype, {2}, {1}));
      copy_to_device(mat1.get(), encode_values({1, 2, 3, 4, 5, 6}, dtype));
      copy_to_device(mat2.get(), encode_values({7, 8, 9, 10, 11, 12}, dtype));
      copy_to_device(bias.get(), encode_values({1, 2}, dtype));
      std::vector<std::unique_ptr<SlimTensor>> outputs;
      for (size_t index = 0; index < 16; ++index) {
        outputs.emplace_back(empty_tensor(dtype, {2, 2}, {2, 1}));
        copy_to_device(
            outputs.back().get(), encode_values({0, 0, 0, 0}, dtype));
      }
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      for (size_t index = 0; index < outputs.size(); ++index) {
        ASSERT_EQ(
            setCurrentCUDAStream(streams[index % streams.size()], 0),
            Error::Ok);
        ASSERT_EQ(
            aoti_torch_musa_addmm_out(
                outputs[index].get(),
                bias.get(),
                mat1.get(),
                mat2.get(),
                1.0,
                index % 2 == 0 ? 1.0 : 0.5),
            Error::Ok);
      }
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      for (size_t index = 0; index < outputs.size(); ++index) {
        expect_values(
            copy_from_device(outputs[index].get(), dtype),
            index % 2 == 0 ? std::vector<float>{59, 66, 140, 156}
                           : std::vector<float>{30, 34, 70.5, 79});
      }
    }
    clearCurrentCUDAStream(0);
    for (auto stream : streams) {
      EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
  });
  worker.join();
}

TEST_F(MusaAddmmTest, BlasHandleStreamSwitchAndThreadExit) {
  std::array<std::thread, 3> workers;
  for (auto& worker : workers) {
    worker = std::thread([] {
      ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
      std::array<cudaStream_t, 2> streams{};
      for (auto& stream : streams) {
        ASSERT_EQ(
            musaStreamCreateWithFlags(&stream, musaStreamNonBlocking),
            musaSuccess);
      }
      for (int repeat = 0; repeat < 3; ++repeat) {
        for (auto stream : streams) {
          ASSERT_EQ(setCurrentCUDAStream(stream, 0), Error::Ok);
          for (auto dtype :
               {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
            run_addmm(dtype, true, false, false);
            run_addmm(dtype, false, false, true, 0.0);
            run_bmm(dtype, true, true, false);
          }
        }
      }
      EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      clearCurrentCUDAStream(0);
      for (auto stream : streams) {
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
}

TEST_F(MusaAddmmTest, AddmmDtypesAndTransposedMat2) {
  for (ScalarType dtype : {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    run_addmm(dtype, false);
    run_addmm(dtype, true);
    run_addmm(dtype, false, true);
    run_addmm(dtype, true, true);
    run_addmm(dtype, false, false, true);
    run_addmm(dtype, true, false, true);
  }
}

TEST_F(MusaAddmmTest, PaddedMat1WithoutBias) {
  for (ScalarType dtype :
       {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    run_addmm(dtype, false, false, true, 0.0);
    run_addmm(dtype, true, false, true, 0.0);
  }
}

TEST_F(MusaAddmmTest, MmDtypes) {
  for (ScalarType dtype : {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    run_mm(dtype);
    run_mm(dtype, true);
  }
}

TEST_F(MusaAddmmTest, BmmDtypes) {
  for (ScalarType dtype : {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    run_bmm(dtype, false);
    run_bmm(dtype, true);
    run_bmm(dtype, false, true);
    run_bmm(dtype, false, false, true);
    run_bmm(dtype, true, true, true);
    run_bmm(dtype, false, false, false, true);
    run_bmm(dtype, true, false, false, true);
    run_bmm(dtype, true, true, true, true);
  }
}

TEST_F(MusaAddmmTest, BmmRejectsOverlappingBatches) {
  SlimTensor* output = empty_tensor(ScalarType::Float, {2, 2, 2}, {4, 2, 1});
  SlimTensor* mat1 = empty_tensor(ScalarType::Float, {2, 2, 3}, {5, 3, 1});
  SlimTensor* mat2 = empty_tensor(ScalarType::Float, {2, 3, 2}, {6, 2, 1});

  EXPECT_EQ(aoti_torch_musa_bmm_out(output, mat1, mat2), Error::NotSupported);

  delete output;
  delete mat1;
  delete mat2;

  output = empty_tensor(ScalarType::Float, {2, 2, 2}, {4, 2, 1});
  mat1 = empty_tensor(ScalarType::Float, {2, 2, 3}, {5, 1, 2});
  mat2 = empty_tensor(ScalarType::Float, {2, 3, 2}, {6, 2, 1});

  EXPECT_EQ(aoti_torch_musa_bmm_out(output, mat1, mat2), Error::NotSupported);

  delete output;
  delete mat1;
  delete mat2;
}

TEST_F(MusaAddmmTest, BmmSingleRowWithZeroRowStride) {
  for (ScalarType dtype : {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    SlimTensor* output = empty_tensor(dtype, {2, 1, 2}, {2, 2, 1});
    SlimTensor* mat1 = empty_tensor(dtype, {2, 1, 3}, {3, 0, 1});
    SlimTensor* mat2 = empty_tensor(dtype, {2, 3, 2}, {6, 2, 1});
    copy_to_device(mat1, encode_values({1, 2, 3, 2, 3, 4}, dtype));
    copy_to_device(mat2, encode_values({7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6}, dtype));

    EXPECT_EQ(aoti_torch_musa_bmm_out(output, mat1, mat2), Error::Ok);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    expect_values(copy_from_device(output, dtype), {58, 64, 31, 40});

    delete output;
    delete mat1;
    delete mat2;
  }
}

TEST_F(MusaAddmmTest, BmmBroadcastsReadonlyBatchOperands) {
  for (ScalarType dtype : {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    for (bool broadcast_mat1 : {true, false}) {
      SlimTensor* output = empty_tensor(dtype, {2, 2, 2}, {4, 2, 1});
      SlimTensor* mat1 = empty_tensor(
          dtype, {2, 2, 3}, {broadcast_mat1 ? 0 : 6, 3, 1});
      SlimTensor* mat2 = empty_tensor(
          dtype, {2, 3, 2}, {broadcast_mat1 ? 6 : 0, 2, 1});
      copy_to_device(
          mat1,
          encode_values(
              broadcast_mat1 ? std::vector<float>{1, 2, 3, 4, 5, 6}
                             : std::vector<float>{1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7},
              dtype));
      copy_to_device(
          mat2,
          encode_values(
              broadcast_mat1 ? std::vector<float>{7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6}
                             : std::vector<float>{7, 8, 9, 10, 11, 12},
              dtype));
      EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

      EXPECT_EQ(aoti_torch_musa_bmm_out(output, mat1, mat2), Error::Ok);
      EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      SCOPED_TRACE(::testing::Message()
                   << "dtype=" << static_cast<int>(dtype)
                   << " broadcast_mat1=" << broadcast_mat1);
      expect_values(
          copy_from_device(output, dtype),
          broadcast_mat1 ? std::vector<float>{58, 64, 139, 154, 22, 28, 49, 64}
                         : std::vector<float>{58, 64, 139, 154, 85, 94, 166, 184});

      delete output;
      delete mat1;
      delete mat2;
    }
  }
}

TEST_F(MusaAddmmTest, AddmmRejectsOverlappingRows) {
  SlimTensor* output = empty_tensor(ScalarType::Float, {2, 2}, {2, 1});
  SlimTensor* bias = empty_tensor(ScalarType::Float, {2}, {1});
  SlimTensor* mat1 = empty_tensor(ScalarType::Float, {2, 3}, {2, 1});
  SlimTensor* mat2 = empty_tensor(ScalarType::Float, {3, 2}, {2, 1});

  EXPECT_EQ(
      aoti_torch_musa_addmm_out(output, bias, mat1, mat2, 1.0, 1.0),
      Error::NotSupported);

  delete output;
  delete bias;
  delete mat1;
  delete mat2;
}

TEST_F(MusaAddmmTest, AddmmBroadcastsBiasAcrossFiveRows) {
  for (ScalarType dtype :
       {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    SlimTensor* output = empty_tensor(dtype, {5, 3}, {3, 1});
    SlimTensor* bias = empty_tensor(dtype, {3}, {1});
    SlimTensor* mat1 = empty_tensor(dtype, {5, 2}, {2, 1});
    SlimTensor* mat2 = empty_tensor(dtype, {2, 3}, {3, 1});
    copy_to_device(mat1, encode_values(std::vector<float>(10, 1.0f), dtype));
    copy_to_device(mat2, encode_values(std::vector<float>(6, 0.5f), dtype));
    copy_to_device(bias, encode_values({1.0f, 2.0f, 3.0f}, dtype));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    SCOPED_TRACE(::testing::Message() << "dtype=" << static_cast<int>(dtype));
    ASSERT_EQ(
        aoti_torch_musa_addmm_out(output, bias, mat1, mat2, 1.0, 1.0),
        Error::Ok);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    expect_values(
        copy_from_device(output, dtype),
        {2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4});

    delete output;
    delete bias;
    delete mat1;
    delete mat2;
  }
}

TEST_F(MusaAddmmTest, TransferThreeIndependentInputs) {
  for (ScalarType dtype :
       {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    SlimTensor* mat1 = empty_tensor(dtype, {2, 3}, {4, 1});
    SlimTensor* mat2 = empty_tensor(dtype, {3, 2}, {2, 1});
    SlimTensor* bias = empty_tensor(dtype, {2}, {1});
    copy_to_device(mat1, encode_values({1, 2, 3, 0, 4, 5, 6}, dtype));
    copy_to_device(mat2, encode_values({7, 8, 9, 10, 11, 12}, dtype));
    copy_to_device(bias, encode_values({1, 2}, dtype));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    SCOPED_TRACE(::testing::Message() << "dtype=" << static_cast<int>(dtype));
    expect_values(copy_from_device(mat1, dtype, 7), {1, 2, 3, 0, 4, 5, 6});
    expect_values(copy_from_device(mat2, dtype), {7, 8, 9, 10, 11, 12});
    expect_values(copy_from_device(bias, dtype), {1, 2});
    delete mat1;
    delete mat2;
    delete bias;
  }
}

TEST_F(MusaAddmmTest, VitAttentionProjection) {
  constexpr int64_t hidden_size = 768;
  constexpr int64_t projected_size = 3 * hidden_size;
  for (ScalarType dtype :
       {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    for (int64_t tokens : {197, 394}) {
      SlimTensor* output = empty_tensor(
          dtype, {tokens, projected_size}, {projected_size, 1});
      SlimTensor* bias = empty_tensor(dtype, {projected_size}, {1});
      SlimTensor* mat1 = empty_tensor(
          dtype, {tokens, hidden_size}, {hidden_size, 1});
      SlimTensor* mat2 = empty_tensor(
          dtype, {hidden_size, projected_size}, {1, hidden_size});
      copy_to_device(
          mat1, encode_values(std::vector<float>(tokens * hidden_size, 0.125f), dtype));
      copy_to_device(
          mat2,
          encode_values(
              std::vector<float>(hidden_size * projected_size, 0.015625f), dtype));
      copy_to_device(
          bias, encode_values(std::vector<float>(projected_size, 1.0f), dtype));
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

      for (int repeat = 0; repeat < 3; ++repeat) {
        SCOPED_TRACE(::testing::Message()
                     << "dtype=" << static_cast<int>(dtype)
                     << " tokens=" << tokens << " repeat=" << repeat);
        ASSERT_EQ(
            aoti_torch_musa_addmm_out(output, bias, mat1, mat2, 1.0, 1.0),
            Error::Ok);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto actual = copy_from_device(output, dtype);
        ASSERT_EQ(actual.size(), tokens * projected_size);
        size_t incorrect = 0;
        for (size_t index = 0; index < actual.size(); ++index) {
          if (!std::isfinite(actual[index]) ||
              std::abs(actual[index] - 2.5f) > 0.01f) {
            if (incorrect < 8) {
              ADD_FAILURE() << "mismatch at " << index << ": " << actual[index];
            }
            ++incorrect;
          }
        }
        EXPECT_EQ(incorrect, 0);
      }

      delete output;
      delete bias;
      delete mat1;
      delete mat2;
    }
  }
}

TEST_F(MusaAddmmTest, VitClassifierProjection) {
  constexpr int64_t hidden_size = 768;
  constexpr int64_t classes = 1000;
  for (ScalarType dtype :
       {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    SlimTensor* output = empty_tensor(dtype, {1, classes}, {classes, 1});
    SlimTensor* bias = empty_tensor(dtype, {classes}, {1});
    SlimTensor* mat1 = empty_tensor(dtype, {1, hidden_size}, {0, 1});
    SlimTensor* mat2 =
        empty_tensor(dtype, {hidden_size, classes}, {1, hidden_size});
    copy_to_device(
        mat1, encode_values(std::vector<float>(hidden_size, 0.125f), dtype));
    copy_to_device(
        mat2,
        encode_values(
            std::vector<float>(hidden_size * classes, 0.015625f), dtype));
    copy_to_device(bias, encode_values(std::vector<float>(classes, 1.0f), dtype));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (int repeat = 0; repeat < 4; ++repeat) {
      SCOPED_TRACE(::testing::Message()
                   << "dtype=" << static_cast<int>(dtype)
                   << " repeat=" << repeat);
      ASSERT_EQ(
          aoti_torch_musa_addmm_out(output, bias, mat1, mat2, 1.0, 1.0),
          Error::Ok);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      const auto actual = copy_from_device(output, dtype);
      ASSERT_EQ(actual.size(), classes);
      size_t incorrect = 0;
      for (size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) ||
            std::abs(actual[index] - 2.5f) > 0.01f) {
          if (incorrect < 8) {
            ADD_FAILURE() << "mismatch at " << index << ": " << actual[index];
          }
          ++incorrect;
        }
      }
      if (incorrect != 0) {
        const auto repeated = copy_from_device(output, dtype);
        const auto offsets = copy_from_device(bias, dtype);
        for (size_t index = classes - 16; index < classes; ++index) {
          ADD_FAILURE() << "tail index=" << index
                        << " original=" << actual[index]
                        << " repeated=" << repeated[index]
                        << " bias=" << offsets[index];
        }
      }
      EXPECT_EQ(incorrect, 0);
    }

    delete output;
    delete bias;
    delete mat1;
    delete mat2;
  }
}

TEST_F(MusaAddmmTest, PaddedClassifierProjection) {
  constexpr int64_t hidden_size = 768;
  constexpr int64_t classes = 1000;
  constexpr int64_t row_stride = 197 * hidden_size;
  for (ScalarType dtype :
       {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    SlimTensor* output = empty_tensor(dtype, {2, classes}, {classes, 1});
    SlimTensor* bias = empty_tensor(dtype, {classes}, {1});
    SlimTensor* mat1 = empty_tensor(dtype, {2, hidden_size}, {row_stride, 1});
    SlimTensor* mat2 =
        empty_tensor(dtype, {hidden_size, classes}, {1, hidden_size});
    std::vector<float> mat1_values(row_stride + hidden_size, 0.0f);
    std::fill(mat1_values.begin(), mat1_values.begin() + hidden_size, 0.125f);
    std::fill(
        mat1_values.begin() + row_stride,
        mat1_values.end(),
        0.125f);
    copy_to_device(mat1, encode_values(mat1_values, dtype));
    copy_to_device(
        mat2,
        encode_values(
            std::vector<float>(hidden_size * classes, 0.015625f), dtype));
    copy_to_device(bias, encode_values(std::vector<float>(classes, 1.0f), dtype));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (int repeat = 0; repeat < 4; ++repeat) {
      SCOPED_TRACE(::testing::Message()
                   << "dtype=" << static_cast<int>(dtype)
                   << " repeat=" << repeat);
      ASSERT_EQ(
          aoti_torch_musa_addmm_out(output, bias, mat1, mat2, 1.0, 1.0),
          Error::Ok);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      const auto actual = copy_from_device(output, dtype);
      ASSERT_EQ(actual.size(), 2 * classes);
      for (size_t index = 0; index < actual.size(); ++index) {
        EXPECT_TRUE(std::isfinite(actual[index]));
        EXPECT_NEAR(actual[index], 2.5f, 0.01f) << "index " << index;
      }
    }

    delete output;
    delete bias;
    delete mat1;
    delete mat2;
  }
}

TEST_F(MusaAddmmTest, VitAttentionScores) {
  constexpr int64_t tokens = 197;
  constexpr int64_t head_size = 64;
  for (ScalarType dtype :
       {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    for (int64_t batch : {12, 24}) {
      SlimTensor* output = empty_tensor(
          dtype, {batch, tokens, tokens}, {tokens * tokens, tokens, 1});
      SlimTensor* query = empty_tensor(
          dtype, {batch, tokens, head_size}, {tokens * head_size, head_size, 1});
      SlimTensor* key = empty_tensor(
          dtype, {batch, head_size, tokens}, {head_size * tokens, tokens, 1});
      copy_to_device(
          query,
          encode_values(std::vector<float>(batch * tokens * head_size, 0.125f), dtype));
      copy_to_device(
          key,
          encode_values(std::vector<float>(batch * head_size * tokens, 0.125f), dtype));

      for (int repeat = 0; repeat < 3; ++repeat) {
        SCOPED_TRACE(::testing::Message()
                     << "dtype=" << static_cast<int>(dtype)
                     << " batch=" << batch << " repeat=" << repeat);
        ASSERT_EQ(aoti_torch_musa_bmm_out(output, query, key), Error::Ok);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto actual = copy_from_device(output, dtype);
        ASSERT_EQ(actual.size(), batch * tokens * tokens);
        size_t incorrect = 0;
        for (size_t index = 0; index < actual.size(); ++index) {
          if (!std::isfinite(actual[index]) ||
              std::abs(actual[index] - 1.0f) > 0.01f) {
            if (incorrect < 8) {
              ADD_FAILURE() << "mismatch at " << index << ": " << actual[index];
            }
            ++incorrect;
          }
        }
        EXPECT_EQ(incorrect, 0);
      }

      delete output;
      delete query;
      delete key;
    }
  }
}

}
