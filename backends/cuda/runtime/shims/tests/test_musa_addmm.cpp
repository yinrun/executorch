#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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
  return static_cast<uint16_t>(
      ((bits >> 16) & 0x8000) | ((exponent - 112) << 10) | (mantissa >> 13));
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
  uint32_t bits = (static_cast<uint32_t>(value & 0x8000) << 16) |
      ((exponent + 112) << 23) | (mantissa << 13);
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

TEST_F(MusaAddmmTest, RejectsMissingStorage) {
  SlimTensor undefined;
  EXPECT_EQ(
      aoti_torch_musa_mm_out(nullptr, &undefined, &undefined),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_musa_mm_out(&undefined, &undefined, &undefined),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_musa_bmm_out(&undefined, &undefined, &undefined),
      Error::InvalidArgument);
}

TEST_F(MusaAddmmTest, RejectsInvalidStorageSpansBeforeWriting) {
  const auto dtype = ScalarType::Float;
  std::unique_ptr<SlimTensor> output(empty_tensor(dtype, {2, 2}, {2, 1}));
  std::unique_ptr<SlimTensor> left(empty_tensor(dtype, {2, 2}, {2, 1}));
  std::unique_ptr<SlimTensor> right(empty_tensor(dtype, {2, 2}, {2, 1}));
  std::unique_ptr<SlimTensor> bias(empty_tensor(dtype, {2, 2}, {2, 1}));
  const std::vector<float> sentinel{17, 19, 23, 29};
  copy_to_device(output.get(), encode_values(sentinel, dtype));
  for (auto* operand : {output.get(), left.get(), right.get(), bias.get()}) {
    for (int64_t offset : {-1, 1, 4}) {
      operand->set_sizes_and_strides(
          std::array<int64_t, 2>{2, 2}, std::array<int64_t, 2>{2, 1}, offset);
      EXPECT_EQ(
          aoti_torch_musa_addmm_out(
              output.get(), bias.get(), left.get(), right.get(), 1, 1),
          Error::InvalidArgument);
      operand->set_sizes_and_strides(
          std::array<int64_t, 2>{2, 2}, std::array<int64_t, 2>{2, 1}, 0);
      EXPECT_EQ(copy_from_device(output.get(), dtype), sentinel);
    }
  }
  left->set_sizes_and_strides(
      std::array<int64_t, 2>{2, 2},
      std::array<int64_t, 2>{std::numeric_limits<int64_t>::max(), 1});
  EXPECT_EQ(
      aoti_torch_musa_mm_out(output.get(), left.get(), right.get()),
      Error::InvalidArgument);
  EXPECT_EQ(copy_from_device(output.get(), dtype), sentinel);
}

TEST_F(MusaAddmmTest, RejectsNullOrMisalignedData) {
  using executorch::backends::aoti::slim::MaybeOwningStorage;
  using executorch::backends::aoti::slim::Storage;
  const auto dtype = ScalarType::Float;
  std::unique_ptr<SlimTensor> output(empty_tensor(dtype, {2, 2}, {2, 1}));
  std::unique_ptr<SlimTensor> right(empty_tensor(dtype, {2, 2}, {2, 1}));
  for (void* address :
       {static_cast<void*>(nullptr),
        static_cast<void*>(static_cast<char*>(right->data_ptr()) + 1)}) {
    SlimTensor invalid(
        Storage(new MaybeOwningStorage(right->device(), address, 16)),
        std::array<int64_t, 2>{2, 2},
        std::array<int64_t, 2>{2, 1},
        dtype);
    EXPECT_EQ(
        aoti_torch_musa_mm_out(output.get(), &invalid, right.get()),
        Error::InvalidArgument);
  }
}

TEST_F(MusaAddmmTest, RejectsOutputOperandAliasesAcrossStorageWrappers) {
  using executorch::backends::aoti::slim::MaybeOwningStorage;
  using executorch::backends::aoti::slim::Storage;
  const auto dtype = ScalarType::Float;
  std::unique_ptr<SlimTensor> left(empty_tensor(dtype, {2, 2}, {2, 1}));
  std::unique_ptr<SlimTensor> right(empty_tensor(dtype, {2, 2}, {2, 1}));
  const std::vector<float> values{1, 2, 3, 4};
  copy_to_device(left.get(), encode_values(values, dtype));
  copy_to_device(right.get(), encode_values(values, dtype));
  for (auto* operand : {left.get(), right.get()}) {
    EXPECT_EQ(
        aoti_torch_musa_mm_out(operand, left.get(), right.get()),
        Error::InvalidArgument);
    SlimTensor alias(
        Storage(new MaybeOwningStorage(
            operand->device(), operand->data_ptr(), operand->nbytes())),
        std::array<int64_t, 2>{2, 2},
        std::array<int64_t, 2>{2, 1},
        dtype);
    EXPECT_EQ(
        aoti_torch_musa_mm_out(&alias, left.get(), right.get()),
        Error::InvalidArgument);
    EXPECT_EQ(copy_from_device(operand, dtype), values);
  }
}

TEST_F(MusaAddmmTest, DisjointOffsetViewsPreserveInputsAndSentinels) {
  for (auto dtype :
       {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    std::unique_ptr<SlimTensor> backing(empty_tensor(dtype, {16}, {1}));
    const std::vector<float> initial{
        17, 17, 1, 2, 3, 4, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17};
    copy_to_device(backing.get(), encode_values(initial, dtype));
    SlimTensor left(
        backing->storage(),
        std::array<int64_t, 2>{2, 2},
        std::array<int64_t, 2>{2, 1},
        dtype,
        2);
    SlimTensor output(
        backing->storage(),
        std::array<int64_t, 2>{2, 2},
        std::array<int64_t, 2>{2, 1},
        dtype,
        9);
    ASSERT_EQ(aoti_torch_musa_mm_out(&output, &left, &left), Error::Ok);
    auto expected = initial;
    const std::vector<float> product{7, 10, 15, 22};
    std::copy(product.begin(), product.end(), expected.begin() + 9);
    EXPECT_EQ(copy_from_device(backing.get(), dtype), expected);
  }
}

TEST_F(MusaAddmmTest, BiasAliasingRequiresExactMatrixView) {
  const auto dtype = ScalarType::Float;
  std::unique_ptr<SlimTensor> backing(empty_tensor(dtype, {8}, {1}));
  std::unique_ptr<SlimTensor> left(empty_tensor(dtype, {2, 2}, {2, 1}));
  std::unique_ptr<SlimTensor> right(empty_tensor(dtype, {2, 2}, {2, 1}));
  copy_to_device(left.get(), encode_values({1, 0, 0, 1}, dtype));
  copy_to_device(right.get(), encode_values({2, 4, 6, 8}, dtype));
  SlimTensor output(
      backing->storage(),
      std::array<int64_t, 2>{2, 2},
      std::array<int64_t, 2>{2, 1},
      dtype);
  SlimTensor partial(
      backing->storage(),
      std::array<int64_t, 2>{2, 2},
      std::array<int64_t, 2>{2, 1},
      dtype,
      1);
  SlimTensor vector(
      backing->storage(),
      std::array<int64_t, 1>{2},
      std::array<int64_t, 1>{1},
      dtype);
  copy_to_device(
      backing.get(), encode_values(std::vector<float>(8, 10), dtype));
  for (auto* bias : {&partial, &vector}) {
    EXPECT_EQ(
        aoti_torch_musa_addmm_out(&output, bias, left.get(), right.get(), 1, 1),
        Error::InvalidArgument);
    EXPECT_EQ(
        copy_from_device(backing.get(), dtype), std::vector<float>(8, 10));
  }
  for (double beta : {0.0, 0.5, 1.0}) {
    copy_to_device(
        backing.get(), encode_values(std::vector<float>(8, 10), dtype));
    SlimTensor exact(
        backing->storage(),
        std::array<int64_t, 2>{2, 2},
        std::array<int64_t, 2>{2, 1},
        dtype);
    ASSERT_EQ(
        aoti_torch_musa_addmm_out(
            &output, &exact, left.get(), right.get(), beta, 1),
        Error::Ok);
    EXPECT_EQ(
        copy_from_device(&output, dtype),
        (std::vector<float>{
            float(2 + 10 * beta),
            float(4 + 10 * beta),
            float(6 + 10 * beta),
            float(8 + 10 * beta)}));
  }
}

TEST_F(MusaAddmmTest, EmptyInnerDimensionScalesBiasAndIgnoresAlpha) {
  for (auto dtype :
       {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    std::unique_ptr<SlimTensor> output(empty_tensor(dtype, {2, 3}, {3, 1}));
    std::unique_ptr<SlimTensor> left(empty_tensor(dtype, {2, 0}, {1, 1}));
    std::unique_ptr<SlimTensor> right(empty_tensor(dtype, {0, 3}, {3, 1}));
    std::unique_ptr<SlimTensor> bias(empty_tensor(dtype, {3}, {1}));
    copy_to_device(bias.get(), encode_values({2, 4, 6}, dtype));
    for (double beta : {0.0, 0.5, 1.0, 2.0}) {
      ASSERT_EQ(
          aoti_torch_musa_addmm_out(
              output.get(),
              bias.get(),
              left.get(),
              right.get(),
              beta,
              std::numeric_limits<double>::quiet_NaN()),
          Error::Ok);
      EXPECT_EQ(
          copy_from_device(output.get(), dtype),
          (std::vector<float>{
              float(2 * beta),
              float(4 * beta),
              float(6 * beta),
              float(2 * beta),
              float(4 * beta),
              float(6 * beta)}));
    }
    copy_to_device(
        output.get(), encode_values(std::vector<float>(6, 4), dtype));
    ASSERT_EQ(
        aoti_torch_musa_addmm_out(
            output.get(), output.get(), left.get(), right.get(), 0.5, 1),
        Error::Ok);
    EXPECT_EQ(copy_from_device(output.get(), dtype), std::vector<float>(6, 2));
    std::vector<uint8_t> nan_values(6 * itemsize(dtype), 255);
    copy_to_device(output.get(), nan_values);
    ASSERT_EQ(
        aoti_torch_musa_mm_out(output.get(), left.get(), right.get()),
        Error::Ok);
    EXPECT_EQ(copy_from_device(output.get(), dtype), std::vector<float>(6, 0));
  }
}

TEST_F(MusaAddmmTest, EmptyOutputIsNoOp) {
  for (const auto& shape :
       {std::array<int64_t, 2>{0, 3}, std::array<int64_t, 2>{2, 0}}) {
    const auto rows = shape[0];
    const auto columns = shape[1];
    std::unique_ptr<SlimTensor> output(empty_tensor(
        ScalarType::BFloat16,
        {rows, columns},
        {std::max<int64_t>(columns, 1), 1}));
    std::unique_ptr<SlimTensor> left(
        empty_tensor(ScalarType::BFloat16, {rows, 4}, {4, 1}));
    std::unique_ptr<SlimTensor> right(empty_tensor(
        ScalarType::BFloat16,
        {4, columns},
        {std::max<int64_t>(columns, 1), 1}));
    EXPECT_EQ(
        aoti_torch_musa_mm_out(output.get(), left.get(), right.get()),
        Error::Ok);
  }
}

TEST_F(MusaAddmmTest, BmmRejectsInvalidSpansAndAliases) {
  const auto dtype = ScalarType::BFloat16;
  std::unique_ptr<SlimTensor> output(empty_tensor(dtype, {2, 2, 2}, {4, 2, 1}));
  std::unique_ptr<SlimTensor> left(empty_tensor(dtype, {2, 2, 2}, {4, 2, 1}));
  std::unique_ptr<SlimTensor> right(empty_tensor(dtype, {2, 2, 2}, {4, 2, 1}));
  copy_to_device(output.get(), encode_values(std::vector<float>(8, 17), dtype));
  for (auto* operand : {output.get(), left.get(), right.get()}) {
    operand->set_sizes_and_strides(
        std::array<int64_t, 3>{2, 2, 2}, std::array<int64_t, 3>{4, 2, 1}, 1);
    EXPECT_EQ(
        aoti_torch_musa_bmm_out(output.get(), left.get(), right.get()),
        Error::InvalidArgument);
    operand->set_sizes_and_strides(
        std::array<int64_t, 3>{2, 2, 2}, std::array<int64_t, 3>{4, 2, 1}, 0);
  }
  EXPECT_EQ(
      aoti_torch_musa_bmm_out(left.get(), left.get(), right.get()),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_musa_bmm_out(right.get(), left.get(), right.get()),
      Error::InvalidArgument);
  EXPECT_EQ(copy_from_device(output.get(), dtype), std::vector<float>(8, 17));
}

TEST_F(MusaAddmmTest, BmmEmptyProducts) {
  for (auto dtype :
       {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    for (int64_t batch : {0, 2}) {
      std::unique_ptr<SlimTensor> output(
          empty_tensor(dtype, {batch, 2, 3}, {6, 3, 1}));
      std::unique_ptr<SlimTensor> left(
          empty_tensor(dtype, {batch, 2, 0}, {2, 1, 1}));
      std::unique_ptr<SlimTensor> right(
          empty_tensor(dtype, {batch, 0, 3}, {3, 3, 1}));
      if (batch) {
        copy_to_device(
            output.get(), encode_values(std::vector<float>(12, 17), dtype));
      }
      ASSERT_EQ(
          aoti_torch_musa_bmm_out(output.get(), left.get(), right.get()),
          Error::Ok);
      if (batch) {
        EXPECT_EQ(
            copy_from_device(output.get(), dtype), std::vector<float>(12, 0));
      }
    }
  }
}

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
            setCurrentCUDAStream(streams[(index / 4) % streams.size()], 0),
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

TEST_F(MusaAddmmTest, BlasStreamRecreation) {
  std::thread worker([] {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    for (int repeat = 0; repeat < 16; ++repeat) {
      cudaStream_t stream = nullptr;
      ASSERT_EQ(
          musaStreamCreateWithFlags(&stream, musaStreamNonBlocking),
          musaSuccess);
      ASSERT_EQ(setCurrentCUDAStream(stream, 0), Error::Ok);
      for (auto dtype :
           {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
        run_addmm(dtype, repeat % 2 == 0, false, true);
        run_bmm(dtype, true, true, false);
      }
      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      clearCurrentCUDAStream(0);
      EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
  });
  worker.join();
}

TEST_F(MusaAddmmTest, AddmmDtypesAndTransposedMat2) {
  for (ScalarType dtype :
       {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
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

TEST_F(MusaAddmmTest, BmmPacksInterleavedAndPaddedMatrices) {
  const std::array<std::vector<int64_t>, 4> left_layouts = {
      std::vector<int64_t>{4, 12, 1}, {18, 6, 1}, {3, 1, 9}, {0, 6, 1}};
  const std::array<std::vector<int64_t>, 4> right_layouts = {
      std::vector<int64_t>{4, 1, 12}, {16, 1, 8}, {2, 6, 1}, {12, 3, 1}};
  for (ScalarType dtype : {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    for (size_t layout = 0; layout < left_layouts.size(); ++layout) {
      for (int64_t offset : {0, 3}) {
        const auto& left_strides = left_layouts[layout];
        const auto& right_strides = right_layouts[layout];
        const int64_t left_size = offset + 1 + 2 * left_strides[0] +
            2 * left_strides[1] + 3 * left_strides[2];
        const int64_t right_size = offset + 1 + 2 * right_strides[0] +
            3 * right_strides[1] + right_strides[2];
        std::vector<float> left_values(left_size, -19);
        std::vector<float> right_values(right_size, -23);
        std::vector<float> expected(18, 0);
        for (int64_t batch = 0; batch < 3; ++batch) {
          for (int64_t row = 0; row < 3; ++row) {
            for (int64_t inner = 0; inner < 4; ++inner) {
              left_values[offset + batch * left_strides[0] + row * left_strides[1] +
                          inner * left_strides[2]] = row - inner;
            }
          }
          for (int64_t inner = 0; inner < 4; ++inner) {
            for (int64_t column = 0; column < 2; ++column) {
              right_values[offset + batch * right_strides[0] + inner * right_strides[1] +
                           column * right_strides[2]] = inner + column + batch;
            }
          }
          for (int64_t row = 0; row < 3; ++row) {
            for (int64_t column = 0; column < 2; ++column) {
              for (int64_t inner = 0; inner < 4; ++inner) {
                expected[batch * 6 + row * 2 + column] +=
                    (row - inner) * (inner + column + batch);
              }
            }
          }
        }
        std::unique_ptr<SlimTensor> left(empty_tensor(dtype, {left_size}, {1}));
        std::unique_ptr<SlimTensor> right(empty_tensor(dtype, {right_size}, {1}));
        std::unique_ptr<SlimTensor> output(empty_tensor(dtype, {3, 3, 2}, {6, 2, 1}));
        copy_to_device(left.get(), encode_values(left_values, dtype));
        copy_to_device(right.get(), encode_values(right_values, dtype));
        left->set_sizes_and_strides(
            std::array<int64_t, 3>{3, 3, 4},
            executorch::runtime::IntArrayRef(left_strides.data(), left_strides.size()),
            offset);
        right->set_sizes_and_strides(
            std::array<int64_t, 3>{3, 4, 2},
            executorch::runtime::IntArrayRef(right_strides.data(), right_strides.size()),
            offset);
        ASSERT_EQ(aoti_torch_musa_bmm_out(output.get(), left.get(), right.get()), Error::Ok);
        EXPECT_EQ(copy_from_device(output.get(), dtype), expected);
        left->set_sizes_and_strides(std::array<int64_t, 1>{left_size}, std::array<int64_t, 1>{1}, 0);
        right->set_sizes_and_strides(std::array<int64_t, 1>{right_size}, std::array<int64_t, 1>{1}, 0);
        EXPECT_EQ(copy_from_device(left.get(), dtype), left_values);
        EXPECT_EQ(copy_from_device(right.get(), dtype), right_values);
      }
    }
  }
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

TEST_F(MusaAddmmTest, BmmSingletonRowStridePreservesRounding) {
  constexpr int64_t batch = 8;
  constexpr int64_t reduction = 64;
  constexpr int64_t columns = 256;
  uint32_t state = 109;
  auto generate = [&state](size_t count) {
    std::vector<float> values(count);
    for (auto& value : values) {
      state = state * 1664525u + 1013904223u;
      value = std::ldexp(
          static_cast<float>((state & 255u) + 1u),
          static_cast<int>((state >> 8) % 10u) - 14);
      if (state & 0x80000000u) {
        value = -value;
      }
    }
    return values;
  };
  const auto left_values = generate(batch * reduction);
  const auto right_values = generate(batch * reduction * columns);
  for (ScalarType dtype : {ScalarType::Float, ScalarType::Half, ScalarType::BFloat16}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    std::unique_ptr<SlimTensor> left(empty_tensor(
        dtype, {batch, 1, reduction}, {reduction, reduction, 1}));
    std::unique_ptr<SlimTensor> zero_stride_left(empty_tensor(
        dtype, {batch, 1, reduction}, {reduction, 0, 1}));
    std::unique_ptr<SlimTensor> right(empty_tensor(
        dtype, {batch, reduction, columns}, {reduction * columns, columns, 1}));
    std::unique_ptr<SlimTensor> expected(empty_tensor(
        dtype, {batch, 1, columns}, {columns, columns, 1}));
    std::unique_ptr<SlimTensor> actual(empty_tensor(
        dtype, {batch, 1, columns}, {columns, columns, 1}));
    copy_to_device(left.get(), encode_values(left_values, dtype));
    copy_to_device(zero_stride_left.get(), encode_values(left_values, dtype));
    copy_to_device(right.get(), encode_values(right_values, dtype));
    ASSERT_EQ(aoti_torch_musa_bmm_out(expected.get(), left.get(), right.get()), Error::Ok);
    ASSERT_EQ(aoti_torch_musa_bmm_out(actual.get(), zero_stride_left.get(), right.get()), Error::Ok);
    EXPECT_EQ(copy_from_device(actual.get(), dtype), copy_from_device(expected.get(), dtype));
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
