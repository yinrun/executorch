#include <gtest/gtest.h>

#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/cuda/runtime/shims/cuda_guard.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/extension/aten_util/aten_device.h>
#include <executorch/extension/cuda/cuda_allocator.h>
#include <executorch/runtime/core/device_memory_buffer.h>
#include <executorch/runtime/platform/platform.h>

namespace {

namespace slim = executorch::backends::aoti::slim;
namespace slimc10 = slim::c10;
using executorch::extension::cuda::CudaAllocator;
using executorch::runtime::DeviceMemoryBuffer;
using executorch::runtime::Error;
using executorch::runtime::etensor::DeviceType;

class MusaDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "MUSA device is not available";
    }
  }
};

TEST(MusaDeviceTypeTest, WireAndAotiDeviceTypesAreDistinct) {
  et_pal_init();
  EXPECT_EQ(static_cast<int>(DeviceType::MUSA), 2);
  EXPECT_EQ(static_cast<int>(slimc10::DeviceType::MUSA), 2);
  EXPECT_EQ(slimc10::device_type_from_aoti(20), slimc10::kMUSA);
  EXPECT_EQ(slimc10::device_type_from_aoti(1), slimc10::kMUSA);
  EXPECT_EQ(slimc10::device_type_to_aoti(slimc10::kMUSA), 20);
  EXPECT_EQ(slimc10::device_type_to_aoti(slimc10::kCUDA), 20);
  EXPECT_EQ(slimc10::device_type_from_aoti(0), slimc10::kCPU);
  EXPECT_EQ(slimc10::device_type_to_aoti(slimc10::kCPU), 0);
  EXPECT_DEATH(slimc10::device_type_from_aoti(2), "");
  const slimc10::Device device("musa:12");
  EXPECT_TRUE(device.is_musa());
  EXPECT_TRUE(device.is_gpu());
  EXPECT_FALSE(device.is_cuda());
  EXPECT_EQ(device.index(), 12);
  EXPECT_EQ(device.str(), "musa:12");
  EXPECT_EQ(slimc10::Device("musa:127").index(), 127);
  EXPECT_DEATH(slimc10::Device("musa:"), "");
  EXPECT_DEATH(slimc10::Device("musa:-1"), "");
  EXPECT_DEATH(slimc10::Device("musa:128"), "");
  EXPECT_DEATH(slimc10::Device("musa:99999999999999999999"), "");
  const auto torch_device = executorch::extension::executorch_to_torch_device(
      executorch::runtime::etensor::Device(DeviceType::MUSA, 3));
  EXPECT_EQ(torch_device.type(), ::c10::DeviceType::PrivateUse1);
  EXPECT_EQ(torch_device.index(), 3);
  const auto roundtrip =
      executorch::extension::torch_to_executorch_device(torch_device);
  ASSERT_TRUE(roundtrip.has_value());
  EXPECT_EQ(roundtrip->type(), DeviceType::MUSA);
  EXPECT_EQ(roundtrip->index(), 3);
}

TEST_F(MusaDeviceTest, NativeAndLegacyAllocatorRegistrations) {
  auto& native = CudaAllocator::instance();
  auto& legacy = CudaAllocator::legacy_cuda_instance();
  EXPECT_EQ(native.device_type(), DeviceType::MUSA);
  EXPECT_EQ(legacy.device_type(), DeviceType::CUDA);
  for (auto* allocator : {&native, &legacy}) {
    if (executorch::runtime::get_device_allocator(allocator->device_type()) ==
        nullptr) {
      executorch::runtime::register_device_allocator(allocator);
    }
    auto buffer = DeviceMemoryBuffer::create(16, allocator->device_type());
    ASSERT_TRUE(buffer.ok());
    float source[] = {1, 2, 3, 4};
    float actual[4] = {};
    ASSERT_EQ(
        allocator->copy_host_to_device(
            buffer->data(), source, sizeof(source), 0),
        Error::Ok);
    ASSERT_EQ(
        allocator->copy_device_to_host(
            actual, buffer->data(), sizeof(actual), 0),
        Error::Ok);
    for (size_t index = 0; index < 4; ++index) {
      EXPECT_EQ(actual[index], source[index]);
    }
  }
}

TEST_F(MusaDeviceTest, SlimTensorAllocationCopyAndAbiRoundTrip) {
  for (int32_t abi_type : {20, 1}) {
    slim::SlimTensor* raw = nullptr;
    const int64_t sizes[] = {1};
    const int64_t strides[] = {1};
    ASSERT_EQ(
        executorch::backends::cuda::aoti_torch_empty_strided(
            1,
            sizes,
            strides,
            static_cast<int32_t>(slimc10::ScalarType::Float),
            abi_type,
            0,
            &raw),
        Error::Ok);
    std::unique_ptr<slim::SlimTensor> tensor(raw);
    ASSERT_TRUE(tensor->is_musa());
    EXPECT_FALSE(tensor->is_cuda());
    float value = 7.5f;
    ASSERT_EQ(
        cudaMemcpy(
            tensor->data_ptr(), &value, sizeof(value), cudaMemcpyHostToDevice),
        cudaSuccess);
    EXPECT_EQ(tensor->item<float>(), value);
    auto cloned = tensor->clone_contiguous();
    EXPECT_TRUE(cloned.is_musa());
    EXPECT_EQ(cloned.item<float>(), value);
    int32_t reported = -1;
    ASSERT_EQ(
        executorch::backends::aoti::aoti_torch_get_device_type(
            tensor.get(), &reported),
        Error::Ok);
    EXPECT_EQ(reported, 20);
  }
}

TEST_F(MusaDeviceTest, AotiGuardsRestoreNestedAndDefaultStreams) {
  using namespace executorch::backends::cuda;
  EXPECT_EQ(
      executorch::backends::aoti::aoti_torch_device_type_privateuse1(), 20);
  CUDAGuardHandle device_guard = nullptr;
  ASSERT_EQ(aoti_torch_create_musa_guard(0, &device_guard), Error::Ok);
  EXPECT_EQ(aoti_torch_musa_guard_set_index(device_guard, 0), Error::Ok);
  ASSERT_EQ(aoti_torch_delete_musa_guard(device_guard), Error::Ok);
  EXPECT_EQ(aoti_torch_create_musa_guard(0, nullptr), Error::InvalidArgument);
  EXPECT_EQ(aoti_torch_delete_musa_guard(nullptr), Error::InvalidArgument);
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(setCurrentCUDAStream(nullptr, 0), Error::Ok);
  CUDAStreamGuardHandle outer = nullptr;
  CUDAStreamGuardHandle inner = nullptr;
  ASSERT_EQ(aoti_torch_create_musa_stream_guard(stream, 0, &outer), Error::Ok);
  void* current = nullptr;
  ASSERT_EQ(aoti_torch_get_current_musa_stream(0, &current), Error::Ok);
  EXPECT_EQ(current, stream);
  ASSERT_EQ(aoti_torch_create_musa_stream_guard(nullptr, 0, &inner), Error::Ok);
  ASSERT_EQ(aoti_torch_get_current_musa_stream(0, &current), Error::Ok);
  EXPECT_EQ(current, nullptr);
  ASSERT_EQ(aoti_torch_delete_musa_stream_guard(inner), Error::Ok);
  ASSERT_EQ(aoti_torch_get_current_musa_stream(0, &current), Error::Ok);
  EXPECT_EQ(current, stream);
  ASSERT_EQ(aoti_torch_delete_musa_stream_guard(outer), Error::Ok);
  ASSERT_EQ(aoti_torch_get_current_musa_stream(0, &current), Error::Ok);
  EXPECT_EQ(current, nullptr);
  EXPECT_EQ(
      aoti_torch_create_musa_stream_guard(stream, 256, &outer),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_create_musa_stream_guard(stream, 0, nullptr),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_get_current_musa_stream(256, &current),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_get_current_musa_stream(0, nullptr), Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_delete_musa_stream_guard(nullptr), Error::InvalidArgument);
  clearCurrentCUDAStream(0);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
}
