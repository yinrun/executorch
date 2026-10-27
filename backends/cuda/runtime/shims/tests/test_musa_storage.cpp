#include <gtest/gtest.h>

#include <dlfcn.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/runtime/platform/platform.h>

namespace {

struct CopyCall {
  bool asynchronous;
  musaStream_t stream;
};

thread_local std::vector<CopyCall>* recorded_copies = nullptr;
thread_local size_t copy_depth = 0;

struct CopyTrace {
  std::vector<CopyCall> calls;

  CopyTrace() {
    recorded_copies = &calls;
  }

  ~CopyTrace() {
    recorded_copies = nullptr;
  }
};

template <typename Operation>
void expect_copy_stream(
    Operation operation,
    std::optional<musaStream_t> expected) {
  CopyTrace trace;
  operation();
  ASSERT_FALSE(trace.calls.empty());
  for (const auto& call : trace.calls) {
    EXPECT_EQ(call.asynchronous, expected.has_value());
    EXPECT_EQ(call.stream, expected.value_or(nullptr));
  }
}

namespace slim = executorch::backends::aoti::slim;
namespace slimc10 = slim::c10;
using executorch::backends::cuda::aoti_torch__alloc_from_pool;
using executorch::backends::cuda::aoti_torch__reinterpret_tensor;
using executorch::backends::cuda::clearCurrentCUDAStream;
using executorch::backends::cuda::CUDAStreamGuard;
using executorch::backends::cuda::setCurrentCUDAStream;
using executorch::extension::cuda::CallerStreamGuard;
using executorch::runtime::Error;

class MusaStorageTest : public ::testing::Test {
 protected:
  std::array<musaStream_t, 2> streams{};

  void SetUp() override {
    et_pal_init();
    int count = 0;
    if (musaGetDeviceCount(&count) != musaSuccess || count == 0) {
      GTEST_SKIP() << "MUSA device is not available";
    }
    ASSERT_EQ(musaSetDevice(0), musaSuccess);
    for (auto& stream : streams) {
      ASSERT_EQ(
          musaStreamCreateWithFlags(&stream, musaStreamNonBlocking),
          musaSuccess);
    }
    ASSERT_EQ(setCurrentCUDAStream(nullptr, 0), Error::Ok);
  }

  void TearDown() override {
    clearCurrentCUDAStream(0);
    for (auto stream : streams) {
      if (stream != nullptr) {
        EXPECT_EQ(musaStreamSynchronize(stream), musaSuccess);
        EXPECT_EQ(musaStreamDestroy(stream), musaSuccess);
      }
    }
  }

  void check_copies(std::optional<musaStream_t> expected) {
    for (auto device_type : {slimc10::kMUSA, slimc10::kCUDA}) {
      const slimc10::Device device(device_type, 0);
      auto source = slim::empty({1}, slimc10::ScalarType::Byte);
      *static_cast<uint8_t*>(source.data_ptr()) = 42;
      auto first = slim::empty({1}, slimc10::ScalarType::Byte, device);
      auto second = slim::empty({1}, slimc10::ScalarType::Byte, device);
      auto output = slim::empty({1}, slimc10::ScalarType::Byte);
      expect_copy_stream([&] { first.copy_(source); }, expected);
      expect_copy_stream([&] { second.copy_(first); }, expected);
      expect_copy_stream([&] { output.copy_(second); }, expected);
      EXPECT_EQ(*static_cast<uint8_t*>(output.data_ptr()), 42);
      expect_copy_stream(
          [&] { EXPECT_EQ(second.item<uint8_t>(), 42); }, expected);
    }
  }
};

TEST_F(MusaStorageTest, PoolViewsShareStorageAndOutlivePool) {
  auto guard = CUDAStreamGuard::create(streams[0], 0);
  ASSERT_TRUE(guard.ok());
  for (auto device_type : {slimc10::kMUSA, slimc10::kCUDA}) {
    auto pool = slim::empty(
        {128}, slimc10::ScalarType::Byte, slimc10::Device(device_type, 0));
    const int64_t sizes[] = {2, 3};
    const int64_t strides[] = {4, 1};
    slim::SlimTensor* handle = nullptr;
    ASSERT_EQ(
        aoti_torch__alloc_from_pool(&pool, 16, 6, 2, sizes, strides, &handle),
        Error::Ok);
    std::unique_ptr<slim::SlimTensor> view(handle);
    EXPECT_EQ(view->storage(), pool.storage());
    EXPECT_EQ(view->device_type(), device_type);
    EXPECT_EQ(view->dtype(), slimc10::ScalarType::Float);
    EXPECT_EQ(view->storage_offset(), 4);
    EXPECT_EQ(view->data_ptr(), static_cast<uint8_t*>(pool.data_ptr()) + 16);
    EXPECT_EQ(view->numel(), 6);
    EXPECT_FALSE(view->is_contiguous());
    EXPECT_EQ(
        musaMemsetAsync(pool.data_ptr(), 0x5A, 128, streams[0]), musaSuccess);
    pool.reset();
    auto host = slim::empty({128}, slimc10::ScalarType::Byte);
    slim::SlimTensor bytes(
        view->storage(), {128}, {1}, slimc10::ScalarType::Byte);
    host.copy_(bytes);
    const auto* values = static_cast<const uint8_t*>(host.data_ptr());
    for (size_t index = 0; index < 128; ++index) {
      EXPECT_EQ(values[index], 0x5A);
    }
  }
}

TEST_F(MusaStorageTest, PoolViewsSupportDtypesAndScalarOffsets) {
  auto pool = slim::empty(
      {128}, slimc10::ScalarType::Byte, slimc10::Device(slimc10::kMUSA, 0));
  for (int32_t dtype : {0, 1, 2, 3, 4, 5, 6, 11, 15}) {
    slim::SlimTensor* handle = nullptr;
    ASSERT_EQ(
        aoti_torch__alloc_from_pool(
            &pool, 16, dtype, 0, nullptr, nullptr, &handle),
        Error::Ok);
    std::unique_ptr<slim::SlimTensor> view(handle);
    EXPECT_EQ(view->numel(), 1);
    EXPECT_EQ(view->dtype(), static_cast<slimc10::ScalarType>(dtype));
    EXPECT_EQ(view->data_ptr(), static_cast<uint8_t*>(pool.data_ptr()) + 16);
    EXPECT_EQ(view->storage_offset(), 16 / view->itemsize());
  }
}

TEST_F(MusaStorageTest, ReinterpretPreservesPoolOffset) {
  auto pool = slim::empty(
      {128}, slimc10::ScalarType::Byte, slimc10::Device(slimc10::kMUSA, 0));
  const int64_t sizes[] = {2};
  const int64_t strides[] = {1};
  slim::SlimTensor* handle = nullptr;
  ASSERT_EQ(
      aoti_torch__alloc_from_pool(&pool, 16, 6, 1, sizes, strides, &handle),
      Error::Ok);
  std::unique_ptr<slim::SlimTensor> first(handle);
  for (int64_t increment : {0, 2, -2}) {
    ASSERT_EQ(
        aoti_torch__reinterpret_tensor(
            first.get(), 1, sizes, strides, increment, &handle),
        Error::Ok);
    std::unique_ptr<slim::SlimTensor> view(handle);
    EXPECT_EQ(view->storage(), pool.storage());
    EXPECT_EQ(view->storage_offset(), 4 + increment);
    EXPECT_EQ(
        view->data_ptr(),
        static_cast<uint8_t*>(pool.data_ptr()) + 16 + increment * 4);
  }
  for (int64_t increment : {-5L, std::numeric_limits<int64_t>::max()}) {
    handle = nullptr;
    EXPECT_EQ(
        aoti_torch__reinterpret_tensor(
            first.get(), 1, sizes, strides, increment, &handle),
        Error::InvalidArgument);
    EXPECT_EQ(handle, nullptr);
  }
}

TEST_F(MusaStorageTest, PoolViewsSupportEmptyAndBroadcastShapes) {
  auto pool = slim::empty(
      {64}, slimc10::ScalarType::Byte, slimc10::Device(slimc10::kMUSA, 0));
  const int64_t sizes[] = {3, 0, 2};
  const int64_t strides[] = {2, 2, 1};
  slim::SlimTensor* handle = nullptr;
  ASSERT_EQ(
      aoti_torch__alloc_from_pool(&pool, 64, 6, 3, sizes, strides, &handle),
      Error::Ok);
  std::unique_ptr<slim::SlimTensor> empty(handle);
  EXPECT_EQ(empty->numel(), 0);
  const int64_t broadcast_sizes[] = {3, 2};
  const int64_t broadcast_strides[] = {0, 1};
  ASSERT_EQ(
      aoti_torch__alloc_from_pool(
          &pool, 56, 6, 2, broadcast_sizes, broadcast_strides, &handle),
      Error::Ok);
  std::unique_ptr<slim::SlimTensor> broadcast(handle);
  EXPECT_EQ(broadcast->numel(), 6);
  EXPECT_EQ(broadcast->strides()[0], 0);
}

TEST_F(MusaStorageTest, PoolViewsRejectInvalidMetadata) {
  auto pool = slim::empty(
      {64}, slimc10::ScalarType::Byte, slimc10::Device(slimc10::kMUSA, 0));
  const int64_t sizes[] = {2, 3};
  const int64_t strides[] = {4, 1};
  const int64_t negative[] = {-1, 1};
  const int64_t overflow[] = {std::numeric_limits<int64_t>::max(), 3};
  const int64_t zero_strides[] = {0, 0};
  slim::SlimTensor* handle = nullptr;
  for (int32_t dtype : {-1, 7, 8, 9, 10, 12, 13, 14, 16, 256}) {
    EXPECT_EQ(
        aoti_torch__alloc_from_pool(
            &pool, 0, dtype, 2, sizes, strides, &handle),
        Error::InvalidArgument);
    EXPECT_EQ(handle, nullptr);
  }
  for (int64_t offset : {-4, 1, 40, 64, 68}) {
    EXPECT_EQ(
        aoti_torch__alloc_from_pool(
            &pool, offset, 6, 2, sizes, strides, &handle),
        Error::InvalidArgument);
    EXPECT_EQ(handle, nullptr);
  }
  EXPECT_EQ(
      aoti_torch__alloc_from_pool(nullptr, 0, 6, 2, sizes, strides, &handle),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch__alloc_from_pool(&pool, 0, 6, 2, sizes, strides, nullptr),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch__alloc_from_pool(&pool, 0, 6, -1, sizes, strides, &handle),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch__alloc_from_pool(&pool, 0, 6, 2, nullptr, strides, &handle),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch__alloc_from_pool(&pool, 0, 6, 2, sizes, nullptr, &handle),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch__alloc_from_pool(&pool, 0, 6, 2, negative, strides, &handle),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch__alloc_from_pool(&pool, 0, 6, 2, sizes, negative, &handle),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch__alloc_from_pool(&pool, 0, 6, 2, sizes, overflow, &handle),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch__alloc_from_pool(
          &pool, 0, 0, 2, overflow, zero_strides, &handle),
      Error::InvalidArgument);
  const int64_t empty_overflow[] = {std::numeric_limits<int64_t>::max(), 3, 0};
  const int64_t empty_strides[] = {0, 0, 0};
  EXPECT_EQ(
      aoti_torch__alloc_from_pool(
          &pool, 0, 0, 3, empty_overflow, empty_strides, &handle),
      Error::InvalidArgument);
  auto offset_pool = pool.as_strided({32}, {1}, 8);
  EXPECT_EQ(
      aoti_torch__alloc_from_pool(
          &offset_pool, 0, 6, 2, sizes, strides, &handle),
      Error::InvalidArgument);
  slim::SlimTensor undefined;
  EXPECT_EQ(
      aoti_torch__alloc_from_pool(&undefined, 0, 6, 2, sizes, strides, &handle),
      Error::InvalidArgument);
}

TEST_F(MusaStorageTest, BlockingCopiesUseSelectedStream) {
  auto guard = CUDAStreamGuard::create(streams[0], 0);
  ASSERT_TRUE(guard.ok());
  check_copies(streams[0]);
}

TEST_F(MusaStorageTest, CallerStreamOverridesSelection) {
  auto guard = CUDAStreamGuard::create(streams[0], 0);
  ASSERT_TRUE(guard.ok());
  {
    CallerStreamGuard caller(streams[1]);
    check_copies(streams[1]);
  }
  {
    CallerStreamGuard caller(nullptr);
    check_copies(std::optional<musaStream_t>{nullptr});
  }
  check_copies(streams[0]);
}

TEST_F(MusaStorageTest, ExplicitDefaultAndUnsetStreams) {
  ASSERT_EQ(setCurrentCUDAStream(nullptr, 0), Error::Ok);
  check_copies(std::optional<musaStream_t>{nullptr});
  clearCurrentCUDAStream(0);
  auto source = slim::empty({1}, slimc10::ScalarType::Byte);
  *static_cast<uint8_t*>(source.data_ptr()) = 42;
  auto tensor = slim::empty(
      {1}, slimc10::ScalarType::Byte, slimc10::Device(slimc10::kMUSA, 0));
  clearCurrentCUDAStream(0);
  expect_copy_stream([&] { tensor.copy_(source); }, std::nullopt);
  expect_copy_stream(
      [&] { EXPECT_EQ(tensor.item<uint8_t>(), 42); },
      std::optional<musaStream_t>{nullptr});
  EXPECT_FALSE(
      executorch::backends::cuda::peekCurrentCUDAStream(0).has_value());
  ASSERT_EQ(setCurrentCUDAStream(nullptr, 0), Error::Ok);
}

}

extern "C" musaError_t MUSARTAPI musaMemcpy(
    void* destination,
    const void* source,
    size_t bytes,
    musaMemcpyKind kind) {
  using Copy = decltype(&musaMemcpy);
  static const auto copy =
      reinterpret_cast<Copy>(dlsym(RTLD_NEXT, "musaMemcpy"));
  if (copy == nullptr) {
    return musaErrorUnknown;
  }
  if (copy_depth++ == 0 && recorded_copies != nullptr) {
    recorded_copies->push_back({false, nullptr});
  }
  const auto result = copy(destination, source, bytes, kind);
  --copy_depth;
  return result;
}

extern "C" musaError_t MUSARTAPI musaMemcpyAsync(
    void* destination,
    const void* source,
    size_t bytes,
    musaMemcpyKind kind,
    musaStream_t stream) {
  using Copy = decltype(&musaMemcpyAsync);
  static const auto copy =
      reinterpret_cast<Copy>(dlsym(RTLD_NEXT, "musaMemcpyAsync"));
  if (copy == nullptr) {
    return musaErrorUnknown;
  }
  if (copy_depth++ == 0 && recorded_copies != nullptr) {
    recorded_copies->push_back({true, stream});
  }
  const auto result = copy(destination, source, bytes, kind, stream);
  --copy_depth;
  return result;
}
