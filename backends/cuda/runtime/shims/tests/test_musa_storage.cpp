#include <gtest/gtest.h>

#include <dlfcn.h>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include <executorch/backends/aoti/slim/factory/empty.h>
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
