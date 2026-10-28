#include <dlfcn.h>
#include <executorch/extension/cuda/cuda_allocator.h>
#include <executorch/runtime/platform/platform.h>
#include <gtest/gtest.h>
#include <musa_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using executorch::extension::cuda::CudaAllocator;
using executorch::runtime::Error;

namespace {
std::atomic<int> fail_allocations{0};
}

extern "C" musaError_t musaMalloc(void** pointer, size_t bytes) {
  if (fail_allocations.exchange(0) != 0) {
    return musaErrorMemoryAllocation;
  }
  static auto allocate = reinterpret_cast<musaError_t (*)(void**, size_t)>(
      dlsym(RTLD_NEXT, "musaMalloc"));
  return allocate == nullptr ? musaErrorUnknown : allocate(pointer, bytes);
}

class MusaAllocatorTest : public testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    ASSERT_EQ(musaGetDeviceCount(&devices_), musaSuccess);
    ASSERT_GT(devices_, 0);
    ASSERT_EQ(musaSetDevice(0), musaSuccess);
    ASSERT_EQ(
        CudaAllocator::configure_musa_allocation_cache(1 << 20), Error::Ok);
  }

  void TearDown() override {
    for (int device = 0; device < devices_; ++device) {
      EXPECT_EQ(musaSetDevice(device), musaSuccess);
      EXPECT_EQ(musaDeviceSynchronize(), musaSuccess);
    }
    CudaAllocator::release_cached_memory(-1);
    const auto stats = CudaAllocator::musa_allocation_cache_stats();
    EXPECT_EQ(stats.cached_bytes, 0);
    EXPECT_EQ(stats.cached_blocks, 0);
    EXPECT_EQ(stats.live_blocks, 0);
    EXPECT_EQ(CudaAllocator::configure_musa_allocation_cache(0), Error::Ok);
    EXPECT_EQ(musaSetDevice(0), musaSuccess);
  }

  int devices_ = 0;
};

TEST_F(MusaAllocatorTest, DisabledPreservesUncachedBehavior) {
  ASSERT_EQ(CudaAllocator::configure_musa_allocation_cache(0), Error::Ok);
  auto allocation = CudaAllocator::instance().allocate(4096, 0);
  ASSERT_TRUE(allocation.ok());
  CudaAllocator::deallocate_async(allocation.get(), 0, nullptr);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  EXPECT_EQ(CudaAllocator::musa_allocation_cache_stats().allocation_count, 0);
}

TEST_F(MusaAllocatorTest, ReusesOnlyAfterCompletionAndTrims) {
  auto first = CudaAllocator::instance().allocate(4096, 0);
  ASSERT_TRUE(first.ok());
  ASSERT_EQ(musaMemsetAsync(first.get(), 0x35, 4096, nullptr), musaSuccess);
  CudaAllocator::deallocate_async(first.get(), 0, nullptr);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  auto second = CudaAllocator::instance().allocate(4096, 0);
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(second.get(), first.get());
  std::vector<unsigned char> values(4096);
  ASSERT_EQ(
      musaMemcpy(
          values.data(), second.get(), values.size(), musaMemcpyDeviceToHost),
      musaSuccess);
  EXPECT_EQ(values, std::vector<unsigned char>(4096, 0x35));
  EXPECT_EQ(CudaAllocator::musa_allocation_cache_stats().reuse_count, 1);
  CudaAllocator::deallocate_async(second.get(), 0, nullptr);
}

TEST_F(MusaAllocatorTest, PendingBlockCannotCrossStreams) {
  musaStream_t producer = nullptr;
  musaStream_t consumer = nullptr;
  ASSERT_EQ(
      musaStreamCreateWithFlags(&producer, musaStreamNonBlocking), musaSuccess);
  ASSERT_EQ(
      musaStreamCreateWithFlags(&consumer, musaStreamNonBlocking), musaSuccess);
  auto pending = CudaAllocator::instance().allocate(4096, 0);
  auto ready = CudaAllocator::instance().allocate(4096, 0);
  ASSERT_TRUE(pending.ok());
  ASSERT_TRUE(ready.ok());
  std::atomic<bool> released{false};
  const auto wait = [](void* data) {
    auto* flag = static_cast<std::atomic<bool>*>(data);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!flag->load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
  };
  ASSERT_EQ(musaLaunchHostFunc(producer, wait, &released), musaSuccess);
  CudaAllocator::deallocate_async(pending.get(), 0, producer);
  CudaAllocator::deallocate_async(ready.get(), 0, consumer);
  EXPECT_EQ(musaStreamSynchronize(consumer), musaSuccess);
  auto selected = CudaAllocator::instance().allocate(4096, 0);
  released.store(true);
  ASSERT_TRUE(selected.ok());
  EXPECT_EQ(selected.get(), ready.get());
  EXPECT_NE(selected.get(), pending.get());
  EXPECT_EQ(musaStreamSynchronize(producer), musaSuccess);
  CudaAllocator::instance().deallocate(selected.get(), 0);
  EXPECT_EQ(musaStreamDestroy(producer), musaSuccess);
  EXPECT_EQ(musaStreamDestroy(consumer), musaSuccess);
}

TEST_F(MusaAllocatorTest, RetentionLimitAndExplicitReconfiguration) {
  ASSERT_EQ(CudaAllocator::configure_musa_allocation_cache(4096), Error::Ok);
  auto first = CudaAllocator::instance().allocate(4096, 0);
  auto second = CudaAllocator::instance().allocate(4096, 0);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(
      CudaAllocator::configure_musa_allocation_cache(0),
      Error::InvalidArgument);
  CudaAllocator::deallocate_async(first.get(), 0, nullptr);
  CudaAllocator::deallocate_async(second.get(), 0, nullptr);
  EXPECT_EQ(CudaAllocator::musa_allocation_cache_stats().cached_bytes, 4096);
  EXPECT_EQ(CudaAllocator::musa_allocation_cache_stats().live_blocks, 0);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  CudaAllocator::release_cached_memory(-1);
  EXPECT_EQ(CudaAllocator::configure_musa_allocation_cache(0), Error::Ok);
}

TEST_F(MusaAllocatorTest, SynchronousFreeAndLargeAlignmentBypassRetention) {
  auto first = CudaAllocator::instance().allocate(4096, 0);
  ASSERT_TRUE(first.ok());
  CudaAllocator::instance().deallocate(first.get(), 0);
  EXPECT_EQ(CudaAllocator::musa_allocation_cache_stats().live_blocks, 0);
  auto aligned = CudaAllocator::instance().allocate(4096, 0, 1024);
  ASSERT_TRUE(aligned.ok());
  EXPECT_EQ(reinterpret_cast<uintptr_t>(aligned.get()) % 1024, 0);
  EXPECT_EQ(CudaAllocator::musa_allocation_cache_stats().live_blocks, 0);
  CudaAllocator::instance().deallocate(aligned.get(), 0);
}

TEST_F(MusaAllocatorTest, AllocationFailureReclaimsReadyBlocksAndRetries) {
  auto cached = CudaAllocator::instance().allocate(4096, 0);
  ASSERT_TRUE(cached.ok());
  CudaAllocator::deallocate_async(cached.get(), 0, nullptr);
  ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  fail_allocations.store(1);
  auto allocation = CudaAllocator::instance().allocate(8192, 0);
  ASSERT_TRUE(allocation.ok());
  EXPECT_EQ(fail_allocations.load(), 0);
  EXPECT_EQ(CudaAllocator::musa_allocation_cache_stats().cached_bytes, 0);
  CudaAllocator::instance().deallocate(allocation.get(), 0);
}

TEST_F(MusaAllocatorTest, InvalidDeviceDoesNotChangeState) {
  auto allocation = CudaAllocator::instance().allocate(4096, devices_);
  EXPECT_FALSE(allocation.ok());
  EXPECT_EQ(CudaAllocator::musa_allocation_cache_stats().live_blocks, 0);
  int current = -1;
  ASSERT_EQ(musaGetDevice(&current), musaSuccess);
  EXPECT_EQ(current, 0);
}

void concurrent_streams(bool enable_cache) {
  ASSERT_EQ(
      CudaAllocator::configure_musa_allocation_cache(
          enable_cache ? 1 << 20 : 0),
      Error::Ok);
  std::vector<std::thread> threads;
  for (int worker = 0; worker < 4; ++worker) {
    threads.emplace_back([worker] {
      ASSERT_EQ(musaSetDevice(0), musaSuccess);
      musaStream_t stream = nullptr;
      ASSERT_EQ(
          musaStreamCreateWithFlags(&stream, musaStreamNonBlocking),
          musaSuccess);
      void* host = nullptr;
      ASSERT_EQ(musaHostAlloc(&host, 4096, musaHostAllocDefault), musaSuccess);
      auto* values = static_cast<unsigned char*>(host);
      for (int iteration = 0; iteration < 100; ++iteration) {
        auto allocation = CudaAllocator::instance().allocate(4096, 0);
        ASSERT_TRUE(allocation.ok());
        ASSERT_EQ(
            musaMemsetAsync(allocation.get(), worker + 1, 4096, stream),
            musaSuccess);
        ASSERT_EQ(musaStreamSynchronize(stream), musaSuccess);
        std::fill(values, values + 4096, 0);
        ASSERT_EQ(
            musaMemcpyAsync(
                values, allocation.get(), 4096, musaMemcpyDeviceToHost, stream),
            musaSuccess);
        ASSERT_EQ(musaStreamSynchronize(stream), musaSuccess);
        const auto mismatch =
            std::find_if(values, values + 4096, [worker](auto value) {
              return value != worker + 1;
            });
        EXPECT_EQ(mismatch, values + 4096)
            << "worker=" << worker << " iteration=" << iteration
            << " first_mismatch=" << std::distance(values, mismatch)
            << " value=" << (mismatch == values + 4096 ? -1 : *mismatch);
        CudaAllocator::deallocate_async(allocation.get(), 0, stream);
      }
      ASSERT_EQ(musaStreamSynchronize(stream), musaSuccess);
      ASSERT_EQ(musaStreamDestroy(stream), musaSuccess);
      ASSERT_EQ(musaFreeHost(host), musaSuccess);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  const auto stats = CudaAllocator::musa_allocation_cache_stats();
  if (enable_cache) {
    EXPECT_GT(stats.reuse_count, 0);
  } else {
    EXPECT_EQ(stats.reuse_count, 0);
  }
  EXPECT_LE(stats.cached_bytes, stats.limit_bytes);
}

TEST_F(MusaAllocatorTest, ConcurrentStreamsRemainBounded) {
  concurrent_streams(true);
}

TEST_F(MusaAllocatorTest, UncachedConcurrentStreamsControl) {
  concurrent_streams(false);
}

TEST_F(MusaAllocatorTest, DeviceScopedReuseAndTrimRestoreCallerDevice) {
  if (devices_ < 2) {
    GTEST_SKIP() << "Requires two physical MUSA devices";
  }
  for (int device = 0; device < 2; ++device) {
    ASSERT_EQ(musaSetDevice(device), musaSuccess);
    auto allocation = CudaAllocator::instance().allocate(4096, device);
    ASSERT_TRUE(allocation.ok());
    CudaAllocator::deallocate_async(allocation.get(), device, nullptr);
    ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
  }
  ASSERT_EQ(musaSetDevice(1), musaSuccess);
  auto allocation = CudaAllocator::instance().allocate(4096, 0);
  ASSERT_TRUE(allocation.ok());
  int current = -1;
  ASSERT_EQ(musaGetDevice(&current), musaSuccess);
  EXPECT_EQ(current, 1);
  CudaAllocator::instance().deallocate(allocation.get(), 0);
  CudaAllocator::release_cached_memory(-1);
  ASSERT_EQ(musaGetDevice(&current), musaSuccess);
  EXPECT_EQ(current, 1);
}
