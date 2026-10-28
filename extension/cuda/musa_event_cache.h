#pragma once

#include <executorch/extension/cuda/device_guard.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace executorch::extension::cuda {
namespace {

class MusaEventCache final {
 public:
  bool enabled() const { return limit_.load(std::memory_order_relaxed) != 0; }

  runtime::Error configure(size_t limit) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!live_.empty() || !pending_.empty()) {
      return runtime::Error::InvalidArgument;
    }
    limit_.store(limit, std::memory_order_relaxed);
    hits_ = 0;
    allocations_ = 0;
    return runtime::Error::Ok;
  }

  runtime::Result<void*> reuse(size_t bytes, int device) {
    const std::lock_guard<std::mutex> lock(mutex_);
    for (auto entry = pending_.begin(); entry != pending_.end(); ++entry) {
      if (entry->bytes != bytes || entry->device != device) {
        continue;
      }
      const auto status = musaEventQuery(entry->ready);
      if (status == musaErrorNotReady) {
        continue;
      }
      if (status != musaSuccess ||
          musaEventDestroy(entry->ready) != musaSuccess) {
        return runtime::Error::Internal;
      }
      auto block = *entry;
      block.ready = nullptr;
      live_.emplace(block.pointer, block);
      cached_bytes_ -= block.bytes;
      pending_.erase(entry);
      ++hits_;
      return block.pointer;
    }
    return static_cast<void*>(nullptr);
  }

  void remember(void* pointer, size_t bytes, int device) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (bytes == 0 ||
        bytes > std::min(limit_.load(), size_t{16 * 1024 * 1024})) {
      return;
    }
    live_.emplace(pointer, Block{pointer, bytes, device, nullptr});
    ++allocations_;
  }

  bool retire(void* pointer, musaStream_t stream) {
    if (!enabled()) {
      return false;
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    auto entry = live_.find(pointer);
    if (entry == live_.end() ||
        entry->second.bytes > limit_.load() - cached_bytes_) {
      return false;
    }
    int current = -1;
    musaStreamCaptureStatus capture = musaStreamCaptureStatusNone;
    if (musaGetDevice(&current) != musaSuccess ||
        current != entry->second.device ||
        musaStreamIsCapturing(stream, &capture) != musaSuccess ||
        capture != musaStreamCaptureStatusNone) {
      return false;
    }
    auto block = entry->second;
    if (musaEventCreateWithFlags(&block.ready, musaEventDisableTiming) !=
        musaSuccess) {
      return false;
    }
    if (musaEventRecord(block.ready, stream) != musaSuccess) {
      musaEventDestroy(block.ready);
      return false;
    }
    live_.erase(entry);
    pending_.push_back(block);
    cached_bytes_ += block.bytes;
    return true;
  }

  void forget(void* pointer) {
    if (enabled()) {
      const std::lock_guard<std::mutex> lock(mutex_);
      live_.erase(pointer);
    }
  }

  size_t trim(int device) {
    const std::lock_guard<std::mutex> lock(mutex_);
    size_t released = 0;
    for (auto entry = pending_.begin(); entry != pending_.end();) {
      if (device >= 0 && entry->device != device) {
        ++entry;
        continue;
      }
      auto guard = CUDAGuard::create(entry->device);
      if (!guard.ok() || musaEventQuery(entry->ready) != musaSuccess) {
        ++entry;
        continue;
      }
      if (musaFree(entry->pointer) != musaSuccess) {
        ++entry;
        continue;
      }
      if (musaEventDestroy(entry->ready) != musaSuccess) {
        ET_LOG(Error, "MUSA cache event destruction failed");
      }
      released += entry->bytes;
      cached_bytes_ -= entry->bytes;
      entry = pending_.erase(entry);
    }
    return released;
  }

  MusaAllocationCacheStats stats() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return {
        limit_.load(),
        cached_bytes_,
        pending_.size(),
        live_.size(),
        hits_,
        allocations_};
  }

 private:
  struct Block {
    void* pointer;
    size_t bytes;
    int device;
    musaEvent_t ready;
  };
  std::mutex mutex_;
  std::atomic<size_t> limit_{0};
  std::unordered_map<void*, Block> live_;
  std::vector<Block> pending_;
  size_t cached_bytes_ = 0;
  size_t hits_ = 0;
  size_t allocations_ = 0;
};

MusaEventCache& musa_event_cache() {
  static auto* cache = new MusaEventCache();
  return *cache;
}

}
}
