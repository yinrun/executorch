#pragma once

#include <fstream>
#include <iomanip>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <executorch/extension/cuda/device_guard.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/platform/platform.h>

namespace executorch::examples::musa {

class Benchmark final {
 public:
  Benchmark(
      const runtime::MethodMeta& metadata,
      std::string output,
      uint32_t warmup,
      uint32_t iterations)
      : output_(std::move(output)), warmup_(warmup) {
    ET_CHECK_MSG(iterations > 0, "Benchmark requires measured executions");
    ET_CHECK_MSG(
        metadata.uses_backend("MusaBackend"), "Benchmark requires MusaBackend");
    std::set<int> devices;
    for (size_t index = 0; index < metadata.num_memory_planned_buffers();
         ++index) {
      auto device = metadata.memory_planned_buffer_device(index);
      ET_CHECK(device.ok());
      if (!device->is_cpu()) {
        ET_CHECK(
            device->type() == runtime::etensor::DeviceType::MUSA ||
            device->type() == runtime::etensor::DeviceType::CUDA);
        devices.insert(device->index());
      }
    }
    ET_CHECK_MSG(
        devices.size() == 1, "Benchmark requires exactly one MUSA device");
    device_ = *devices.begin();
    execute_ms_.reserve(iterations);
    prepare_execute_ms_.reserve(iterations);
  }

  void synchronize() const {
    auto guard = extension::cuda::CUDAGuard::create(device_);
    ET_CHECK(guard.ok());
    ET_CHECK_MSG(
        cudaDeviceSynchronize() == cudaSuccess, "MUSA synchronization failed");
  }

  void begin() {
    synchronize();
    before_prepare_ = runtime::pal_current_ticks();
  }

  void record(
      uint32_t iteration,
      et_timestamp_t before_execute,
      et_timestamp_t after_execute) {
    if (iteration < warmup_) {
      return;
    }
    const auto ratio = runtime::pal_ticks_to_ns_multiplier();
    const double milliseconds =
        static_cast<double>(ratio.numerator) / ratio.denominator / 1000000.0;
    execute_ms_.push_back((after_execute - before_execute) * milliseconds);
    prepare_execute_ms_.push_back(
        (after_execute - before_prepare_) * milliseconds);
  }

  void write() const {
    std::ofstream output(output_);
    ET_CHECK_MSG(output.good(), "Cannot open MUSA benchmark report");
    output << std::setprecision(17)
           << "{\"version\":1,\"synchronization\":\"musaDeviceSynchronize\","
           << "\"scope\":\"host_synchronized_method\",\"device_index\":"
           << device_ << ",\"warmup_executions\":" << warmup_
           << ",\"num_executions\":" << execute_ms_.size();
    for (const auto& samples :
         {std::pair<const char*, const std::vector<double>&>(
              "execute_ms", execute_ms_),
          std::pair<const char*, const std::vector<double>&>(
              "prepare_execute_ms", prepare_execute_ms_)}) {
      output << ",\"" << samples.first << "\":[";
      for (size_t index = 0; index < samples.second.size(); ++index) {
        if (index != 0) {
          output << ',';
        }
        output << samples.second[index];
      }
      output << ']';
    }
    output << "}\n";
    output.close();
    ET_CHECK_MSG(!output.fail(), "Cannot write MUSA benchmark report");
  }

 private:
  std::string output_;
  uint32_t warmup_;
  int device_;
  et_timestamp_t before_prepare_{};
  std::vector<double> execute_ms_;
  std::vector<double> prepare_execute_ms_;
};

}
