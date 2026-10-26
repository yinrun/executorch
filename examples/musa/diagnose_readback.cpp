#include <musa_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(musaError_t status) {
  if (status != musaSuccess) {
    throw std::runtime_error(musaGetErrorString(status));
  }
}

struct Buffers {
  musaStream_t stream{nullptr};
  void* device{nullptr};
  void* pinned{nullptr};

  ~Buffers() {
    if (stream != nullptr) {
      musaStreamSynchronize(stream);
    }
    if (pinned != nullptr) {
      musaFreeHost(pinned);
    }
    if (device != nullptr) {
      musaFree(device);
    }
    if (stream != nullptr) {
      musaStreamDestroy(stream);
    }
  }
};

}

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      throw std::runtime_error(
          "Usage: diagnose_readback pageable-async|pageable-sync|pinned-async|pageable-staged repetitions");
    }
    const std::string mode = argv[1];
    if (mode != "pageable-async" && mode != "pageable-sync" &&
        mode != "pinned-async" && mode != "pageable-staged") {
      throw std::runtime_error("Unknown transfer mode");
    }
    size_t consumed = 0;
    const auto repetitions = std::stoul(argv[2], &consumed);
    if (consumed != std::strlen(argv[2]) || repetitions == 0 ||
        repetitions > 1000000) {
      throw std::runtime_error("Repetitions must be between 1 and 1000000");
    }
    size_t failures = 0;
    for (size_t elements : {1000, 1001, 1999}) {
      const auto bytes = elements * sizeof(float);
      Buffers buffers;
      check(musaStreamCreateWithFlags(&buffers.stream, musaStreamNonBlocking));
      check(musaMalloc(&buffers.device, bytes));
      if (mode == "pinned-async" || mode == "pageable-staged") {
        check(musaMallocHost(&buffers.pinned, bytes));
      }
      const std::vector<float> expected(elements, 2.5f);
      check(musaMemcpy(
          buffers.device, expected.data(), bytes, musaMemcpyHostToDevice));
      check(musaDeviceSynchronize());
      const auto failures_before = failures;
      for (size_t repeat = 0; repeat < repetitions; ++repeat) {
        std::vector<uint8_t> pageable(mode == "pinned-async" ? 0 : bytes);
        void* destination =
            mode == "pinned-async" ? buffers.pinned : pageable.data();
        std::memset(destination, 0, bytes);
        if (mode == "pageable-sync") {
          check(musaMemcpy(
              destination, buffers.device, bytes, musaMemcpyDeviceToHost));
        } else if (mode == "pageable-staged") {
          check(musaMemcpyAsync(
              buffers.pinned,
              buffers.device,
              bytes,
              musaMemcpyDeviceToHost,
              buffers.stream));
          check(musaStreamSynchronize(buffers.stream));
          std::memcpy(destination, buffers.pinned, bytes);
        } else {
          check(musaMemcpyAsync(
              destination,
              buffers.device,
              bytes,
              musaMemcpyDeviceToHost,
              buffers.stream));
          check(musaStreamSynchronize(buffers.stream));
        }
        std::vector<float> decoded(elements);
        std::memcpy(decoded.data(), destination, bytes);
        auto mismatch =
            std::find_if(decoded.begin(), decoded.end(), [](float value) {
              return value != 2.5f;
            });
        if (mismatch != decoded.end()) {
          ++failures;
          if (failures <= 10) {
            const auto index = static_cast<size_t>(mismatch - decoded.begin());
            std::vector<float> repeated(elements);
            check(musaMemcpy(
                repeated.data(),
                buffers.device,
                bytes,
                musaMemcpyDeviceToHost));
            std::cerr << "mode=" << mode << " elements=" << elements
                      << " repeat=" << repeat << " index=" << index
                      << " first=" << *mismatch << " second=" << repeated[index]
                      << '\n';
          }
        }
      }
      std::cout << "{\"mode\":\"" << mode << "\",\"elements\":" << elements
                << ",\"repetitions\":" << repetitions
                << ",\"failed_readbacks\":" << failures - failures_before
                << "}\n";
    }
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
