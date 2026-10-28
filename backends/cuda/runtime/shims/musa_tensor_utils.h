#pragma once

#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/assert.h>

#include <cstdint>
#include <limits>

namespace executorch::backends::cuda {

struct MusaTensorSpan {
  uintptr_t begin;
  size_t bytes;
};

inline executorch::runtime::Result<MusaTensorSpan> checked_musa_tensor_span(
    const executorch::backends::aoti::slim::SlimTensor& tensor) {
  ET_CHECK_OR_RETURN_ERROR(
      tensor.storage() != nullptr,
      InvalidArgument,
      "MUSA tensor requires storage");
  const size_t element_size = tensor.itemsize();
  const size_t capacity = tensor.storage()->nbytes() / element_size;
  ET_CHECK_OR_RETURN_ERROR(
      tensor.storage_offset() >= 0 &&
          static_cast<uint64_t>(tensor.storage_offset()) <= capacity,
      InvalidArgument,
      "MUSA tensor offset exceeds storage");
  bool empty = false;
  for (int64_t dimension = 0; dimension < tensor.dim(); ++dimension) {
    ET_CHECK_OR_RETURN_ERROR(
        tensor.size(dimension) >= 0 && tensor.stride(dimension) >= 0,
        InvalidArgument,
        "MUSA tensor requires nonnegative sizes and strides");
    empty |= tensor.size(dimension) == 0;
  }
  if (empty) {
    return MusaTensorSpan{0, 0};
  }
  const size_t offset = static_cast<size_t>(tensor.storage_offset());
  ET_CHECK_OR_RETURN_ERROR(
      offset < capacity && tensor.storage()->data() != nullptr,
      InvalidArgument,
      "Nonempty MUSA tensor requires backed storage");
  size_t last = offset;
  for (int64_t dimension = 0; dimension < tensor.dim(); ++dimension) {
    const auto stride = static_cast<uint64_t>(tensor.stride(dimension));
    const auto count = static_cast<uint64_t>(tensor.size(dimension) - 1);
    ET_CHECK_OR_RETURN_ERROR(
        stride == 0 || count <= (capacity - 1 - last) / stride,
        InvalidArgument,
        "MUSA tensor strides exceed storage");
    last += count * stride;
  }
  const auto base = reinterpret_cast<uintptr_t>(tensor.storage()->data());
  ET_CHECK_OR_RETURN_ERROR(
      base % element_size == 0 &&
          (last + 1) <=
              (std::numeric_limits<uintptr_t>::max() - base) / element_size,
      InvalidArgument,
      "Invalid MUSA tensor address range or alignment");
  return MusaTensorSpan{
      base + offset * element_size, (last - offset + 1) * element_size};
}

inline bool musa_tensor_spans_overlap(
    const MusaTensorSpan& first,
    const MusaTensorSpan& second) {
  if (first.bytes == 0 || second.bytes == 0) {
    return false;
  }
  return first.begin <= second.begin
      ? second.begin - first.begin < first.bytes
      : first.begin - second.begin < second.bytes;
}

}
