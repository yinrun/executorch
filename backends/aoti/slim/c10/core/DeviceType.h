/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <ostream>
#include <string>

#include <executorch/runtime/platform/assert.h>

namespace executorch::backends::aoti::slim::c10 {

/// Enum representing the type of device.
enum class DeviceType : int8_t {
  CPU = 0,
  CUDA = 1,
  MUSA = 2,
  COMPILE_TIME_MAX_DEVICE_TYPES = 3,
};

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kMUSA = DeviceType::MUSA;
#if defined(EXECUTORCH_USE_MUSA)
constexpr DeviceType kGPU = kMUSA;
#else
constexpr DeviceType kGPU = kCUDA;
#endif

inline DeviceType canonical_device_type(DeviceType device_type) {
#if defined(EXECUTORCH_USE_MUSA)
  if (device_type == DeviceType::CUDA) {
    return DeviceType::MUSA;
  }
#endif
  return device_type;
}

inline DeviceType device_type_from_aoti(int32_t device_type) {
#if defined(EXECUTORCH_USE_MUSA)
  if (device_type == 20) {
    return DeviceType::MUSA;
  }
#endif
  ET_CHECK_MSG(
      device_type == 0 || device_type == 1,
      "Unsupported AOTI device type: %d",
      device_type);
  return canonical_device_type(static_cast<DeviceType>(device_type));
}

inline int32_t device_type_to_aoti(DeviceType device_type) {
  if (canonical_device_type(device_type) == DeviceType::MUSA) {
    return 20;
  }
  return static_cast<int32_t>(device_type);
}

/// Maximum number of device types at compile time.
constexpr int COMPILE_TIME_MAX_DEVICE_TYPES =
    static_cast<int>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);

/// Returns the name of the device type as a string.
/// @param d The device type.
/// @param lower_case If true, returns the name in lower case.
/// @return The name of the device type.
inline std::string DeviceTypeName(DeviceType d, bool lower_case = false) {
  switch (d) {
    case DeviceType::CPU:
      return lower_case ? "cpu" : "CPU";
    case DeviceType::CUDA:
      return lower_case ? "cuda" : "CUDA";
    case DeviceType::MUSA:
      return lower_case ? "musa" : "MUSA";
    default:
      ET_CHECK_MSG(false, "Unknown device type: %d", static_cast<int>(d));
  }
}

/// Checks if the device type is valid.
/// @param d The device type to check.
/// @return true if the device type is valid, false otherwise.
inline bool isValidDeviceType(DeviceType d) {
  return d == DeviceType::CPU || d == DeviceType::CUDA || d == DeviceType::MUSA;
}

inline std::ostream& operator<<(std::ostream& stream, DeviceType type) {
  stream << DeviceTypeName(type, /* lower_case */ true);
  return stream;
}

} // namespace executorch::backends::aoti::slim::c10

namespace std {
template <>
struct hash<executorch::backends::aoti::slim::c10::DeviceType> {
  std::size_t operator()(
      executorch::backends::aoti::slim::c10::DeviceType k) const {
    return std::hash<int>()(static_cast<int>(k));
  }
};
} // namespace std
