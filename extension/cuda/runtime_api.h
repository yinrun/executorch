/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

#if defined(EXECUTORCH_USE_MUSA)

#include <musa_runtime.h>
#include <cstring>

using cudaError_t = musaError_t;
using cudaEvent_t = musaEvent_t;
using cudaGraph_t = musaGraph_t;
using cudaGraphExec_t = musaGraphExec_t;
using cudaMemcpyKind = musaMemcpyKind;
using cudaMemoryType = musaMemoryType;
using cudaStreamCaptureMode = musaStreamCaptureMode;
using cudaStream_t = musaStream_t;

inline constexpr cudaError_t cudaSuccess = musaSuccess;
inline constexpr unsigned int cudaEventDisableTiming = musaEventDisableTiming;
inline constexpr cudaMemcpyKind cudaMemcpyDeviceToDevice =
    musaMemcpyDeviceToDevice;
inline constexpr cudaMemcpyKind cudaMemcpyDeviceToHost = musaMemcpyDeviceToHost;
inline constexpr cudaMemcpyKind cudaMemcpyHostToDevice = musaMemcpyHostToDevice;
inline constexpr cudaMemoryType cudaMemoryTypeDevice = musaMemoryTypeDevice;
inline constexpr cudaMemoryType cudaMemoryTypeManaged = musaMemoryTypeManaged;
inline constexpr cudaStreamCaptureMode cudaStreamCaptureModeRelaxed =
    musaStreamCaptureModeRelaxed;
inline constexpr unsigned long long cudaGraphInstantiateFlagAutoFreeOnLaunch =
    musaGraphInstantiateFlagAutoFreeOnLaunch;

#ifndef cudaStreamPerThread
#define cudaStreamPerThread musaStreamPerThread
#endif

struct cudaPointerAttributes {
  cudaMemoryType type{};
  int device = -1;
};

inline cudaError_t cudaDeviceSynchronize() {
  return musaDeviceSynchronize();
}

inline cudaError_t cudaEventCreateWithFlags(
    cudaEvent_t* event,
    unsigned int flags) {
  return musaEventCreateWithFlags(event, flags);
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
  return musaEventDestroy(event);
}

inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  return musaEventRecord(event, stream);
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return musaEventSynchronize(event);
}

inline cudaError_t cudaFree(void* ptr) {
  return musaFree(ptr);
}

inline cudaError_t cudaFreeAsync(void* ptr, cudaStream_t stream) {
  return musaFreeAsync(ptr, stream);
}

inline cudaError_t cudaFreeHost(void* ptr) {
  return musaFreeHost(ptr);
}

inline cudaError_t cudaGetDevice(int* device) {
  return musaGetDevice(device);
}

inline cudaError_t cudaGetDeviceCount(int* count) {
  return musaGetDeviceCount(count);
}

inline const char* cudaGetErrorString(cudaError_t error) {
  return musaGetErrorString(error);
}

inline cudaError_t cudaGetLastError() {
  return musaGetLastError();
}

inline cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  return musaGraphDestroy(graph);
}

inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graph_exec) {
  return musaGraphExecDestroy(graph_exec);
}

inline cudaError_t cudaGraphInstantiate(
    cudaGraphExec_t* graph_exec,
    cudaGraph_t graph,
    unsigned long long flags) {
  return musaGraphInstantiate(graph_exec, graph, flags);
}

inline cudaError_t cudaGraphLaunch(
    cudaGraphExec_t graph_exec,
    cudaStream_t stream) {
  return musaGraphLaunch(graph_exec, stream);
}

inline cudaError_t cudaMalloc(void** ptr, size_t size) {
  return musaMalloc(ptr, size);
}

inline cudaError_t
cudaMallocAsync(void** ptr, size_t size, cudaStream_t stream) {
  return musaMallocAsync(ptr, size, stream);
}

inline cudaError_t cudaMallocHost(void** ptr, size_t size) {
  return musaMallocHost(ptr, size, musaMemAttachGlobal);
}

namespace executorch::extension::cuda::detail {

inline musaError_t copy_musa_to_host(
    void* destination,
    const void* source,
    size_t bytes,
    musaStream_t stream,
    bool asynchronous) {
  if (bytes == 0 || destination == nullptr || source == nullptr) {
    return asynchronous
        ? musaMemcpyAsync(
              destination, source, bytes, musaMemcpyDeviceToHost, stream)
        : musaMemcpy(destination, source, bytes, musaMemcpyDeviceToHost);
  }
  musaPointerAttributes attributes{};
  auto status = musaPointerGetAttributes(&attributes, destination);
  if (status == musaErrorInvalidValue) {
    musaGetLastError();
    attributes.type = musaMemoryTypeUnregistered;
  } else if (status != musaSuccess) {
    return status;
  }
  if (attributes.type != musaMemoryTypeUnregistered) {
    return asynchronous
        ? musaMemcpyAsync(
              destination, source, bytes, musaMemcpyDeviceToHost, stream)
        : musaMemcpy(destination, source, bytes, musaMemcpyDeviceToHost);
  }
  void* staging = nullptr;
  status = musaMallocHost(&staging, bytes);
  if (status != musaSuccess) {
    return status;
  }
  status =
      musaMemcpyAsync(staging, source, bytes, musaMemcpyDeviceToHost, stream);
  if (status == musaSuccess) {
    status = musaStreamSynchronize(stream);
  }
  if (status == musaSuccess) {
    std::memcpy(destination, staging, bytes);
  }
  const auto cleanup = musaFreeHost(staging);
  return status == musaSuccess ? cleanup : status;
}

}

inline cudaError_t
cudaMemcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
  if (kind == cudaMemcpyDeviceToHost) {
    return executorch::extension::cuda::detail::copy_musa_to_host(
        dst, src, size, nullptr, false);
  }
  return musaMemcpy(dst, src, size, kind);
}

inline cudaError_t cudaMemcpyAsync(
    void* dst,
    const void* src,
    size_t size,
    cudaMemcpyKind kind,
    cudaStream_t stream) {
  if (kind == cudaMemcpyDeviceToHost) {
    return executorch::extension::cuda::detail::copy_musa_to_host(
        dst, src, size, stream, true);
  }
  return musaMemcpyAsync(dst, src, size, kind, stream);
}

inline cudaError_t cudaMemcpy2DAsync(
    void* dst,
    size_t dpitch,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height,
    cudaMemcpyKind kind,
    cudaStream_t stream) {
  return musaMemcpy2DAsync(
      dst, dpitch, src, spitch, width, height, kind, stream);
}

inline cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
  return musaMemGetInfo(free, total);
}

inline cudaError_t
cudaMemsetAsync(void* ptr, int value, size_t size, cudaStream_t stream) {
  return musaMemsetAsync(ptr, value, size, stream);
}

inline cudaError_t cudaPointerGetAttributes(
    cudaPointerAttributes* attributes,
    const void* ptr) {
  musaPointerAttributes musa_attributes{};
  const auto error = musaPointerGetAttributes(&musa_attributes, ptr);
  if (error == musaSuccess) {
    attributes->type = musa_attributes.type;
    attributes->device = musa_attributes.device;
  }
  return error;
}

inline cudaError_t cudaSetDevice(int device) {
  return musaSetDevice(device);
}

inline cudaError_t cudaStreamBeginCapture(
    cudaStream_t stream,
    cudaStreamCaptureMode mode) {
  return musaStreamBeginCapture(stream, mode);
}

inline cudaError_t cudaStreamCreate(cudaStream_t* stream) {
  return musaStreamCreate(stream);
}

inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  return musaStreamDestroy(stream);
}

inline cudaError_t cudaStreamEndCapture(
    cudaStream_t stream,
    cudaGraph_t* graph) {
  return musaStreamEndCapture(stream, graph);
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  return musaStreamSynchronize(stream);
}

inline cudaError_t cudaStreamWaitEvent(
    cudaStream_t stream,
    cudaEvent_t event,
    unsigned int flags) {
  return musaStreamWaitEvent(stream, event, flags);
}

#elif defined(EXECUTORCH_USE_HIP)

#include <hip/hip_runtime.h>

using cudaError_t = hipError_t;
using cudaEvent_t = hipEvent_t;
using cudaGraph_t = hipGraph_t;
using cudaGraphExec_t = hipGraphExec_t;
using cudaMemcpyKind = hipMemcpyKind;
using cudaMemoryType = hipMemoryType;
using cudaStreamCaptureMode = hipStreamCaptureMode;
using cudaStream_t = hipStream_t;

inline constexpr cudaError_t cudaSuccess = hipSuccess;
inline constexpr unsigned int cudaEventDisableTiming = hipEventDisableTiming;
inline constexpr cudaMemcpyKind cudaMemcpyDeviceToDevice =
    hipMemcpyDeviceToDevice;
inline constexpr cudaMemcpyKind cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
inline constexpr cudaMemcpyKind cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
inline constexpr cudaMemoryType cudaMemoryTypeDevice = hipMemoryTypeDevice;
inline constexpr cudaMemoryType cudaMemoryTypeManaged = hipMemoryTypeManaged;
inline constexpr cudaStreamCaptureMode cudaStreamCaptureModeRelaxed =
    hipStreamCaptureModeRelaxed;
inline constexpr unsigned long long cudaGraphInstantiateFlagAutoFreeOnLaunch =
    hipGraphInstantiateFlagAutoFreeOnLaunch;

// A macro, unlike the aliases above, because hipStreamPerThread casts an
// integer to a pointer type and so is not a constant expression. Guarded in
// case a HIP header ever defines the CUDA spelling itself.
#ifndef cudaStreamPerThread
#define cudaStreamPerThread hipStreamPerThread
#endif

struct cudaPointerAttributes {
  cudaMemoryType type{};
  int device = -1;
};

inline cudaError_t cudaDeviceSynchronize() {
  return hipDeviceSynchronize();
}

inline cudaError_t cudaEventCreateWithFlags(
    cudaEvent_t* event,
    unsigned int flags) {
  return hipEventCreateWithFlags(event, flags);
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
  return hipEventDestroy(event);
}

inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  return hipEventRecord(event, stream);
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return hipEventSynchronize(event);
}

inline cudaError_t cudaFree(void* ptr) {
  return hipFree(ptr);
}

inline cudaError_t cudaFreeAsync(void* ptr, cudaStream_t stream) {
  return hipFreeAsync(ptr, stream);
}

inline cudaError_t cudaFreeHost(void* ptr) {
  return hipHostFree(ptr);
}

inline cudaError_t cudaGetDevice(int* device) {
  return hipGetDevice(device);
}

inline cudaError_t cudaGetDeviceCount(int* count) {
  return hipGetDeviceCount(count);
}

inline const char* cudaGetErrorString(cudaError_t error) {
  return hipGetErrorString(error);
}

inline cudaError_t cudaGetLastError() {
  return hipGetLastError();
}

inline cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  return hipGraphDestroy(graph);
}

inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graph_exec) {
  return hipGraphExecDestroy(graph_exec);
}

inline cudaError_t cudaGraphInstantiate(
    cudaGraphExec_t* graph_exec,
    cudaGraph_t graph,
    unsigned long long flags) {
  return hipGraphInstantiateWithFlags(graph_exec, graph, flags);
}

inline cudaError_t cudaGraphLaunch(
    cudaGraphExec_t graph_exec,
    cudaStream_t stream) {
  return hipGraphLaunch(graph_exec, stream);
}

inline cudaError_t cudaMalloc(void** ptr, size_t size) {
  return hipMalloc(ptr, size);
}

inline cudaError_t
cudaMallocAsync(void** ptr, size_t size, cudaStream_t stream) {
  return hipMallocAsync(ptr, size, stream);
}

inline cudaError_t cudaMallocHost(void** ptr, size_t size) {
  return hipHostMalloc(ptr, size, 0);
}

inline cudaError_t
cudaMemcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
  return hipMemcpy(dst, src, size, kind);
}

inline cudaError_t cudaMemcpyAsync(
    void* dst,
    const void* src,
    size_t size,
    cudaMemcpyKind kind,
    cudaStream_t stream) {
  return hipMemcpyAsync(dst, src, size, kind, stream);
}

inline cudaError_t cudaMemcpy2DAsync(
    void* dst,
    size_t dpitch,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height,
    cudaMemcpyKind kind,
    cudaStream_t stream) {
  return hipMemcpy2DAsync(
      dst, dpitch, src, spitch, width, height, kind, stream);
}

inline cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
  return hipMemGetInfo(free, total);
}

inline cudaError_t
cudaMemsetAsync(void* ptr, int value, size_t size, cudaStream_t stream) {
  return hipMemsetAsync(ptr, value, size, stream);
}

inline cudaError_t cudaPointerGetAttributes(
    cudaPointerAttributes* attributes,
    const void* ptr) {
  hipPointerAttribute_t hip_attributes{};
  const auto error = hipPointerGetAttributes(&hip_attributes, ptr);
  if (error == hipSuccess) {
    attributes->type = hip_attributes.type;
    attributes->device = hip_attributes.device;
  }
  return error;
}

inline cudaError_t cudaSetDevice(int device) {
  return hipSetDevice(device);
}

inline cudaError_t cudaStreamBeginCapture(
    cudaStream_t stream,
    cudaStreamCaptureMode mode) {
  return hipStreamBeginCapture(stream, mode);
}

inline cudaError_t cudaStreamCreate(cudaStream_t* stream) {
  return hipStreamCreate(stream);
}

inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  return hipStreamDestroy(stream);
}

inline cudaError_t cudaStreamEndCapture(
    cudaStream_t stream,
    cudaGraph_t* graph) {
  return hipStreamEndCapture(stream, graph);
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  return hipStreamSynchronize(stream);
}

inline cudaError_t cudaStreamWaitEvent(
    cudaStream_t stream,
    cudaEvent_t event,
    unsigned int flags) {
  return hipStreamWaitEvent(stream, event, flags);
}

#else

#include <cuda_runtime.h>

#endif
