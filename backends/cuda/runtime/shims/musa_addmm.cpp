/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <mublas_v2.h>

#include <cstdint>
#include <limits>

#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/c10/cuda/Exception.h>
#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/core/portable_type/half.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {

using executorch::backends::aoti::slim::SlimTensor;
using executorch::backends::aoti::slim::c10::ScalarType;
using executorch::runtime::Error;
using AOTITorchError = Error;

namespace {

AOTITorchError check_mublas(mublasStatus status, const char* operation) {
  if (status == MUBLAS_STATUS_SUCCESS) {
    return Error::Ok;
  }
  ET_LOG(
      Error,
      "MUSA addmm: %s failed with mublas status %d",
      operation,
      static_cast<int>(status));
  return Error::Internal;
}

bool fits_mublas_dimension(int64_t value) {
  return value >= 0 &&
      value <= static_cast<int64_t>(std::numeric_limits<int>::max());
}

} // namespace

extern "C" {

AOTI_SHIM_EXPORT AOTITorchError aoti_torch_musa_addmm_out(
    SlimTensor* out,
    SlimTensor* self,
    SlimTensor* mat1,
    SlimTensor* mat2,
    double beta,
    double alpha) {
  ET_CHECK_OR_RETURN_ERROR(
      out != nullptr && self != nullptr && mat1 != nullptr && mat2 != nullptr,
      InvalidArgument,
      "MUSA addmm received a null tensor handle");

  const auto dtype = out->dtype();
  ET_CHECK_OR_RETURN_ERROR(
      (dtype == ScalarType::Float || dtype == ScalarType::Half ||
       dtype == ScalarType::BFloat16) &&
          self->dtype() == dtype && mat1->dtype() == dtype &&
          mat2->dtype() == dtype,
      NotSupported,
      "MUSA addmm supports float32, float16, and bfloat16 tensors only");

  ET_CHECK_OR_RETURN_ERROR(
      out->is_cuda() && self->is_cuda() && mat1->is_cuda() && mat2->is_cuda(),
      InvalidArgument,
      "MUSA addmm expects device tensors");

  ET_CHECK_OR_RETURN_ERROR(
      out->dim() == 2 && (self->dim() == 1 || self->dim() == 2) &&
          mat1->dim() == 2 && mat2->dim() == 2,
      NotSupported,
      "MUSA addmm supports a one-dimensional or two-dimensional bias only");

  const int64_t m = mat1->size(0);
  const int64_t k = mat1->size(1);
  const int64_t n = mat2->size(1);
  ET_CHECK_OR_RETURN_ERROR(
      mat2->size(0) == k &&
          ((self->dim() == 1 && self->size(0) == n) ||
           (self->dim() == 2 && self->size(0) == m && self->size(1) == n)) &&
          out->size(0) == m && out->size(1) == n,
      InvalidArgument,
      "MUSA addmm tensor dimensions are incompatible");

  ET_CHECK_OR_RETURN_ERROR(
      out->is_contiguous() && self->is_contiguous() && mat1->is_contiguous(),
      NotSupported,
      "MUSA addmm currently supports contiguous output, bias, and mat1");

  const auto mat2_strides = mat2->strides();
  const bool mat2_row_major =
      mat2->is_contiguous() && mat2_strides[0] == n && mat2_strides[1] == 1;
  const bool mat2_column_major =
      mat2_strides[0] == 1 && mat2_strides[1] == k;
  ET_CHECK_OR_RETURN_ERROR(
      mat2_row_major || mat2_column_major,
      NotSupported,
      "MUSA addmm supports contiguous or transposed mat2 only");

  ET_CHECK_OR_RETURN_ERROR(
      out->device_index() == self->device_index() &&
          out->device_index() == mat1->device_index() &&
          out->device_index() == mat2->device_index(),
      InvalidArgument,
      "MUSA addmm tensors must use the same device");

  ET_CHECK_OR_RETURN_ERROR(
      fits_mublas_dimension(m) && fits_mublas_dimension(k) &&
          fits_mublas_dimension(n),
      NotSupported,
      "MUSA addmm dimensions exceed mublas limits");

  auto stream_result = getCurrentCUDAStream(out->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(stream_result.error());
  cudaStream_t stream = stream_result.get();

  mublasHandle_t handle = nullptr;
  AOTITorchError error = check_mublas(mublasCreate(&handle), "mublasCreate");
  if (error != Error::Ok) {
    return error;
  }

  error = check_mublas(
      mublasSetStream(handle, reinterpret_cast<MUstream>(stream)),
      "mublasSetStream");
  if (error != Error::Ok) {
    mublasDestroy(handle);
    return error;
  }

  error = check_mublas(
      mublasSetPointerMode(handle, MUBLAS_POINTER_MODE_HOST),
      "mublasSetPointerMode");
  if (error != Error::Ok) {
    mublasDestroy(handle);
    return error;
  }

  if (beta != 0.0) {
    if (self->dim() == 1) {
      const size_t row_bytes =
          static_cast<size_t>(n) * static_cast<size_t>(out->itemsize());
      for (int64_t row = 0; row < m; ++row) {
        ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpyAsync(
            static_cast<char*>(out->data_ptr()) + row * row_bytes,
            self->data_ptr(),
            row_bytes,
            cudaMemcpyDeviceToDevice,
            stream));
      }
    } else if (out->data_ptr() != self->data_ptr()) {
      ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpyAsync(
          out->data_ptr(),
          self->data_ptr(),
          out->nbytes(),
          cudaMemcpyDeviceToDevice,
          stream));
    }
  }

  const float alpha_float = static_cast<float>(alpha);
  const float beta_float = static_cast<float>(beta);
  const executorch::runtime::etensor::Half alpha_half(alpha_float);
  const executorch::runtime::etensor::Half beta_half(beta_float);
  const void* alpha_ptr = dtype == ScalarType::Half
      ? static_cast<const void*>(&alpha_half)
      : static_cast<const void*>(&alpha_float);
  const void* beta_ptr = dtype == ScalarType::Half
      ? static_cast<const void*>(&beta_half)
      : static_cast<const void*>(&beta_float);
  const mublasOperation_t mat2_operation =
      mat2_column_major ? MUBLAS_OP_T : MUBLAS_OP_N;
  const int mat2_leading_dimension =
      static_cast<int>(mat2_column_major ? k : n);
  const musaDataType_t data_type =
      dtype == ScalarType::Float
      ? MUSA_R_32F
      : (dtype == ScalarType::Half ? MUSA_R_16F : MUSA_R_16BF);
  const mublasComputeType_t compute_type =
      dtype == ScalarType::Half ? MUBLAS_COMPUTE_16F : MUBLAS_COMPUTE_32F;

  // mublas uses column-major matrices. Swapping the row-major operands and
  // dimensions computes the same row-major mat1 @ mat2 result in-place. A
  // transposed row-major weight is already column-major for the logical mat2.
  error = check_mublas(
      mublasGemmEx(
          handle,
          mat2_operation,
          MUBLAS_OP_N,
          static_cast<int>(n),
          static_cast<int>(m),
          static_cast<int>(k),
          alpha_ptr,
          mat2->data_ptr(),
          data_type,
          mat2_leading_dimension,
          mat1->data_ptr(),
          data_type,
          static_cast<int>(k),
          beta_ptr,
          out->data_ptr(),
          data_type,
          static_cast<int>(n),
          compute_type,
          MUBLAS_GEMM_DEFAULT),
      "mublasGemmEx");

  const AOTITorchError destroy_error =
      check_mublas(mublasDestroy(handle), "mublasDestroy");
  return error == Error::Ok ? destroy_error : error;
}

} // extern "C"
} // namespace executorch::backends::cuda
