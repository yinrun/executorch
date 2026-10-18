/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <mublas_v2.h>
#include <mudnncxx/mudnn.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/c10/cuda/Exception.h>
#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
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

AOTITorchError check_mudnn(
    ::musa::dnn::Status status,
    const char* operation) {
  if (status == ::musa::dnn::Status::SUCCESS) {
    return Error::Ok;
  }
  ET_LOG(
      Error,
      "MUSA matrix operation: %s failed with mudnn status %d",
      operation,
      static_cast<int>(status));
  return Error::Internal;
}

::musa::dnn::Tensor::Type mudnn_type(ScalarType dtype) {
  if (dtype == ScalarType::Float) {
    return ::musa::dnn::Tensor::Type::FLOAT;
  }
  return dtype == ScalarType::Half
      ? ::musa::dnn::Tensor::Type::HALF
      : ::musa::dnn::Tensor::Type::BFLOAT16;
}

AOTITorchError set_mudnn_tensor(
    ::musa::dnn::Tensor& descriptor,
    const SlimTensor* tensor,
    bool transpose = false) {
  AOTITorchError error =
      check_mudnn(descriptor.SetAddr(tensor->data_ptr()), "Tensor.SetAddr");
  if (error != Error::Ok) {
    return error;
  }
  error = check_mudnn(
      descriptor.SetType(mudnn_type(tensor->dtype())), "Tensor.SetType");
  if (error != Error::Ok) {
    return error;
  }
  if (transpose) {
    if (tensor->dim() == 3) {
      const int64_t sizes[] = {
          tensor->size(0), tensor->size(2), tensor->size(1)};
      const int64_t strides[] = {
          tensor->stride(0), tensor->stride(2), tensor->stride(1)};
      return check_mudnn(
          descriptor.SetNdInfo(3, sizes, strides), "Tensor.SetNdInfo");
    }
    const int64_t sizes[] = {tensor->size(1), tensor->size(0)};
    const int64_t strides[] = {tensor->stride(1), tensor->stride(0)};
    return check_mudnn(
        descriptor.SetNdInfo(2, sizes, strides), "Tensor.SetNdInfo");
  }
  if (tensor->dim() == 2 && tensor->size(0) == 1 &&
      tensor->stride(0) == 0) {
    const int64_t sizes[] = {1, tensor->size(1)};
    const int64_t strides[] = {tensor->size(1), tensor->stride(1)};
    return check_mudnn(
        descriptor.SetNdInfo(2, sizes, strides), "Tensor.SetNdInfo");
  }
  return check_mudnn(
      descriptor.SetNdInfo(
          static_cast<int64_t>(tensor->dim()),
          tensor->sizes().data(),
          tensor->strides().data()),
      "Tensor.SetNdInfo");
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
      out->is_contiguous() && self->is_contiguous(),
      NotSupported,
      "MUSA addmm currently supports contiguous output and bias");

  const auto mat1_strides = mat1->strides();
  const bool mat1_broadcast_row =
      m == 1 && mat1_strides[0] == 0 && mat1_strides[1] == 1;
  const bool mat1_row_major = mat1->is_contiguous() || mat1_broadcast_row;
  const bool mat1_padded_rows =
      mat1_strides[1] == 1 && mat1_strides[0] >= k &&
      fits_mublas_dimension(mat1_strides[0]);
  const bool mat1_column_major =
      mat1_strides[0] == 1 && mat1_strides[1] == m;
  ET_CHECK_OR_RETURN_ERROR(
      mat1_row_major || mat1_padded_rows || mat1_column_major,
      NotSupported,
      "MUSA addmm supports contiguous, padded-row, or transposed mat1 only");

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

  std::optional<SlimTensor> packed_mat1;
  if (mat1_padded_rows && !mat1_row_major) {
    const int64_t packed_strides[] = {k, 1};
    packed_mat1.emplace(executorch::backends::aoti::slim::empty_strided(
        mat1->sizes(),
        executorch::backends::aoti::slim::IntArrayRef(packed_strides, 2),
        dtype,
        mat1->device()));
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpy2DAsync(
        packed_mat1->data_ptr(),
        static_cast<size_t>(k) * out->itemsize(),
        mat1->data_ptr(),
        static_cast<size_t>(mat1_strides[0]) * out->itemsize(),
        static_cast<size_t>(k) * out->itemsize(),
        static_cast<size_t>(m),
        cudaMemcpyDeviceToDevice,
        stream));
  }

  const bool use_mublas =
      ((dtype == ScalarType::BFloat16 && m <= 8) || mat1_padded_rows ||
       mat1_broadcast_row) &&
      (beta == 0.0 || beta == 1.0);
  if (use_mublas) {
    mublasHandle_t handle = nullptr;
    AOTITorchError error =
        check_mublas(mublasCreate(&handle), "mublasCreate");
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
    const musaDataType_t data_type =
        dtype == ScalarType::Float
        ? MUSA_R_32F
        : (dtype == ScalarType::Half ? MUSA_R_16F : MUSA_R_16BF);
    const auto compute_type =
        dtype == ScalarType::Half ? MUBLAS_COMPUTE_16F : MUBLAS_COMPUTE_32F;
    const auto mat2_operation =
        mat2_column_major ? MUBLAS_OP_T : MUBLAS_OP_N;
    const auto mat1_operation = mat1_row_major || packed_mat1
        ? MUBLAS_OP_N
        : MUBLAS_OP_T;
    const int mat2_leading_dimension =
        static_cast<int>(mat2_column_major ? k : n);
    const int mat1_leading_dimension =
        static_cast<int>(mat1_row_major || packed_mat1 ? k : m);
    const void* mat1_data = packed_mat1 ? packed_mat1->data_ptr()
                                        : mat1->data_ptr();
    if (self->dim() == 1 && beta != 0.0) {
      for (int64_t row = 0; row < m; ++row) {
        ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpyAsync(
            static_cast<char*>(out->data_ptr()) +
                row * n * out->itemsize(),
            self->data_ptr(),
            n * out->itemsize(),
            cudaMemcpyDeviceToDevice,
            stream));
      }
    } else if (self->dim() == 2 && beta != 0.0 &&
               out->data_ptr() != self->data_ptr()) {
      ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpyAsync(
          out->data_ptr(),
          self->data_ptr(),
          out->nbytes(),
          cudaMemcpyDeviceToDevice,
          stream));
    }

    error = check_mublas(
        mublasGemmEx(
            handle,
            mat2_operation,
            mat1_operation,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            alpha_ptr,
            mat2->data_ptr(),
            data_type,
            mat2_leading_dimension,
            mat1_data,
            data_type,
            mat1_leading_dimension,
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

  ::musa::dnn::Handle handle(out->device_index());
  AOTITorchError error =
      check_mudnn(handle.SetStream(stream), "Handle.SetStream");
  if (error != Error::Ok) {
    return error;
  }
  error = check_mudnn(handle.SetAllowTF32(false), "Handle.SetAllowTF32");
  if (error != Error::Ok) {
    return error;
  }
  ::musa::dnn::Tensor left;
  ::musa::dnn::Tensor right;
  ::musa::dnn::Tensor output;
  error = set_mudnn_tensor(
      left, packed_mat1 ? &*packed_mat1 : mat1, mat1_column_major);
  if (error != Error::Ok) {
    return error;
  }
  error = set_mudnn_tensor(right, mat2, mat2_column_major);
  if (error != Error::Ok) {
    return error;
  }
  error = set_mudnn_tensor(output, out);
  if (error != Error::Ok) {
    return error;
  }

  ::musa::dnn::MatMul operation;
  error = check_mudnn(
      operation.SetComputeMode(
          dtype == ScalarType::Float
              ? ::musa::dnn::MatMul::ComputeMode::SCALAR
              : ::musa::dnn::MatMul::ComputeMode::TENSOR),
      "MatMul.SetComputeMode");
  if (error != Error::Ok) {
    return error;
  }
  error = check_mudnn(
      operation.SetTranspose(mat1_column_major, mat2_column_major),
      "MatMul.SetTranspose");
  if (error != Error::Ok) {
    return error;
  }
  error = check_mudnn(operation.SetAlpha(alpha), "MatMul.SetAlpha");
  if (error != Error::Ok) {
    return error;
  }
  if (beta != 0.0 && self->dim() == 1) {
    ::musa::dnn::Tensor bias;
    error = set_mudnn_tensor(bias, self);
    if (error != Error::Ok) {
      return error;
    }
    error = check_mudnn(operation.SetGamma(beta), "MatMul.SetGamma");
    if (error != Error::Ok) {
      return error;
    }
    return check_mudnn(
        operation.RunWithBiasAdd(handle, output, left, right, bias),
        "MatMul.RunWithBiasAdd");
  }
  if (beta != 0.0) {
    if (out->data_ptr() != self->data_ptr()) {
      ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpyAsync(
          out->data_ptr(),
          self->data_ptr(),
          out->nbytes(),
          cudaMemcpyDeviceToDevice,
          stream));
    }
    error = check_mudnn(operation.SetBeta(beta), "MatMul.SetBeta");
    if (error != Error::Ok) {
      return error;
    }
  }
  return check_mudnn(operation.Run(handle, output, left, right), "MatMul.Run");
}

AOTI_SHIM_EXPORT AOTITorchError aoti_torch_musa_mm_out(
    SlimTensor* out,
    SlimTensor* mat1,
    SlimTensor* mat2) {
  return aoti_torch_musa_addmm_out(out, out, mat1, mat2, 0.0, 1.0);
}

AOTI_SHIM_EXPORT AOTITorchError aoti_torch_musa_bmm_out(
    SlimTensor* out,
    SlimTensor* mat1,
    SlimTensor* mat2) {
  ET_CHECK_OR_RETURN_ERROR(
      out != nullptr && mat1 != nullptr && mat2 != nullptr,
      InvalidArgument,
      "MUSA bmm received a null tensor handle");

  const auto dtype = out->dtype();
  ET_CHECK_OR_RETURN_ERROR(
      (dtype == ScalarType::Float || dtype == ScalarType::Half ||
       dtype == ScalarType::BFloat16) &&
          mat1->dtype() == dtype && mat2->dtype() == dtype,
      NotSupported,
      "MUSA bmm supports float32, float16, and bfloat16 tensors only");
  ET_CHECK_OR_RETURN_ERROR(
      out->is_cuda() && mat1->is_cuda() && mat2->is_cuda(),
      InvalidArgument,
      "MUSA bmm expects device tensors");
  ET_CHECK_OR_RETURN_ERROR(
      out->dim() == 3 && mat1->dim() == 3 && mat2->dim() == 3,
      NotSupported,
      "MUSA bmm expects three-dimensional tensors");

  const int64_t batch = mat1->size(0);
  const int64_t m = mat1->size(1);
  const int64_t k = mat1->size(2);
  const int64_t n = mat2->size(2);
  ET_CHECK_OR_RETURN_ERROR(
      mat2->size(0) == batch && mat2->size(1) == k &&
          out->size(0) == batch && out->size(1) == m && out->size(2) == n,
      InvalidArgument,
      "MUSA bmm tensor dimensions are incompatible");
  ET_CHECK_OR_RETURN_ERROR(
      fits_mublas_dimension(batch) && fits_mublas_dimension(m) &&
          fits_mublas_dimension(k) && fits_mublas_dimension(n),
      NotSupported,
      "MUSA bmm dimensions exceed mublas limits");

  const auto mat1_strides = mat1->strides();
  const bool mat1_row_major =
      (m == 1 || mat1_strides[1] == k) && mat1_strides[2] == 1 &&
      (batch <= 1 || mat1_strides[0] == 0 || mat1_strides[0] >= m * k);
  const bool mat1_column_major =
      mat1_strides[1] == 1 && mat1_strides[2] == m &&
      (batch <= 1 || mat1_strides[0] == 0 || mat1_strides[0] >= m * k);
  ET_CHECK_OR_RETURN_ERROR(
      out->is_contiguous() && (mat1_row_major || mat1_column_major),
      NotSupported,
      "MUSA bmm requires contiguous output and contiguous or transposed mat1 batches");

  const auto mat2_strides = mat2->strides();
  const bool mat2_row_major =
      mat2_strides[1] == n && mat2_strides[2] == 1 &&
      (batch <= 1 || mat2_strides[0] == 0 || mat2_strides[0] >= k * n);
  const bool mat2_column_major =
      mat2_strides[1] == 1 && mat2_strides[2] == k &&
      (batch <= 1 || mat2_strides[0] == 0 || mat2_strides[0] >= k * n);
  ET_CHECK_OR_RETURN_ERROR(
      mat2_row_major || mat2_column_major,
      NotSupported,
      "MUSA bmm supports contiguous or transposed mat2 only");
  ET_CHECK_OR_RETURN_ERROR(
      out->device_index() == mat1->device_index() &&
          out->device_index() == mat2->device_index(),
      InvalidArgument,
      "MUSA bmm tensors must use the same device");
  auto stream_result = getCurrentCUDAStream(out->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(stream_result.error());
  cudaStream_t stream = stream_result.get();

  if (batch > 0 && m > 0 && k > 0 && n > 0 &&
      mat1_strides[1] > 0 && mat2_strides[1] > 0 &&
      mat1_strides[0] == m * k && mat2_strides[0] == k * n &&
      out->strides()[0] == m * n) {
    ::musa::dnn::Handle handle(out->device_index());
    AOTITorchError error =
        check_mudnn(handle.SetStream(stream), "Handle.SetStream");
    if (error != Error::Ok) {
      return error;
    }
    error = check_mudnn(handle.SetAllowTF32(false), "Handle.SetAllowTF32");
    if (error != Error::Ok) {
      return error;
    }
    ::musa::dnn::Tensor left;
    ::musa::dnn::Tensor right;
    ::musa::dnn::Tensor output;
    error = set_mudnn_tensor(left, mat1, mat1_column_major);
    if (error != Error::Ok) {
      return error;
    }
    error = set_mudnn_tensor(right, mat2, mat2_column_major);
    if (error != Error::Ok) {
      return error;
    }
    error = set_mudnn_tensor(output, out);
    if (error != Error::Ok) {
      return error;
    }
    ::musa::dnn::BatchMatMul operation;
    error = check_mudnn(
        operation.SetComputeMode(
            dtype == ScalarType::Float
                ? ::musa::dnn::BatchMatMul::ComputeMode::SCALAR
                : ::musa::dnn::BatchMatMul::ComputeMode::TENSOR),
        "BatchMatMul.SetComputeMode");
    if (error != Error::Ok) {
      return error;
    }
    error = check_mudnn(
        operation.SetTranspose(mat1_column_major, mat2_column_major),
        "BatchMatMul.SetTranspose");
    if (error != Error::Ok) {
      return error;
    }
    error = check_mudnn(operation.SetAlpha(1.0), "BatchMatMul.SetAlpha");
    if (error != Error::Ok) {
      return error;
    }
    error = check_mudnn(operation.SetBeta(0.0), "BatchMatMul.SetBeta");
    if (error != Error::Ok) {
      return error;
    }
    return check_mudnn(
        operation.Run(handle, output, left, right), "BatchMatMul.Run");
  }

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

  const float alpha_float = 1.0f;
  const float beta_float = 0.0f;
  const executorch::runtime::etensor::Half alpha_half(alpha_float);
  const executorch::runtime::etensor::Half beta_half(beta_float);
  const void* alpha_ptr = dtype == ScalarType::Half
      ? static_cast<const void*>(&alpha_half)
      : static_cast<const void*>(&alpha_float);
  const void* beta_ptr = dtype == ScalarType::Half
      ? static_cast<const void*>(&beta_half)
      : static_cast<const void*>(&beta_float);
  const musaDataType_t data_type =
      dtype == ScalarType::Float
      ? MUSA_R_32F
      : (dtype == ScalarType::Half ? MUSA_R_16F : MUSA_R_16BF);
  const mublasComputeType_t compute_type =
      dtype == ScalarType::Half ? MUBLAS_COMPUTE_16F : MUBLAS_COMPUTE_32F;
  const mublasOperation_t mat2_operation =
      mat2_column_major ? MUBLAS_OP_T : MUBLAS_OP_N;
  const mublasOperation_t mat1_operation =
      mat1_row_major ? MUBLAS_OP_N : MUBLAS_OP_T;
  const int mat2_leading_dimension =
      static_cast<int>(mat2_column_major ? k : n);
  const int mat1_leading_dimension =
      static_cast<int>(mat1_row_major ? k : m);
  const long long mat2_batch_stride =
      static_cast<long long>(mat2_strides[0]);
  const long long mat1_batch_stride =
      static_cast<long long>(mat1_strides[0]);
  const long long out_batch_stride = static_cast<long long>(out->strides()[0]);

  if (mat1_batch_stride == 0 || mat2_batch_stride == 0) {
    const size_t element_size = out->itemsize();
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
      error = check_mublas(
          mublasGemmEx(
              handle,
              mat2_operation,
              mat1_operation,
              static_cast<int>(n),
              static_cast<int>(m),
              static_cast<int>(k),
              alpha_ptr,
              static_cast<const char*>(mat2->data_ptr()) +
                  batch_index * mat2_batch_stride * element_size,
              data_type,
              mat2_leading_dimension,
              static_cast<const char*>(mat1->data_ptr()) +
                  batch_index * mat1_batch_stride * element_size,
              data_type,
              mat1_leading_dimension,
              beta_ptr,
              static_cast<char*>(out->data_ptr()) +
                  batch_index * out_batch_stride * element_size,
              data_type,
              static_cast<int>(n),
              compute_type,
              MUBLAS_GEMM_DEFAULT),
          "mublasGemmEx");
      if (error != Error::Ok) {
        break;
      }
    }
  } else {
    error = check_mublas(
        mublasGemmStridedBatchedEx(
            handle,
            mat2_operation,
            mat1_operation,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            alpha_ptr,
            mat2->data_ptr(),
            data_type,
            mat2_leading_dimension,
            mat2_batch_stride,
            mat1->data_ptr(),
            data_type,
            mat1_leading_dimension,
            mat1_batch_stride,
            beta_ptr,
            out->data_ptr(),
            data_type,
            static_cast<int>(n),
            out_batch_stride,
            static_cast<int>(batch),
            compute_type,
            MUBLAS_GEMM_DEFAULT),
        "mublasGemmStridedBatchedEx");
  }

  const AOTITorchError destroy_error =
      check_mublas(mublasDestroy(handle), "mublasDestroy");
  return error == Error::Ok ? destroy_error : error;
}

} // extern "C"
} // namespace executorch::backends::cuda
