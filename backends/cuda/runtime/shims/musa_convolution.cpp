#include <mudnn.h>
#include <mudnncxx/mudnn_xmma.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {

using executorch::backends::aoti::slim::SlimTensor;
using executorch::backends::aoti::slim::c10::DeviceType;
using executorch::backends::aoti::slim::c10::ScalarType;
using executorch::runtime::Error;
using AOTITorchError = Error;

namespace {

AOTITorchError check_mudnn(mudnnStatus_t status, const char* operation) {
  if (status == MUDNN_STATUS_SUCCESS) {
    return Error::Ok;
  }
  ET_LOG(
      Error,
      "MUSA convolution: %s failed with mudnn status %d",
      operation,
      static_cast<int>(status));
  return Error::Internal;
}

bool fits_mudnn_dimension(int64_t value) {
  return value > 0 &&
      value <= static_cast<int64_t>(std::numeric_limits<int>::max());
}

mudnnDataType_t mudnn_data_type(ScalarType dtype) {
  if (dtype == ScalarType::Float) {
    return MUDNN_DATA_FLOAT;
  }
  if (dtype == ScalarType::Half) {
    return MUDNN_DATA_HALF;
  }
  return MUDNN_DATA_BFLOAT16;
}

AOTITorchError recommend_forward_algorithm(
    const SlimTensor& input,
    const SlimTensor& weight,
    const SlimTensor& output,
    const int64_t* padding,
    const int64_t* stride,
    const int64_t* dilation,
    int groups,
    cudaStream_t stream,
    mudnnConvolutionFwdAlgo_t& algorithm) {
  using Tensor = ::musa::dnn::Tensor;
  using Convolution = ::musa::dnn::Convolution;
  auto check = [](::musa::dnn::Status status) {
    ET_CHECK_OR_RETURN_ERROR(
        status == ::musa::dnn::Status::SUCCESS,
        Internal,
        "MUSA convolution recommendation failed with muDNN C++ status %d",
        static_cast<int>(status));
    return Error::Ok;
  };
  ::musa::dnn::Handle handle(input.device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(check(handle.SetStream(stream)));
  const auto dtype = input.dtype() == ScalarType::Float
      ? Tensor::Type::FLOAT
      : (input.dtype() == ScalarType::Half ? Tensor::Type::HALF
                                           : Tensor::Type::BFLOAT16);
  auto configure = [&](Tensor& descriptor, const SlimTensor& tensor) {
    std::vector<int64_t> sizes(tensor.sizes().begin(), tensor.sizes().end());
    if (sizes.size() == 3) {
      sizes.push_back(1);
    }
    ET_CHECK_OK_OR_RETURN_ERROR(check(descriptor.SetAddr(tensor.data_ptr())));
    ET_CHECK_OK_OR_RETURN_ERROR(check(descriptor.SetType(dtype)));
    ET_CHECK_OK_OR_RETURN_ERROR(check(descriptor.SetFormat(
        sizes.size() == 4 ? Tensor::Format::NCHW : Tensor::Format::NCDHW)));
    return check(
        descriptor.SetNdInfo(static_cast<int>(sizes.size()), sizes.data()));
  };
  Tensor input_descriptor;
  Tensor weight_descriptor;
  Tensor output_descriptor;
  ET_CHECK_OK_OR_RETURN_ERROR(configure(input_descriptor, input));
  ET_CHECK_OK_OR_RETURN_ERROR(configure(weight_descriptor, weight));
  ET_CHECK_OK_OR_RETURN_ERROR(configure(output_descriptor, output));
  const int dimensions = input.dim() == 3 ? 2 : input.dim() - 2;
  std::vector<int> padding_values(dimensions, 0);
  std::vector<int> stride_values(dimensions, 1);
  std::vector<int> dilation_values(dimensions, 1);
  for (int index = 0; index < input.dim() - 2; ++index) {
    padding_values[index] = static_cast<int>(padding[index]);
    stride_values[index] = static_cast<int>(stride[index]);
    dilation_values[index] = static_cast<int>(dilation[index]);
  }
  Convolution convolution;
  ET_CHECK_OK_OR_RETURN_ERROR(check(convolution.SetNdInfo(
      dimensions,
      padding_values.data(),
      stride_values.data(),
      dilation_values.data())));
  ET_CHECK_OK_OR_RETURN_ERROR(check(convolution.SetGroups(groups)));
  ET_CHECK_OK_OR_RETURN_ERROR(check(convolution.SetComputeMode(
      input.dtype() == ScalarType::Float ? Convolution::ComputeMode::SCALAR
                                         : Convolution::ComputeMode::TENSOR)));
  Convolution::Algorithm recommended;
  ET_CHECK_OK_OR_RETURN_ERROR(check(convolution.GetRecommendForwardAlgorithm(
      handle,
      recommended,
      output_descriptor,
      input_descriptor,
      weight_descriptor)));
  switch (recommended) {
    case Convolution::Algorithm::IMPLICIT_GEMM:
      algorithm = MUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
      break;
    case Convolution::Algorithm::DIRECT:
      algorithm = MUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
      break;
    case Convolution::Algorithm::WINOGRAD_NONFUSED:
      algorithm = MUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
      break;
    case Convolution::Algorithm::GEMM:
      algorithm = MUDNN_CONVOLUTION_FWD_ALGO_GEMM;
      break;
    default:
      return Error::NotSupported;
  }
  return Error::Ok;
}

struct MudnnContext {
  mudnnHandle_t handle = nullptr;
  mudnnTensorDescriptor_t input = nullptr;
  mudnnTensorDescriptor_t bias = nullptr;
  mudnnTensorDescriptor_t output = nullptr;
  mudnnFilterDescriptor_t weight = nullptr;
  mudnnConvolutionDescriptor_t convolution = nullptr;
  mudnnActivationDescriptor_t activation = nullptr;

  ~MudnnContext() {
    if (activation != nullptr) {
      mudnnDestroyActivationDescriptor(activation);
    }
    if (convolution != nullptr) {
      mudnnDestroyConvolutionDescriptor(convolution);
    }
    if (weight != nullptr) {
      mudnnDestroyFilterDescriptor(weight);
    }
    if (output != nullptr) {
      mudnnDestroyTensorDescriptor(output);
    }
    if (bias != nullptr) {
      mudnnDestroyTensorDescriptor(bias);
    }
    if (input != nullptr) {
      mudnnDestroyTensorDescriptor(input);
    }
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
  }
};

struct DeviceWorkspace {
  void* data = nullptr;

  ~DeviceWorkspace() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }
};

}

extern "C" {

AOTI_SHIM_EXPORT AOTITorchError aoti_torch_musa_convolution(
    SlimTensor* input,
    SlimTensor* weight,
    SlimTensor* const* bias_handle,
    const int64_t* stride,
    int64_t stride_len,
    const int64_t* padding,
    int64_t padding_len,
    const int64_t* dilation,
    int64_t dilation_len,
    int32_t transposed,
    const int64_t* output_padding,
    int64_t output_padding_len,
    int64_t groups,
    SlimTensor** ret_output) {
  ET_CHECK_OR_RETURN_ERROR(
      input != nullptr && weight != nullptr && ret_output != nullptr,
      InvalidArgument,
      "MUSA convolution received a null tensor handle");
  *ret_output = nullptr;
  SlimTensor* bias = bias_handle != nullptr ? *bias_handle : nullptr;

  const auto dtype = input->dtype();
  ET_CHECK_OR_RETURN_ERROR(
      (dtype == ScalarType::Float || dtype == ScalarType::Half ||
       dtype == ScalarType::BFloat16) && weight->dtype() == dtype &&
          (bias == nullptr || bias->dtype() == dtype),
      NotSupported,
      "MUSA convolution supports float32, float16, and bfloat16 tensors only");
  ET_CHECK_OR_RETURN_ERROR(
      input->is_gpu() && weight->is_gpu() &&
          (bias == nullptr || bias->is_gpu()),
      InvalidArgument,
      "MUSA convolution expects device tensors");
  ET_CHECK_OR_RETURN_ERROR(
      input->device_index() == weight->device_index() &&
          (bias == nullptr || input->device_index() == bias->device_index()),
      InvalidArgument,
      "MUSA convolution tensors must use the same device");
  ET_CHECK_OR_RETURN_ERROR(
      input->dim() >= 3 && input->dim() <= 5 &&
          weight->dim() == input->dim(),
      NotSupported,
      "MUSA convolution supports 1D, 2D, and 3D convolution only");
  const int64_t spatial_dims = input->dim() - 2;
  ET_CHECK_OR_RETURN_ERROR(
      stride != nullptr && padding != nullptr && dilation != nullptr &&
          output_padding != nullptr &&
          stride_len == spatial_dims && padding_len == spatial_dims &&
          dilation_len == spatial_dims && output_padding_len == spatial_dims,
      NotSupported,
      "MUSA convolution expects one stride, padding, dilation, and output "
      "padding per spatial dimension");
  for (int64_t index = 0; index < spatial_dims; ++index) {
    ET_CHECK_OR_RETURN_ERROR(
        stride[index] > 0 && dilation[index] > 0 && padding[index] >= 0 &&
            output_padding[index] >= 0 &&
            (transposed ? (output_padding[index] < stride[index] ||
                           output_padding[index] < dilation[index])
                        : output_padding[index] == 0),
        InvalidArgument,
        "MUSA convolution received invalid convolution parameters");
  }
  ET_CHECK_OR_RETURN_ERROR(
      fits_mudnn_dimension(groups),
      InvalidArgument,
      "MUSA convolution received invalid convolution parameters");
  ET_CHECK_OR_RETURN_ERROR(
      !transposed ||
          (weight->size(1) > 0 &&
           groups <= std::numeric_limits<int64_t>::max() / weight->size(1)),
      InvalidArgument,
      "MUSA convolution tensor dimensions are incompatible");

  std::unique_ptr<SlimTensor> contiguous_input;
  std::unique_ptr<SlimTensor> contiguous_weight;
  std::unique_ptr<SlimTensor> contiguous_bias;
  const SlimTensor* input_for_mudnn = input;
  const SlimTensor* weight_for_mudnn = weight;
  const SlimTensor* bias_for_mudnn = bias;
  if (!input->is_contiguous()) {
    contiguous_input = std::make_unique<SlimTensor>(input->clone_contiguous());
    input_for_mudnn = contiguous_input.get();
  }
  if (!weight->is_contiguous()) {
    contiguous_weight =
        std::make_unique<SlimTensor>(weight->clone_contiguous());
    weight_for_mudnn = contiguous_weight.get();
  }
  if (bias != nullptr && !bias->is_contiguous()) {
    contiguous_bias = std::make_unique<SlimTensor>(bias->clone_contiguous());
    bias_for_mudnn = contiguous_bias.get();
  }

  const int64_t batch = input->size(0);
  const int64_t input_channels = input->size(1);
  const int64_t output_channels =
      transposed ? weight->size(1) * groups : weight->size(0);
  ET_CHECK_OR_RETURN_ERROR(
      input_channels % groups == 0 && output_channels % groups == 0 &&
          (transposed ? weight->size(0) == input_channels
                      : weight->size(1) == input_channels / groups) &&
          (bias == nullptr || bias->dim() == 1) &&
          (bias == nullptr || bias->size(0) == output_channels),
      InvalidArgument,
      "MUSA convolution tensor dimensions are incompatible");

  std::vector<int64_t> output_sizes(input->dim());
  output_sizes[0] = batch;
  output_sizes[1] = output_channels;
  for (int64_t index = 0; index < spatial_dims; ++index) {
    const int64_t input_size = input->size(index + 2);
    const int64_t kernel_size = weight->size(index + 2);
    output_sizes[index + 2] = transposed
        ? (input_size - 1) * stride[index] - 2 * padding[index] +
            dilation[index] * (kernel_size - 1) + output_padding[index] + 1
        : (input_size + 2 * padding[index] -
           dilation[index] * (kernel_size - 1) - 1) /
                stride[index] +
            1;
  }
  std::vector<int64_t> output_strides(output_sizes.size());
  output_strides.back() = 1;
  for (int64_t index = output_strides.size() - 2; index >= 0; --index) {
    output_strides[index] = output_strides[index + 1] * output_sizes[index + 1];
  }
  ET_CHECK_OR_RETURN_ERROR(
      fits_mudnn_dimension(batch) && fits_mudnn_dimension(input_channels) &&
          fits_mudnn_dimension(output_channels),
      NotSupported,
      "MUSA convolution dimensions exceed muDNN limits");
  for (int64_t size : output_sizes) {
    ET_CHECK_OR_RETURN_ERROR(
        fits_mudnn_dimension(size),
        NotSupported,
        "MUSA convolution dimensions exceed muDNN limits");
  }
  SlimTensor* raw_output = nullptr;
  auto error = aoti_torch_empty_strided(
      output_sizes.size(),
      output_sizes.data(),
      output_strides.data(),
      static_cast<int32_t>(dtype),
      executorch::backends::aoti::slim::c10::device_type_to_aoti(
          executorch::backends::aoti::slim::c10::kGPU),
      input->device_index(),
      &raw_output);
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  std::unique_ptr<SlimTensor> output(raw_output);

  MudnnContext context;
  error = check_mudnn(mudnnCreate(&context.handle), "mudnnCreate");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  auto stream_result = getCurrentCUDAStream(input->device_index());
  ET_CHECK_OK_OR_RETURN_ERROR(stream_result.error());
  error = check_mudnn(
      mudnnSetStream(context.handle, stream_result.get()), "mudnnSetStream");
  ET_CHECK_OK_OR_RETURN_ERROR(error);

  error = check_mudnn(
      mudnnCreateTensorDescriptor(&context.input), "mudnnCreateTensorDescriptor");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  error = check_mudnn(
      mudnnCreateTensorDescriptor(&context.output), "mudnnCreateTensorDescriptor");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  error = check_mudnn(
      mudnnCreateFilterDescriptor(&context.weight), "mudnnCreateFilterDescriptor");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  error = check_mudnn(
      mudnnCreateConvolutionDescriptor(&context.convolution),
      "mudnnCreateConvolutionDescriptor");
  ET_CHECK_OK_OR_RETURN_ERROR(error);

  const auto data_type = mudnn_data_type(dtype);
  const bool one_dim_as_two = spatial_dims == 1;
  const int descriptor_dims = one_dim_as_two ? 4 : static_cast<int>(input->dim());
  std::vector<int> input_dims(descriptor_dims);
  std::vector<int> output_dims(descriptor_dims);
  std::vector<int> weight_dims(descriptor_dims);
  if (one_dim_as_two) {
    input_dims = {
        static_cast<int>(input->size(0)),
        static_cast<int>(input->size(1)),
        static_cast<int>(input->size(2)),
        1};
    output_dims = {
        static_cast<int>(output_sizes[0]),
        static_cast<int>(output_sizes[1]),
        static_cast<int>(output_sizes[2]),
        1};
    weight_dims = {
        static_cast<int>(weight->size(0)),
        static_cast<int>(weight->size(1)),
        static_cast<int>(weight->size(2)),
        1};
  } else {
    for (int index = 0; index < descriptor_dims; ++index) {
      input_dims[index] = static_cast<int>(input->size(index));
      output_dims[index] = static_cast<int>(output_sizes[index]);
      weight_dims[index] = static_cast<int>(weight->size(index));
    }
  }
  std::vector<int> padding_int(spatial_dims);
  std::vector<int> stride_int(spatial_dims);
  std::vector<int> dilation_int(spatial_dims);
  for (int64_t index = 0; index < spatial_dims; ++index) {
    padding_int[index] = static_cast<int>(padding[index]);
    stride_int[index] = static_cast<int>(stride[index]);
    dilation_int[index] = static_cast<int>(dilation[index]);
  }
  error = check_mudnn(
      mudnnSetTensorNdDescriptorEx(
          context.input,
          one_dim_as_two || descriptor_dims == 4
              ? MUDNN_TENSOR_NCHW
              : MUDNN_TENSOR_NCDHW,
          data_type,
          descriptor_dims,
          input_dims.data()),
      "mudnnSetTensorNdDescriptorEx(input)");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  error = check_mudnn(
      mudnnSetTensorNdDescriptorEx(
          context.output,
          one_dim_as_two || descriptor_dims == 4
              ? MUDNN_TENSOR_NCHW
              : MUDNN_TENSOR_NCDHW,
          data_type,
          descriptor_dims,
          output_dims.data()),
      "mudnnSetTensorNdDescriptorEx(output)");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  const auto tensor_format = one_dim_as_two
      ? MUDNN_TENSOR_NCHW
      : (descriptor_dims == 4 ? MUDNN_TENSOR_NCHW : MUDNN_TENSOR_NCDHW);
  error = check_mudnn(
      mudnnSetFilterNdDescriptor(
          context.weight,
          data_type,
          tensor_format,
          descriptor_dims,
          weight_dims.data()),
      "mudnnSetFilterNdDescriptor");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  if (one_dim_as_two) {
    error = check_mudnn(
        mudnnSetConvolution2dDescriptor(
            context.convolution,
            padding_int[0],
            0,
            stride_int[0],
            1,
            dilation_int[0],
            1,
            MUDNN_CROSS_CORRELATION,
            MUDNN_DATA_FLOAT),
        "mudnnSetConvolution2dDescriptor");
  } else if (descriptor_dims == 4) {
    error = check_mudnn(
        mudnnSetConvolution2dDescriptor(
            context.convolution,
            padding_int[0],
            padding_int[1],
            stride_int[0],
            stride_int[1],
            dilation_int[0],
            dilation_int[1],
            MUDNN_CROSS_CORRELATION,
            MUDNN_DATA_FLOAT),
        "mudnnSetConvolution2dDescriptor");
  } else {
    error = check_mudnn(
        mudnnSetConvolutionNdDescriptor(
            context.convolution,
            static_cast<int>(spatial_dims),
            padding_int.data(),
            stride_int.data(),
            dilation_int.data(),
            MUDNN_CROSS_CORRELATION,
            MUDNN_DATA_FLOAT),
        "mudnnSetConvolutionNdDescriptor");
  }
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  error = check_mudnn(
      mudnnSetConvolutionGroupCount(
          context.convolution, static_cast<int>(groups)),
      "mudnnSetConvolutionGroupCount");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  if (dtype == ScalarType::Float) {
    error = check_mudnn(
        mudnnSetConvolutionMathType(context.convolution, MUDNN_FMA_MATH),
        "mudnnSetConvolutionMathType");
    ET_CHECK_OK_OR_RETURN_ERROR(error);
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto forward_algorithm = MUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  if (!transposed) {
    ET_CHECK_OK_OR_RETURN_ERROR(recommend_forward_algorithm(
        *input_for_mudnn,
        *weight_for_mudnn,
        *output,
        padding,
        stride,
        dilation,
        static_cast<int>(groups),
        stream_result.get(),
        forward_algorithm));
  }
  const auto backward_data_algorithm = transposed && one_dim_as_two
      ? MUDNN_CONVOLUTION_BWD_DATA_ALGO_GEMM
      : MUDNN_CONVOLUTION_BWD_DATA_ALGO_IMPLICIT_GEMM;
  size_t workspace_size = 0;
  error = transposed
      ? check_mudnn(
            mudnnGetConvolutionBackwardDataWorkspaceSize(
                context.handle,
                context.weight,
                context.input,
                context.convolution,
                context.output,
                backward_data_algorithm,
                &workspace_size),
            "mudnnGetConvolutionBackwardDataWorkspaceSize")
      : check_mudnn(
            mudnnGetConvolutionForwardWorkspaceSize(
                context.handle,
                context.input,
                context.weight,
                context.convolution,
                context.output,
                forward_algorithm,
                &workspace_size),
            "mudnnGetConvolutionForwardWorkspaceSize");
  ET_CHECK_OK_OR_RETURN_ERROR(error);
  DeviceWorkspace workspace;
  if (workspace_size != 0) {
    if (cudaMalloc(&workspace.data, workspace_size) != cudaSuccess) {
      ET_LOG(Error, "MUSA convolution: cudaMalloc workspace failed");
      return Error::Internal;
    }
  }
  if (bias != nullptr) {
    std::vector<int> bias_dims(descriptor_dims, 1);
    bias_dims[1] = static_cast<int>(output_channels);
    error = check_mudnn(
        mudnnCreateTensorDescriptor(&context.bias),
        "mudnnCreateTensorDescriptor");
    ET_CHECK_OK_OR_RETURN_ERROR(error);
    if (transposed) {
      error = check_mudnn(
          mudnnSetTensorNdDescriptorEx(
              context.bias,
              tensor_format,
              data_type,
              descriptor_dims,
              bias_dims.data()),
          "mudnnSetTensorNdDescriptorEx(bias)");
    } else {
      const int bias_size = static_cast<int>(output_channels);
      const int bias_stride = 1;
      error = check_mudnn(
          mudnnSetTensorNdDescriptor(
              context.bias, data_type, 1, &bias_size, &bias_stride),
          "mudnnSetTensorNdDescriptor(bias)");
    }
    ET_CHECK_OK_OR_RETURN_ERROR(error);
  }
  if (!transposed && bias != nullptr) {
    error = check_mudnn(
        mudnnCreateActivationDescriptor(&context.activation),
        "mudnnCreateActivationDescriptor");
    ET_CHECK_OK_OR_RETURN_ERROR(error);
    error = check_mudnn(
        mudnnSetActivationDescriptor(
            context.activation,
            MUDNN_ACTIVATION_IDENTITY,
            MUDNN_NOT_PROPAGATE_NAN,
            0.0),
        "mudnnSetActivationDescriptor");
    ET_CHECK_OK_OR_RETURN_ERROR(error);
    error = check_mudnn(
        mudnnConvolutionBiasActivationForward(
            context.handle,
            &alpha,
            context.input,
            input_for_mudnn->data_ptr(),
            context.weight,
            weight_for_mudnn->data_ptr(),
            context.convolution,
            forward_algorithm,
            workspace.data,
            workspace_size,
            &beta,
            nullptr,
            nullptr,
            context.bias,
            bias_for_mudnn->data_ptr(),
            context.activation,
            context.output,
            output->data_ptr()),
        "mudnnConvolutionBiasActivationForward");
  } else {
    error = transposed ? check_mudnn(
                             mudnnConvolutionBackwardData(
                                 context.handle,
                                 &alpha,
                                 context.weight,
                                 weight_for_mudnn->data_ptr(),
                                 context.input,
                                 input_for_mudnn->data_ptr(),
                                 context.convolution,
                                 backward_data_algorithm,
                                 workspace.data,
                                 workspace_size,
                                 &beta,
                                 context.output,
                                 output->data_ptr()),
                             "mudnnConvolutionBackwardData")
                       : check_mudnn(
                             mudnnConvolutionForward(
                                 context.handle,
                                 &alpha,
                                 context.input,
                                 input_for_mudnn->data_ptr(),
                                 context.weight,
                                 weight_for_mudnn->data_ptr(),
                                 context.convolution,
                                 forward_algorithm,
                                 workspace.data,
                                 workspace_size,
                                 &beta,
                                 context.output,
                                 output->data_ptr()),
                             "mudnnConvolutionForward");
  }
  ET_CHECK_OK_OR_RETURN_ERROR(error);

  if (transposed && bias != nullptr) {
    error = check_mudnn(
        mudnnAddTensor(
            context.handle,
            &alpha,
            context.bias,
            bias_for_mudnn->data_ptr(),
            &alpha,
            context.output,
            output->data_ptr()),
        "mudnnAddTensor");
    ET_CHECK_OK_OR_RETURN_ERROR(error);
  }

  *ret_output = output.release();
  return Error::Ok;
}
}
}
