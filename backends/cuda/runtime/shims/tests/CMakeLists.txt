# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
project(aoti_cuda_shim_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
if(EXECUTORCH_BUILD_MUSA)
  find_package(MUSAToolkit REQUIRED)
  set(_gpu_runtime MUSA::musart)
  set(_gpu_include_dirs ${MUSAToolkit_INCLUDE_DIRS})
else()
  find_package(CUDAToolkit REQUIRED)
  set(_gpu_runtime CUDA::cudart)
  set(_gpu_include_dirs ${CUDAToolkit_INCLUDE_DIRS})
endif()

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE
)
FetchContent_MakeAvailable(googletest)

# Get EXECUTORCH_ROOT
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
endif()

# Find installed ExecuTorch
find_package(executorch CONFIG REQUIRED HINTS ${CMAKE_INSTALL_PREFIX})

# List of test files
set(CUDA_SHIM_TESTS
    test_aoti_torch_create_tensor_from_blob_v2
    test_aoti_torch_empty_strided
    test_aoti_torch_delete_tensor_object
    test_aoti_torch__reinterpret_tensor
    test_aoti_torch_copy_
    test_aoti_torch_zero_
    test_aoti_torch_cuda_rand
    test_aoti_torch_new_tensor_handle
    test_aoti_torch_item_bool
    test_aoti_torch_assign_tensors_out
)

# CUDA-specific tests requiring GPU kernels
set(CUDA_KERNEL_TESTS test_aoti_torch_cuda__weight_int4pack_mm)

if(EXECUTORCH_BUILD_MUSA)
  list(APPEND CUDA_SHIM_TESTS test_musa_addmm)
endif()

enable_testing()

foreach(test_name ${CUDA_SHIM_TESTS})
  add_executable(${test_name} ${test_name}.cpp)

  target_include_directories(
    ${test_name} PRIVATE ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}
                         ${_gpu_include_dirs}
  )

  target_compile_definitions(${test_name} PRIVATE CUDA_AVAILABLE=1)
  if(test_name STREQUAL "test_musa_addmm")
    target_compile_definitions(${test_name} PRIVATE EXECUTORCH_USE_MUSA=1)
  endif()

  target_link_libraries(
    ${test_name} PRIVATE GTest::gtest GTest::gtest_main aoti_cuda_shims
                         executorch_core ${_gpu_runtime}
  )

  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

foreach(test_name ${CUDA_KERNEL_TESTS})
  add_executable(${test_name} ${test_name}.cpp)

  target_include_directories(
    ${test_name} PRIVATE ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}
                         ${_gpu_include_dirs}
  )

  target_compile_definitions(${test_name} PRIVATE CUDA_AVAILABLE=1)

  target_link_libraries(
    ${test_name} PRIVATE GTest::gtest GTest::gtest_main aoti_cuda_shims
                         executorch_core ${_gpu_runtime}
  )

  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
