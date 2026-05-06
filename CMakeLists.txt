cmake_minimum_required(VERSION 3.18)
project(ggml_cuda_add LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

# Paths provided by user
set(GGML_INCLUDE_DIR "$ENV{HOME}/sdk/src/ggml/include")
set(GGML_LIB_DIR "$ENV{HOME}/sdk/build/ggml/cuda/src")
set(GGML_CUDA_LIB_DIR "${GGML_LIB_DIR}/ggml-cuda")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

include_directories(${GGML_INCLUDE_DIR})

# Find libraries
find_library(GGML_LIB ggml PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
find_library(GGML_BASE_LIB ggml-base PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
find_library(GGML_CPU_LIB ggml-cpu PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
find_library(GGML_CUDA_LIB ggml-cuda PATHS ${GGML_CUDA_LIB_DIR} NO_DEFAULT_PATH)

# Common libraries
set(GGML_LIBS
    ${GGML_LIB}
    ${GGML_BASE_LIB}
    ${GGML_CPU_LIB}
    ${GGML_CUDA_LIB}
    CUDA::cudart
)

# GGML Add (Unified)
add_executable(ggml_add ggml_add.cpp)
target_link_libraries(ggml_add ${GGML_LIBS})
