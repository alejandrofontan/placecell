cmake_minimum_required(VERSION 3.18)
project(placecell VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PLACECELL_BUILD_EXAMPLES "Build the example executables" ON)
option(PLACECELL_BUILD_PYTHON "Build the Python bindings (nanobind)" OFF)
option(PLACECELL_WITH_MEGALOC "Build the MegaLoc embedder (CUDA + TensorRT)" OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

if(PLACECELL_WITH_MEGALOC)
    find_package(CUDAToolkit REQUIRED)   # CUDA::cudart (cudaMalloc, streams, memcpy)
    find_package(OpenCV REQUIRED)        # image I/O + preprocessing
    find_package(yaml-cpp REQUIRED)      # sidecar <onnx>.yaml parsing

    # TensorRT ships no CMake config - locate headers/libs in the pixi env manually.
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS $ENV{CONDA_PREFIX}/include)
    find_library(NVINFER_LIB nvinfer HINTS $ENV{CONDA_PREFIX}/lib)
    find_library(NVONNXPARSER_LIB nvonnxparser HINTS $ENV{CONDA_PREFIX}/lib)
    if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB)
        message(FATAL_ERROR "TensorRT not found - run the build inside the pixi env "
                            "(CONDA_PREFIX=$ENV{CONDA_PREFIX})")
    endif()
    message(STATUS "TensorRT (MegaLoc): ${NVINFER_LIB}")

    add_library(placecell_megaloc
        src/megaloc/tensorrt_megaloc.cpp
        src/megaloc/megaloc_embedder.cpp
    )
    add_library(placecell::megaloc ALIAS placecell_megaloc)

    target_include_directories(placecell_megaloc PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/megaloc   # tensorrt_megaloc.hpp, until Step 4's public header exists
        ${TENSORRT_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(placecell_megaloc
        PUBLIC ${NVINFER_LIB} ${NVONNXPARSER_LIB} ${OpenCV_LIBS} CUDA::cudart yaml-cpp
        PUBLIC placecell::placecell
    )

    target_compile_options(placecell_megaloc PRIVATE -Wall -Wextra)
    set_target_properties(placecell_megaloc PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# ---- Library ----------------------------------------------------------------
add_library(placecell
    src/placecell.cpp
)
add_library(placecell::placecell ALIAS placecell)

target_include_directories(placecell PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(placecell PUBLIC Eigen3::Eigen)
target_compile_options(placecell PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(placecell PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Examples ---------------------------------------------------------------
if(PLACECELL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---- Python bindings --------------------------------------------------------
if(PLACECELL_BUILD_PYTHON)
    add_subdirectory(python)
endif()
