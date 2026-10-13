cmake_minimum_required(VERSION 3.18)
project(placecell VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PLACECELL_BUILD_EXAMPLES "Build the example executables" ON)
option(PLACECELL_BUILD_PYTHON "Build the Python bindings (nanobind)" OFF)
option(PLACECELL_WITH_MEGALOC "Build the MegaLoc embedder (CUDA + TensorRT)" OFF)
option(PLACECELL_WITH_VIZ "Build the visualization module (OpenCV)" OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# The inference backend of the MegaLoc embedder: "onnxruntime" (default; one C++ path, the
# hardware is an execution-provider choice: CUDA on Linux, CoreML on macOS, CPU anywhere) or
# "tensorrt" (NVIDIA only, builds a GPU-specific engine; kept until ONNX Runtime parity is
# measured, see AllFeature-VSLAM docs/plans/megaloc_onnxruntime.md).
set(PLACECELL_MEGALOC_BACKEND "onnxruntime" CACHE STRING "MegaLoc inference backend: onnxruntime | tensorrt")
set_property(CACHE PLACECELL_MEGALOC_BACKEND PROPERTY STRINGS onnxruntime tensorrt)

if(PLACECELL_WITH_MEGALOC)
    find_package(OpenCV REQUIRED)        # image I/O + preprocessing
    find_package(yaml-cpp REQUIRED)      # sidecar <onnx>.yaml parsing

    set(_megaloc_sources
        src/megaloc/megaloc_common.cpp
        src/megaloc/megaloc_embedder.cpp
        src/megaloc/megaloc_placecell.cpp
    )
    set(_megaloc_includes ${CMAKE_CURRENT_SOURCE_DIR}/src/megaloc ${OpenCV_INCLUDE_DIRS})
    set(_megaloc_libs ${OpenCV_LIBS} yaml-cpp)
    set(_megaloc_defines)

    if(PLACECELL_MEGALOC_BACKEND STREQUAL "onnxruntime")
        # conda-forge's onnxruntime-cpp ships a CMake config (target onnxruntime::onnxruntime); fall back to
        # the headers/library under the pixi env for other layouts. The CUDA / CoreML execution providers
        # live inside the ONNX Runtime library (the CUDA one dlopens its CUDA libs at runtime), so no CUDA
        # toolkit is needed here.
        find_package(onnxruntime CONFIG QUIET)
        if(onnxruntime_FOUND)
            message(STATUS "ONNX Runtime (MegaLoc): ${onnxruntime_DIR}")
            list(APPEND _megaloc_libs onnxruntime::onnxruntime)
            # conda-forge keeps the source layout (include/onnxruntime/core/session/onnxruntime_cxx_api.h)
            # while the release tarballs are flat; add the subdirectory so <onnxruntime_cxx_api.h> works on both
            get_target_property(_ort_includes onnxruntime::onnxruntime INTERFACE_INCLUDE_DIRECTORIES)
            foreach(_dir ${_ort_includes})
                if(EXISTS ${_dir}/core/session/onnxruntime_cxx_api.h)
                    list(APPEND _megaloc_includes ${_dir}/core/session)
                endif()
            endforeach()
        else()
            find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
                      HINTS $ENV{CONDA_PREFIX}/include/onnxruntime $ENV{CONDA_PREFIX}/include/onnxruntime/core/session
                            $ENV{CONDA_PREFIX}/include)
            find_library(ONNXRUNTIME_LIB onnxruntime HINTS $ENV{CONDA_PREFIX}/lib)
            if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
                message(FATAL_ERROR "ONNX Runtime not found - install onnxruntime-cpp in the pixi env "
                                    "(CONDA_PREFIX=$ENV{CONDA_PREFIX}) or set PLACECELL_MEGALOC_BACKEND=tensorrt")
            endif()
            message(STATUS "ONNX Runtime (MegaLoc): ${ONNXRUNTIME_LIB}")
            list(APPEND _megaloc_includes ${ONNXRUNTIME_INCLUDE_DIR})
            list(APPEND _megaloc_libs ${ONNXRUNTIME_LIB})
        endif()
        list(APPEND _megaloc_sources src/megaloc/onnxruntime_megaloc.cpp)
        list(APPEND _megaloc_defines PLACECELL_MEGALOC_ONNXRUNTIME=1)
    elseif(PLACECELL_MEGALOC_BACKEND STREQUAL "tensorrt")
        find_package(CUDAToolkit REQUIRED)   # CUDA::cudart (cudaMalloc, streams, memcpy)
        # TensorRT ships no CMake config - locate headers/libs in the pixi env manually.
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS $ENV{CONDA_PREFIX}/include)
        find_library(NVINFER_LIB nvinfer HINTS $ENV{CONDA_PREFIX}/lib)
        find_library(NVONNXPARSER_LIB nvonnxparser HINTS $ENV{CONDA_PREFIX}/lib)
        if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB)
            message(FATAL_ERROR "TensorRT not found - run the build inside the pixi env "
                                "(CONDA_PREFIX=$ENV{CONDA_PREFIX})")
        endif()
        message(STATUS "TensorRT (MegaLoc): ${NVINFER_LIB}")
        list(APPEND _megaloc_sources src/megaloc/tensorrt_megaloc.cpp)
        list(APPEND _megaloc_includes ${TENSORRT_INCLUDE_DIR})
        list(APPEND _megaloc_libs ${NVINFER_LIB} ${NVONNXPARSER_LIB} CUDA::cudart)
        list(APPEND _megaloc_defines PLACECELL_MEGALOC_TENSORRT=1)
    else()
        message(FATAL_ERROR "PLACECELL_MEGALOC_BACKEND must be onnxruntime or tensorrt, got '${PLACECELL_MEGALOC_BACKEND}'")
    endif()
    message(STATUS "MegaLoc backend: ${PLACECELL_MEGALOC_BACKEND}")

    add_library(placecell_megaloc ${_megaloc_sources})
    add_library(placecell::megaloc ALIAS placecell_megaloc)

    target_include_directories(placecell_megaloc PUBLIC ${_megaloc_includes})
    target_compile_definitions(placecell_megaloc PUBLIC ${_megaloc_defines})
    target_link_libraries(placecell_megaloc
        PUBLIC ${_megaloc_libs}
        PUBLIC placecell::placecell
    )

    target_compile_options(placecell_megaloc PRIVATE -Wall -Wextra)
    set_target_properties(placecell_megaloc PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# ---- Library ----------------------------------------------------------------
add_library(placecell
    src/placecell.cpp
    src/placecell_items.cpp
    src/placecell_query.cpp
    src/placecell_events.cpp
    src/placecell_cull.cpp
    src/placecell_gram_greedy.cpp
    src/log.cpp
    src/profiler.cpp
    src/recorder.cpp
    src/kernel_io.cpp
)
add_library(placecell::placecell ALIAS placecell)

target_include_directories(placecell PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(placecell PUBLIC Eigen3::Eigen)
target_compile_options(placecell PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(placecell PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Visualization (OpenCV) --------------------------------------------------
# Kernel heatmap, unexplained-information history and alive-information strip, rendered
# to cv::Mat from a PlaceCell + its Recorder; optional windows via cv::imshow.
if(PLACECELL_WITH_VIZ)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)
    add_library(placecell_viz
        src/viz/viz.cpp
    )
    add_library(placecell::viz ALIAS placecell_viz)
    target_include_directories(placecell_viz PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(placecell_viz PUBLIC placecell::placecell ${OpenCV_LIBS})
    target_compile_options(placecell_viz PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(placecell_viz PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# ---- Examples ---------------------------------------------------------------
if(PLACECELL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---- Python bindings --------------------------------------------------------
if(PLACECELL_BUILD_PYTHON)
    add_subdirectory(python)
endif()
