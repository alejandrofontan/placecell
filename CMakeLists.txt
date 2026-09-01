cmake_minimum_required(VERSION 3.18)
project(placecell VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PLACECELL_BUILD_EXAMPLES "Build the example executables" ON)
option(PLACECELL_BUILD_PYTHON "Build the Python bindings (nanobind)" OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

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
