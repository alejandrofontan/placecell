find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# Locate nanobind through the Python package (works in pixi/conda and pip envs alike)
execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_placecell bindings.cpp)
target_link_libraries(_placecell PRIVATE placecell::placecell)

# `pip install .` (scikit-build-core) places the module inside the placecell package
if(SKBUILD)
    install(TARGETS _placecell LIBRARY DESTINATION placecell)
endif()
