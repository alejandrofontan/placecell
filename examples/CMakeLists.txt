add_executable(placecell_example main.cpp)
target_link_libraries(placecell_example PRIVATE placecell::placecell)
target_compile_options(placecell_example PRIVATE -Wall -Wextra -Wpedantic)

if(TARGET placecell::megaloc)
    add_executable(megaloc_test test_megaloc.cpp)
    target_link_libraries(megaloc_test PRIVATE placecell::megaloc)
    target_compile_options(megaloc_test PRIVATE -Wall -Wextra)
endif()