add_executable(placecell_example main.cpp)
target_link_libraries(placecell_example PRIVATE placecell::placecell)
target_compile_options(placecell_example PRIVATE -Wall -Wextra -Wpedantic)
