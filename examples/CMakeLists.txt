add_executable(placecell_example main.cpp)
target_link_libraries(placecell_example PRIVATE placecell::placecell)
target_compile_options(placecell_example PRIVATE -Wall -Wextra -Wpedantic)

# GPU-free end-to-end exercise + smoke test of the core and its managers (log / profile /
# record); renders the plots too when the viz module is built.
add_executable(synthetic_demo synthetic_demo.cpp)
target_link_libraries(synthetic_demo PRIVATE placecell::placecell)
target_compile_options(synthetic_demo PRIVATE -Wall -Wextra -Wpedantic)
if(TARGET placecell::viz)
    target_link_libraries(synthetic_demo PRIVATE placecell::viz)
    target_compile_definitions(synthetic_demo PRIVATE PLACECELL_HAS_VIZ)
endif()

if(TARGET placecell::megaloc)
    add_executable(megaloc_test test_megaloc.cpp)
    target_link_libraries(megaloc_test PRIVATE placecell::megaloc)
    target_compile_options(megaloc_test PRIVATE -Wall -Wextra)

    add_executable(megaloc_embedder_smoke megaloc_embedder_smoke.cpp)
    target_link_libraries(megaloc_embedder_smoke PRIVATE placecell::megaloc)
    target_compile_options(megaloc_embedder_smoke PRIVATE -Wall -Wextra -Wpedantic)
endif()
