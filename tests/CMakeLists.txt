# ---------------------------------------------------------------------------
# Test infrastructure
#
# Uses Catch2 v3.  Prefers a system-installed copy; falls back to FetchContent.
# ---------------------------------------------------------------------------

find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
    message(STATUS "Catch2 not found on system — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.3
    )
    FetchContent_MakeAvailable(Catch2)
endif()

include(CTest)
include(Catch)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ---------------------------------------------------------------------------
# kfbim_add_test(<target> <sources...>)
#
# Creates a test executable, links it against kfbim_core + kfbim_problems +
# Catch2, and registers all TEST_CASEs for CTest discovery.
# ---------------------------------------------------------------------------
function(kfbim_add_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE
        kfbim_core
        kfbim_problems
        Catch2::Catch2WithMain
    )
    target_compile_definitions(${target} PRIVATE
        KFBIM_TEST_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
        KFBIM_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
        KFBIM_SCREENED_VIS_SCRIPT="${PROJECT_SOURCE_DIR}/scripts/visualize_screened_star3.py"
        KFBIM_TRANSMISSION_VIS_SCRIPT="${PROJECT_SOURCE_DIR}/scripts/visualize_transmission_constant_ratio_2d.py"
    )
    catch_discover_tests(${target})
endfunction()

# ---------------------------------------------------------------------------
# Test executables
# Each file focuses on one layer or component; add new ones as they are built.
# ---------------------------------------------------------------------------

# Layer 0 utilities (header-only inline functions): always compilable.
kfbim_add_test(test_core test_core.cpp)

kfbim_add_test(test_grid test_grid.cpp)

# Layer 2 bulk solver (FFT)
kfbim_add_test(test_solver test_solver.cpp)

# Layer 0 geometry (GridPair)
kfbim_add_test(test_grid_pair test_grid_pair.cpp)

# IIM defect correction + FFT solve (Layer 0 + 2 integration)
kfbim_add_test(test_iim_2d test_iim_2d.cpp)

# Modular Layer 3 potential evaluator
kfbim_add_test(test_laplace_potential_2d test_laplace_potential_2d.cpp)

# Full pipeline — interior Dirichlet, Chebyshev-Lobatto panels.
kfbim_add_test(test_laplace_interior_2d test_laplace_interior_2d.cpp)

# Full pipeline — screened Poisson interior Dirichlet, 3-fold star
kfbim_add_test(test_laplace_interior_screened_2d test_laplace_interior_screened_2d.cpp)

# Full pipeline — constant-ratio discontinuous coefficient transmission problem
kfbim_add_test(test_laplace_transmission_constant_ratio_2d
               test_laplace_transmission_constant_ratio_2d.cpp)
