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

# ---------------------------------------------------------------------------
# kfbim_add_test(<target> <sources...>)
#
# Creates a test executable, links it against kfbim_core + Catch2, and
# registers all TEST_CASEs for CTest discovery.
# ---------------------------------------------------------------------------
function(kfbim_add_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE
        kfbim_core
        Catch2::Catch2WithMain
    )
    target_compile_definitions(${target} PRIVATE
        KFBIM_TEST_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
    )
    catch_discover_tests(${target})
endfunction()

# ---------------------------------------------------------------------------
# Active PDE-level convergence programs.
# Component and legacy tests live in tests/archive/ and are not registered here.
# ---------------------------------------------------------------------------

kfbim_add_test(test_interface test_interface.cpp)
kfbim_add_test(test_interface_3d test_interface_3d.cpp)
kfbim_add_test(test_projection_2d test_projection_2d.cpp)
kfbim_add_test(test_projection_3d test_projection_3d.cpp)
kfbim_add_test(test_refactor_utilities test_refactor_utilities.cpp)
kfbim_add_test(test_bvp test_bvp.cpp)
kfbim_add_test(test_bvp_3d test_bvp_3d.cpp)
kfbim_add_test(test_transmission test_transmission.cpp)
kfbim_add_test(test_transmission_periodic_2d test_transmission_periodic_2d.cpp)
kfbim_add_test(test_transmission_3d test_transmission_3d.cpp)
kfbim_add_test(test_transmission_ellipsoid_3d test_transmission_ellipsoid_3d.cpp)
kfbim_add_test(test_transmission_torus_3d test_transmission_torus_3d.cpp)
