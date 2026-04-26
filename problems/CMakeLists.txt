# ---------------------------------------------------------------------------
# kfbim_problems — Layer 5 problem API
#
# Currently header-only.  Same transition notes as core/CMakeLists.txt.
# All problem .cpp files go here once implemented.
# ---------------------------------------------------------------------------

add_library(kfbim_problems INTERFACE)

target_include_directories(kfbim_problems INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(kfbim_problems INTERFACE kfbim_core)

target_compile_features(kfbim_problems INTERFACE cxx_std_17)
