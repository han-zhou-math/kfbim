# ---------------------------------------------------------------------------
# kfbim_core — compiled core library
#
# Add new .cpp files here as each layer is implemented.
# Eigen is PUBLIC (types appear in public headers).
# CGAL is PUBLIC for now; switch to PRIVATE once GridPair pimpl is in place.
# ---------------------------------------------------------------------------

add_library(kfbim_core STATIC
    # Layer 0 — Grid
    grid/cartesian_grid_2d.cpp
    grid/cartesian_grid_3d.cpp
    grid/mac_grid_2d.cpp
    grid/mac_grid_3d.cpp

    # Layer 0 — Interface
    interface/interface_2d.cpp
    interface/interface_3d.cpp

    # Layer 0 — Geometry
    geometry/grid_pair_2d.cpp
    geometry/grid_pair_3d.cpp

    # Layer 1.5 — Local Cauchy Solver (add when implemented)
    # local_cauchy/laplace_local_solver_2d.cpp

    # Layer 1 — Transfer Operators (add when implemented)
    transfer/laplace_spread_2d.cpp
    transfer/laplace_restrict_2d.cpp

    # Layer 2 — Bulk Solver
    solver/laplace_fft_bulk_solver_2d.cpp
    solver/zfft_engine_2d.cpp
    solver/laplace_zfft_bulk_solver_2d.cpp
    solver/laplace_zfft_bulk_solver_3d.cpp

    # Layer 3 — KFBI Operator
    operator/laplace_kfbi_operator.cpp
    operator/laplace_potential.cpp

    # Layer 3 utility — Interface solver pipeline
    problems/laplace_interface_solver_2d.cpp

    # Layer 4 — Outer Solver
    gmres/gmres.cpp
)

target_include_directories(kfbim_core PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(kfbim_core PUBLIC Eigen3::Eigen zfft)

if(CGAL_FOUND)
    target_link_libraries(kfbim_core PUBLIC CGAL::CGAL)
    target_compile_definitions(kfbim_core PUBLIC KFBIM_HAS_CGAL)
endif()

if(FFTW3_FOUND)
    target_sources(kfbim_core PRIVATE solver/fftw3_engine.cpp)
    # Library is PRIVATE (only kfbim_core links against libfftw3); include dir is
    # PUBLIC so callers that #include "fftw3_engine.hpp" can find fftw3.h.
    target_link_libraries(kfbim_core PRIVATE FFTW3::FFTW3)
    target_include_directories(kfbim_core PUBLIC "${FFTW3_INC}")
    target_compile_definitions(kfbim_core PUBLIC KFBIM_HAS_FFTW3)
endif()

target_compile_features(kfbim_core PUBLIC cxx_std_17)
