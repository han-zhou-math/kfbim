cmake_minimum_required(VERSION 3.20)
project(kfbim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# Eigen3: header-only, fetched automatically if not installed.
# To install manually: brew install eigen
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found locally — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING       OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen3)
endif()

# CGAL: required only by core/geometry (GridPair).  pimpl keeps CGAL headers
# out of the public API so no other module needs to link it.
# Install: brew install cgal
# CGAL is optional during early development; geometry tests are skipped when absent.
find_package(CGAL QUIET)
if(NOT CGAL_FOUND)
    message(STATUS "CGAL not found — core/geometry (GridPair) will not be built. "
                   "Install with: brew install cgal")
endif()

# FFTW3: required for LaplaceFftBulkSolver.
# Install: brew install fftw
find_library(FFTW3_LIB  NAMES fftw3  HINTS /opt/homebrew/lib  /usr/local/lib)
find_path   (FFTW3_INC  NAMES fftw3.h HINTS /opt/homebrew/include /usr/local/include)
if(FFTW3_LIB AND FFTW3_INC)
    message(STATUS "Found FFTW3: ${FFTW3_LIB}")
    add_library(FFTW3::FFTW3 UNKNOWN IMPORTED)
    set_target_properties(FFTW3::FFTW3 PROPERTIES
        IMPORTED_LOCATION             "${FFTW3_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INC}")
    set(FFTW3_FOUND TRUE)
else()
    message(STATUS "FFTW3 not found — FFT bulk solver will not be built. "
                   "Install with: brew install fftw")
    set(FFTW3_FOUND FALSE)
endif()

# ---------------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------------
add_subdirectory(third_party/zfft)

# ---------------------------------------------------------------------------
# Library targets (dependency order)
# ---------------------------------------------------------------------------
add_subdirectory(core)
add_subdirectory(problems)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
option(KFBIM_BUILD_TESTS "Build tests" ON)
if(KFBIM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
