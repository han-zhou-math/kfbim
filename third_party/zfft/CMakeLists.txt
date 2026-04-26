add_library(zfft STATIC
    src/zfft.cc
    src/zfft_ext.cc
)

target_include_directories(zfft PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Optional OpenMP support
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(zfft PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_features(zfft PUBLIC cxx_std_17)
