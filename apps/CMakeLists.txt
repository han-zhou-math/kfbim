add_executable(shape_opt_transmission_2d
    shape_opt_transmission_2d.cpp
)

add_executable(transmission_center_perturb_2d
    transmission_center_perturb_2d.cpp
)

target_link_libraries(shape_opt_transmission_2d PRIVATE
    kfbim_core
)

target_link_libraries(transmission_center_perturb_2d PRIVATE
    kfbim_core
)

target_compile_definitions(shape_opt_transmission_2d PRIVATE
    KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
)

target_compile_definitions(transmission_center_perturb_2d PRIVATE
    KFBIM_APP_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/output"
)

target_compile_features(shape_opt_transmission_2d PRIVATE cxx_std_17)
target_compile_features(transmission_center_perturb_2d PRIVATE cxx_std_17)
