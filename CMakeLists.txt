cmake_minimum_required(VERSION 3.20)

project(GpuComp2026SS_02 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PIPELINE_ENABLE_CUDA "Enable optional CUDA pipeline backends" OFF)
set(PIPELINE_CUDA_SOURCES "" CACHE STRING "Semicolon-separated CUDA source files that implement cuda/ops_cuda.hpp")
set(PIPELINE_CUDA_STAGES "" CACHE STRING "Semicolon-separated CUDA stages, e.g. PREPROCESSING;RADIAL_SIGNAL")

add_library(serial_pipeline_modules
    src/serial_cpp/preprocessing.cpp
    src/serial_cpp/cleaning.cpp
    src/serial_cpp/boundary_extraction.cpp
    src/serial_cpp/component_labeling.cpp
    src/serial_cpp/contour_to_signal.cpp
)

target_include_directories(serial_pipeline_modules
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(pipeline_core
    src/pipeline.cpp
)

target_include_directories(pipeline_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(pipeline_core
    PUBLIC
        serial_pipeline_modules
)

add_executable(pipeline
    src/pipeline.cpp
)

target_compile_definitions(pipeline
    PRIVATE
        PIPELINE_BUILD_STANDALONE
)

target_link_libraries(pipeline
    PRIVATE
        serial_pipeline_modules
)

if(PIPELINE_ENABLE_CUDA)
    enable_language(CUDA)

    if(PIPELINE_CUDA_SOURCES)
        add_library(cuda_pipeline_modules
            ${PIPELINE_CUDA_SOURCES}
        )

        target_include_directories(cuda_pipeline_modules
            PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

        target_link_libraries(pipeline_core
            PUBLIC
                cuda_pipeline_modules
        )

        target_link_libraries(pipeline
            PRIVATE
                cuda_pipeline_modules
        )
    endif()

    foreach(stage IN LISTS PIPELINE_CUDA_STAGES)
        target_compile_definitions(pipeline_core
            PUBLIC
                PIPELINE_USE_CUDA_${stage}
        )

        target_compile_definitions(pipeline
            PRIVATE
                PIPELINE_USE_CUDA_${stage}
        )
    endforeach()
endif()
