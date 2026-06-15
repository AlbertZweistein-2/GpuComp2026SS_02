cmake_minimum_required(VERSION 3.20)

project(GpuComp2026SS_02 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(PIPELINE_DEBUG_OUTPUT "Compile serial pipeline debug logging and summary output" ON)
option(PIPELINE_SUB_TIMINGS "Measure and print serial pipeline sub-stage timings when debug output is enabled" ON)
option(PIPELINE_WRITE_OUTPUT_IMAGE "Write a serial pipeline overlay image with piece contours and bounding boxes" OFF)
option(PIPELINE_ENABLE_CUDA "Enable optional CUDA pipeline backends" OFF)
set(PIPELINE_OPTIMIZATION_LEVEL "-O3" CACHE STRING "Optimization flag for serial C++ Release builds, e.g. -O2 or -O3")
set(PIPELINE_CUDA_SOURCES "" CACHE STRING "Semicolon-separated CUDA sources to build when PIPELINE_ENABLE_CUDA is ON")

set(SERIAL_PIPELINE_MODULE_SOURCES
    src/serial/preprocessing.cpp
    src/serial/cleaning.cpp
    src/serial/boundary_extraction.cpp
    src/serial/component_labeling.cpp
    src/serial/contour_to_signal.cpp
    src/serial/signal_analysis.cpp
)

if(PIPELINE_WRITE_OUTPUT_IMAGE)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
    list(APPEND SERIAL_PIPELINE_MODULE_SOURCES
        src/serial/visualization.cpp
    )
endif()

add_library(serial_pipeline_modules
    ${SERIAL_PIPELINE_MODULE_SOURCES}
)

target_include_directories(serial_pipeline_modules
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(serial_pipeline_modules
    PRIVATE
        $<$<CONFIG:Release>:${PIPELINE_OPTIMIZATION_LEVEL}>
        $<$<CONFIG:RelWithDebInfo>:${PIPELINE_OPTIMIZATION_LEVEL}>
)

if(PIPELINE_WRITE_OUTPUT_IMAGE)
    target_include_directories(serial_pipeline_modules
        PUBLIC
            ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(serial_pipeline_modules
        PUBLIC
            ${OpenCV_LIBS}
    )
endif()

add_library(pipeline_core
    src/serial/pipeline.cpp
)

target_include_directories(pipeline_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(pipeline_core
    PUBLIC
        serial_pipeline_modules
)

target_compile_options(pipeline_core
    PRIVATE
        $<$<CONFIG:Release>:${PIPELINE_OPTIMIZATION_LEVEL}>
        $<$<CONFIG:RelWithDebInfo>:${PIPELINE_OPTIMIZATION_LEVEL}>
)

add_executable(pipeline
    src/serial/pipeline.cpp
)

target_compile_definitions(pipeline
    PRIVATE
        PIPELINE_BUILD_STANDALONE
)

target_link_libraries(pipeline
    PRIVATE
        serial_pipeline_modules
)

target_compile_options(pipeline
    PRIVATE
        $<$<CONFIG:Release>:${PIPELINE_OPTIMIZATION_LEVEL}>
        $<$<CONFIG:RelWithDebInfo>:${PIPELINE_OPTIMIZATION_LEVEL}>
)

foreach(target IN ITEMS pipeline_core pipeline)
    if(PIPELINE_DEBUG_OUTPUT)
        target_compile_definitions(${target}
            PRIVATE
                DEBUG_LEVEL=1
        )
    endif()

    if(PIPELINE_DEBUG_OUTPUT AND PIPELINE_SUB_TIMINGS)
        target_compile_definitions(${target}
            PRIVATE
                SUB_TIMINGS=1
        )
    endif()

    if(PIPELINE_WRITE_OUTPUT_IMAGE)
        target_compile_definitions(${target}
            PRIVATE
                PIPELINE_WRITE_OUTPUT_IMAGE=1
        )
    endif()
endforeach()

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
endif()
