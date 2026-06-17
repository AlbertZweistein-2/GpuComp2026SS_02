#pragma once

#include <cstdint>

void erode_cuda_device(
    const uint8_t* d_input,
    const uint8_t* d_kernel,
    uint8_t* d_output,
    int width,
    int height,
    int kernel_width,
    int kernel_height
);

void dilate_cuda_device(
    const uint8_t* d_input,
    const uint8_t* d_kernel,
    uint8_t* d_output,
    int width,
    int height,
    int kernel_width,
    int kernel_height
);

void morphological_open_cuda_device(
    const uint8_t* d_input,
    const uint8_t* d_kernel,
    uint8_t* d_temp,
    uint8_t* d_output,
    int width,
    int height,
    int kernel_width,
    int kernel_height,
    int iterations
);
