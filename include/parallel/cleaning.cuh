#pragma once

#include <cstdint>

void binarize_morphological_open_cuda_device(
    const float* d_input,
    const float* d_threshold,
    uint8_t* d_temp,
    uint8_t* d_output,
    int width,
    int height
);
