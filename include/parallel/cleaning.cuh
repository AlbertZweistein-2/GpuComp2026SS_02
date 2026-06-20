#pragma once

#include <cstdint>

void morphological_open_cuda_device(
    const uint8_t* d_input,
    uint8_t* d_temp,
    uint8_t* d_output,
    int width,
    int height
);
