#pragma once

#include <cstdint>

#include "types.hpp"

void preprocess_cuda(
    const uint8_t* d_rgb,
    uint8_t* d_cleaned,
    int width,
    int height,
    int ksize,
    float sigma
);
