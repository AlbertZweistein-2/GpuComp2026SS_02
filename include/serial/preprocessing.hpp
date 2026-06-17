#pragma once

#include <cstdint>

#include "types.hpp"

void preprocess(
    const uint8_t* rgb,
    int width,
    int height,
    int ksize,
    float sigma,
    int morphology_kernel_width,
    int morphology_kernel_height,
    int morphology_iterations,
    ImageU8& result
);