#pragma once

#include <cstdint>

#include "types.hpp"

void preprocess_cuda(
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

void preprocess_cuda_device(
    const uint8_t* rgb,
    uint8_t* d_cleaned,
    int width,
    int height,
    int ksize,
    float sigma,
    int morphology_kernel_width,
    int morphology_kernel_height,
    int morphology_iterations
);
