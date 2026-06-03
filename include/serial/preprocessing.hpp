#pragma once

#include <cstdint>

#include "types.hpp"

ImageU8 preprocess(
    const uint8_t* rgb,
    int width,
    int height,
    int ksize = 5,
    float sigma = 1.0f
);
