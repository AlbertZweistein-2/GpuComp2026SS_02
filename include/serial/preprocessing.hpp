#pragma once

#include <cstdint>

#include "types.hpp"

void preprocess(
    const uint8_t* rgb,
    int width,
    int height,
    int ksize,
    float sigma,
    ImageU8& result
);
