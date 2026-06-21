#pragma once

#include <cstdint>

#include "types.hpp"

// Naive serial preprocessing entry point:
// RGB -> grayscale -> normalize -> blur -> Otsu threshold -> binarize -> open.
// The result is a cleaned binary mask used by serial connected components.
void preprocess(
    const uint8_t* rgb,
    int width,
    int height,
    int ksize,
    float sigma,
    ImageU8& result
);
