#pragma once

// ----------------------------------------------------------------------
// STANDARD LIBRARY INCLUDES
#include <cstdint>
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// PUBLIC CLEANING API
// Threshold the blurred grayscale image, erode it, then dilate it again.
// `d_temp` stores the intermediate eroded binary mask.
void binarize_morphological_open_cuda_device(
    const float* d_input,
    const float* d_threshold,
    uint8_t* d_temp,
    uint8_t* d_output,
    int width,
    int height
);
// ----------------------------------------------------------------------
