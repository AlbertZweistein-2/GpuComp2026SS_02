#pragma once

// ----------------------------------------------------------------------
// STANDARD LIBRARY INCLUDES
#include <cstdint>
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// GPU SPECIFIC INCLUDES
#include <thrust/device_vector.h>
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// CUDA SCRATCH BUFFERS
struct PreprocessCudaScratch
{
    // Temporary binary buffer used by morphological opening.
    thrust::device_vector<uint8_t> temp;
    // Grayscale image after RGB conversion and normalization.
    thrust::device_vector<float> gray;
    // Gaussian-blurred grayscale image consumed by thresholding/cleaning.
    thrust::device_vector<float> blurred;
    // Device histogram used by Otsu threshold selection.
    thrust::device_vector<uint32_t> hist;
    // Single device-side Otsu threshold value.
    thrust::device_vector<float> threshold;
    // Device-side pair storing min and max grayscale values.
    thrust::device_vector<float> minmax;
};
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// PUBLIC PREPROCESSING API
// Allocate all device buffers whose size depends only on the input image.
void allocate_preprocess_cuda_scratch(
    int width,
    int height,
    PreprocessCudaScratch &scratch
);

// Convert device RGB input into a cleaned binary mask on the device.
void preprocess_cuda(
    const uint8_t* d_rgb,
    uint8_t* d_cleaned,
    int width,
    int height,
    int ksize,
    float sigma,
    PreprocessCudaScratch &scratch
);
// ----------------------------------------------------------------------
