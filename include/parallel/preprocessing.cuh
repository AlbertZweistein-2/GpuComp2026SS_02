#pragma once

#include <cstdint>

#include <thrust/device_vector.h>

#include "types.hpp"

struct PreprocessCudaScratch
{
    thrust::device_vector<uint8_t> temp;
    thrust::device_vector<float> gray;
    thrust::device_vector<float> blurred;
    thrust::device_vector<uint32_t> hist;
    thrust::device_vector<float> threshold;
    thrust::device_vector<float> minmax;
};

void allocate_preprocess_cuda_scratch(
    int width,
    int height,
    PreprocessCudaScratch &scratch
);

void preprocess_cuda(
    const uint8_t* d_rgb,
    uint8_t* d_cleaned,
    int width,
    int height,
    int ksize,
    float sigma,
    PreprocessCudaScratch &scratch
);
