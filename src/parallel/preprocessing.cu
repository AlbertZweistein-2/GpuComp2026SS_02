/**
 * CUDA C++ — Step 1: Image Preprocessing
 * TU Wien — GPU Architecture & Computing SS2026
 *
 * GPU preprocessing implementation:
 *   1. RGB -> Grayscale      (ITU-R BT.601)
 *   2. Normalization         (Min/Max -> [0, 255])
 *   3. Gaussian Blur         (2-D convolution with separable kernel)
 *   4. Histogram + Otsu      (automatic threshold computation)
 *   5. Binarization
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include "parallel/cleaning.cuh"
#include "parallel/preprocessing.cuh"

#include "helpers.hpp"

// ─── Constants (same as the CUDA implementation) ─────────────────────────────
static constexpr int MAX_KERNEL = 15;
static constexpr int HIST_BINS = 256;

// Gaussian kernel will be allocated in global memory
// CHANGE: was returned to const due to performance reasons

__constant__ float d_gaussian_kernel[MAX_KERNEL * MAX_KERNEL];

// FOR CUDA Impl.
//  -----------------------------------------------------------------------------
// STEP 1 — RGB -> Grayscale
//
//  Formula: Y = 0.299·R + 0.587·G + 0.114·B  (ITU-R BT.601)
//  -----------------------------------------------------------------------------

__global__ void rgb_to_grayscale_kernel(
    const uint8_t *rgb,
    float *gray,
    int width,
    int height)
{
    int num_pixels = width * height;
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = pixel_idx; idx < num_pixels; idx += stride)
    {
        int rgb_idx = idx * 3;

        float r = static_cast<float>(rgb[rgb_idx]);
        float g = static_cast<float>(rgb[rgb_idx + 1]);
        float b = static_cast<float>(rgb[rgb_idx + 2]);

        gray[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

// FOR CUDA Impl.
//  -----------------------------------------------------------------------------
//  STEP 2 — Normalization
//
//  Scales all pixel values linearly to [0, 255].
//  -----------------------------------------------------------------------------
__global__ void copy_minmax_kernel(
    const float *min_ptr,
    const float *max_ptr,
    float *minmax)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        minmax[0] = *min_ptr;
        minmax[1] = *max_ptr;
    }
}

__global__ void normalize_kernel(
    float *gray,
    const float *minmax,
    int num_pixels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float gmin = minmax[0];
    float gmax = minmax[1];
    float range = gmax - gmin;

    if (range < 1e-5f)
        range = 1.0f;

    for (int idx = i; idx < num_pixels; idx += stride)
    {
        gray[idx] = (gray[idx] - gmin) / range * 255.0f;
    }
}

// -----------------------------------------------------------------------------
// STEP 3a — Create Gaussian kernel
//
// Produce a normalized (ksize × ksize) Gaussian kernel.
// -----------------------------------------------------------------------------
static std::vector<float> make_gaussian_kernel(int ksize, float sigma)
{
    std::vector<float> kernel(ksize * ksize);
    int half = ksize / 2;
    float sum = 0.f;

    for (int y = -half; y <= half; ++y)
    {
        for (int x = -half; x <= half; ++x)
        {
            float v = std::exp(-(x * x + y * y) / (2.f * sigma * sigma));
            kernel[(y + half) * ksize + (x + half)] = v;
            sum += v;
        }
    }

    // Normalize: sum of all weights = 1
    for (float &v : kernel)
        v /= sum;

    return kernel;
}

// FOR CUDA Impl.
//  -----------------------------------------------------------------------------
//  STEP 3b — Gaussian Blur (2-D convolution)
//
//  -----------------------------------------------------------------------------

__global__ void gaussian_blur_kernel(
    const float *in,
    float *out,
    int width,
    int height,
    int ksize)
{
    int num_pixels = width * height;
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int half = ksize / 2;

    for (int idx = pixel_idx; idx < num_pixels; idx += stride)
    {
        int y = idx / width;
        int x = idx % width;

        float sum = 0.0f;

        for (int ky = 0; ky < ksize; ++ky)
        {
            for (int kx = 0; kx < ksize; ++kx)
            {
                int sy = y - half + ky;
                int sx = x - half + kx;

                sy = max(0, min(sy, height - 1));
                sx = max(0, min(sx, width - 1));

                float pixel = in[sy * width + sx];
                float weight = d_gaussian_kernel[ky * ksize + kx];

                sum += pixel * weight;
            }
        }

        out[idx] = sum;
    }
}

// -----------------------------------------------------------------------------
// FOR CUDA Impl.
// STEP 4a — Compute histogram
// -----------------------------------------------------------------------------
__global__ void histogram_kernel(
    const float *gray,
    uint32_t *hist,
    int num_pixels)
{
    __shared__ uint32_t local_hist[HIST_BINS];

    for (int i = threadIdx.x; i < HIST_BINS; i += blockDim.x)
        local_hist[i] = 0;

    __syncthreads();

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = global_id; i < num_pixels; i += stride)
    {
        int bin = static_cast<int>(gray[i]);
        if (bin < 0)
            bin = 0;
        if (bin > 255)
            bin = 255;

        atomicAdd(&local_hist[bin], 1u);
    }

    __syncthreads();

    for (int i = threadIdx.x; i < HIST_BINS; i += blockDim.x)
        atomicAdd(&hist[i], local_hist[i]);
}

// -----------------------------------------------------------------------------
// FOR CUDA Impl.
// STEP 4b — Otsu threshold
// -----------------------------------------------------------------------------
__global__ void otsu_threshold_kernel(
    const uint32_t *hist,
    int total_pixels,
    float *threshold_out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    double sum_total = 0.0;
    for (int t = 0; t < HIST_BINS; ++t)
        sum_total += static_cast<double>(t) * hist[t];

    double sum_bg = 0.0;
    double w_bg = 0.0;
    double max_var = 0.0;
    int threshold = 0;

    for (int t = 0; t < HIST_BINS; ++t)
    {
        w_bg += hist[t];
        if (w_bg == 0.0)
            continue;

        double w_fg = static_cast<double>(total_pixels) - w_bg;
        if (w_fg == 0.0)
            break;

        sum_bg += static_cast<double>(t) * hist[t];
        double mean_bg = sum_bg / w_bg;
        double mean_fg = (sum_total - sum_bg) / w_fg;

        double diff = mean_bg - mean_fg;
        double var = w_bg * w_fg * diff * diff;

        if (var > max_var)
        {
            max_var = var;
            threshold = t;
        }
    }

    *threshold_out = static_cast<float>(threshold);
}

// -----------------------------------------------------------------------------
// Preprocessing stages only. Cleaning is in parallel/cleaning.cuh.
// -----------------------------------------------------------------------------

void allocate_preprocess_cuda_scratch(
    int width,
    int height,
    PreprocessCudaScratch &scratch)
{
    const std::size_t num_pixels = static_cast<std::size_t>(width) * height;

    scratch.temp.resize(num_pixels);
    scratch.gray.resize(num_pixels);
    scratch.blurred.resize(num_pixels);
    scratch.hist.resize(HIST_BINS);
    scratch.threshold.resize(1);
    scratch.minmax.resize(2);
}

void preprocess_cuda(
    const uint8_t *d_rgb_ptr,
    uint8_t *d_cleaned_ptr,
    int width,
    int height,
    int ksize,
    float sigma,
    PreprocessCudaScratch &scratch)
{
    const int num_pixels = width * height;

    std::vector<float> kernel = make_gaussian_kernel(ksize, sigma);

    uint8_t *d_temp = thrust::raw_pointer_cast(scratch.temp.data());
    float *d_gray = thrust::raw_pointer_cast(scratch.gray.data());
    float *d_blurred = thrust::raw_pointer_cast(scratch.blurred.data());
    uint32_t *d_hist = thrust::raw_pointer_cast(scratch.hist.data());
    float *d_threshold = thrust::raw_pointer_cast(scratch.threshold.data());
    float *d_minmax = thrust::raw_pointer_cast(scratch.minmax.data());

    CUDA_CHECK(cudaMemcpyToSymbol(
        d_gaussian_kernel,
        kernel.data(),
        kernel.size() * sizeof(float)));

    int block1d = 256;
    int grid1d = (num_pixels + block1d - 1) / block1d;

    rgb_to_grayscale_kernel<<<grid1d, block1d>>>(
        d_rgb_ptr,
        d_gray,
        width,
        height);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<float> gray_begin(d_gray);
    thrust::device_ptr<float> gray_end(d_gray + num_pixels);

    auto minmax_pair = thrust::minmax_element(
        thrust::device,
        gray_begin,
        gray_end);

    copy_minmax_kernel<<<1, 1>>>(
        thrust::raw_pointer_cast(minmax_pair.first),
        thrust::raw_pointer_cast(minmax_pair.second),
        d_minmax);
    CUDA_CHECK(cudaGetLastError());

    normalize_kernel<<<grid1d, block1d>>>(
        d_gray,
        d_minmax,
        num_pixels);
    CUDA_CHECK(cudaGetLastError());

    gaussian_blur_kernel<<<grid1d, block1d>>>(
        d_gray,
        d_blurred,
        width,
        height,
        ksize);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemset(d_hist, 0, HIST_BINS * sizeof(uint32_t)));

    int hist_grid = std::min(grid1d, 1024);

    histogram_kernel<<<hist_grid, block1d>>>(
        d_blurred,
        d_hist,
        num_pixels);
    CUDA_CHECK(cudaGetLastError());

    otsu_threshold_kernel<<<1, 1>>>(
        d_hist,
        num_pixels,
        d_threshold);
    CUDA_CHECK(cudaGetLastError());

    binarize_morphological_open_cuda_device(
        d_blurred,
        d_threshold,
        d_temp,
        d_cleaned_ptr,
        width,
        height);
}
