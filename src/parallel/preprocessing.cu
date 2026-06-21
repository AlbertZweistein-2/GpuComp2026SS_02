// ----------------------------------------------------------------------
// HEADER INCLUDES
#include "parallel/cleaning.cuh"
#include "parallel/preprocessing.cuh"
#include "helpers.hpp"
// ----------------------------------------------------------------------
// STANDARD LIBRARY INCLUDES
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
// ----------------------------------------------------------------------
// GPU SPECIFIC INCLUDES
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// PREPROCESSING CONSTANTS
static constexpr int MAX_KERNEL = 15;
static constexpr int HIST_BINS = 256;

// Gaussian weights live in constant memory because every thread reads the same
// small kernel during blur.
__constant__ float d_gaussian_kernel[MAX_KERNEL * MAX_KERNEL];
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// STEP 1: RGB -> grayscale
// Convert RGB pixels to luminance using ITU-R BT.601 coefficients.
// The grid-stride loop lets a fixed launch geometry cover large images without
// requiring one resident thread per pixel.
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
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// STEP 2: Normalization
// Copy the min/max values found by Thrust into a compact device buffer so the
// normalization kernel can read them without host synchronization.
// The min/max reduction itself is delegated to Thrust below, avoiding a custom
// reduction kernel and keeping the full normalization path on the device.
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

// Scale all grayscale pixel values linearly to [0, 255].
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
// ----------------------------------------------------------------------

// -----------------------------------------------------------------------------
// STEP 3a: Create Gaussian kernel
// Produce a normalized ksize x ksize Gaussian kernel on the host.
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

// ----------------------------------------------------------------------
// STEP 3b: Gaussian blur
// Apply the Gaussian kernel from constant memory. Border pixels clamp to the
// nearest valid image coordinate.
// Constant memory is a good fit here because all threads repeatedly read the
// same small coefficient table, reducing redundant global-memory traffic.
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
// ----------------------------------------------------------------------

// -----------------------------------------------------------------------------
// STEP 4a: Compute histogram
// Build a 256-bin image histogram using one shared-memory histogram per block,
// then atomically merge each block histogram into the global histogram.
// This reduces global atomic contention: most pixel updates hit fast per-block
// shared memory, and only 256 bins per block are merged globally.
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
// STEP 4b: Otsu threshold
// Compute the threshold that maximizes between-class variance. This work is
// small enough that a single device thread avoids an extra device-to-host copy.
// Keeping this on the GPU also keeps the threshold value device-resident for the
// fused binarize/erode cleaning kernel.
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
// CUDA SCRATCH ALLOCATION
// Allocate reusable device buffers for all preprocessing stages. The pipeline
// calls this once during CUDA setup and reuses the buffers in preprocess_cuda.
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

// -----------------------------------------------------------------------------
// MAIN PREPROCESSING FUNCTION
// Run the full preprocessing path on already-allocated device buffers:
// RGB -> grayscale -> normalize -> blur -> histogram/Otsu -> morphological open.
// -----------------------------------------------------------------------------
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

    // Create and upload the Gaussian kernel for this preprocessing run.
    std::vector<float> kernel = make_gaussian_kernel(ksize, sigma);

    // Resolve raw device pointers from the preallocated scratch vectors.
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

    // Most kernels use a 1D grid over all image pixels.
    int block1d = 256;
    int grid1d = (num_pixels + block1d - 1) / block1d;

    // STEP 1: RGB -> grayscale.
    rgb_to_grayscale_kernel<<<grid1d, block1d>>>(
        d_rgb_ptr,
        d_gray,
        width,
        height);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<float> gray_begin(d_gray);
    thrust::device_ptr<float> gray_end(d_gray + num_pixels);

    // STEP 2a: Find min/max grayscale values with Thrust's device reduction.
    auto minmax_pair = thrust::minmax_element(
        thrust::device,
        gray_begin,
        gray_end);

    // STEP 2b: Store min/max values on device for normalization.
    copy_minmax_kernel<<<1, 1>>>(
        thrust::raw_pointer_cast(minmax_pair.first),
        thrust::raw_pointer_cast(minmax_pair.second),
        d_minmax);
    CUDA_CHECK(cudaGetLastError());

    // STEP 2c: Normalize grayscale values to [0, 255].
    normalize_kernel<<<grid1d, block1d>>>(
        d_gray,
        d_minmax,
        num_pixels);
    CUDA_CHECK(cudaGetLastError());

    // STEP 3: Gaussian blur.
    gaussian_blur_kernel<<<grid1d, block1d>>>(
        d_gray,
        d_blurred,
        width,
        height,
        ksize);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemset(d_hist, 0, HIST_BINS * sizeof(uint32_t)));

    // STEP 4a: Histogram. Cap the grid because more blocks only increase
    // global merge atomics after a point.
    int hist_grid = std::min(grid1d, 1024);

    histogram_kernel<<<hist_grid, block1d>>>(
        d_blurred,
        d_hist,
        num_pixels);
    CUDA_CHECK(cudaGetLastError());

    // STEP 4b: Otsu threshold selection.
    otsu_threshold_kernel<<<1, 1>>>(
        d_hist,
        num_pixels,
        d_threshold);
    CUDA_CHECK(cudaGetLastError());

    // STEP 5: Binarize and clean using morphological opening.
    binarize_morphological_open_cuda_device(
        d_blurred,
        d_threshold,
        d_temp,
        d_cleaned_ptr,
        width,
        height);
}
