/**
 * CUDA C++ — Step 1: Image Preprocessing
 * TU Wien — GPU Architecture & Computing SS2026
 *
 * Full GPU pipeline implementation:
 *   1. RGB -> Grayscale      (ITU-R BT.601)
 *   2. Normalization         (Min/Max -> [0, 255])
 *   3. Gaussian Blur         (2-D convolution with separable kernel)
 *   4. Histogram + Otsu      (automatic threshold computation)
 *   5. Binarization
 *
 * Compile:
 *   nvcc -O2 -std=c++17 -I. -o preprocessing_cuda preprocessing.cu
 *
 * Usage:
 *   ./preprocessing_cuda                        # loads ../../data/single.JPG
 *   ./preprocessing_cuda ../../data/1_p1.jpg   # any image
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

// stb_image - single-header image loader (place stb_image.h in the same folder)
#define STB_IMAGE_IMPLEMENTATION
#include "../../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../include/stb_image_write.h"

#include "../../include/serial/preprocessing.hpp"
#include "../../include/helpers.hpp"
// ─── Constants (same as the CUDA implementation) ─────────────────────────────
static constexpr int MAX_KERNEL = 15;
static constexpr int HIST_BINS  = 256;



// -----------------------------------------------------------------------------
// CUDA helper
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(                                        \
                std::string("CUDA error: ") + cudaGetErrorString(err) +       \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
        }                                                                    \
    } while (0)

//FOR CUDA KERNELS 
//Gaussian kernel will be allocated in global memory
//CHANGE: was returned to const due to performance reasons

__constant__ float d_gaussian_kernel[MAX_KERNEL * MAX_KERNEL];


//FOR CUDA Impl. 
// -----------------------------------------------------------------------------
//STEP 1 — RGB -> Grayscale  
//
// Formula: Y = 0.299·R + 0.587·G + 0.114·B  (ITU-R BT.601)
// -----------------------------------------------------------------------------

__global__ void rgb_to_grayscale_kernel(
    const uint8_t* rgb,
    float* gray,
    int width,
    int height)
{
    int num_pixels = width * height;
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = pixel_idx; idx < num_pixels; idx += stride) {
        int rgb_idx = idx * 3;

        float r = static_cast<float>(rgb[rgb_idx]);
        float g = static_cast<float>(rgb[rgb_idx + 1]);
        float b = static_cast<float>(rgb[rgb_idx + 2]);

        gray[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}




//FOR CUDA Impl. 
// -----------------------------------------------------------------------------
// STEP 2 — Normalization
//
// Scales all pixel values linearly to [0, 255].
// -----------------------------------------------------------------------------
__global__ void copy_minmax_kernel(
    const float* min_ptr,
    const float* max_ptr,
    float* minmax)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        minmax[0] = *min_ptr;
        minmax[1] = *max_ptr;
    }
}

__global__ void normalize_kernel(
    float* gray,
    const float* minmax,
    int num_pixels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float gmin = minmax[0];
    float gmax = minmax[1];
    float range = gmax - gmin;

    if (range < 1e-5f)
        range = 1.0f;

    for (int idx = i; idx < num_pixels; idx += stride) {
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
    if (ksize < 1 || ksize % 2 == 0 || ksize > MAX_KERNEL)
        throw std::invalid_argument(
            "ksize must be odd and <= " + std::to_string(MAX_KERNEL));
    if (sigma <= 0.f)
        throw std::invalid_argument("sigma must be > 0");

    std::vector<float> kernel(ksize * ksize);
    int half = ksize / 2;
    float sum = 0.f;

    for (int y = -half; y <= half; ++y) {
        for (int x = -half; x <= half; ++x) {
            float v = std::exp(-(x*x + y*y) / (2.f * sigma * sigma));
            kernel[(y + half) * ksize + (x + half)] = v;
            sum += v;
        }
    }

    // Normalize: sum of all weights = 1
    for (float& v : kernel)
        v /= sum;

    return kernel;
}


//FOR CUDA Impl. 
// -----------------------------------------------------------------------------
// STEP 3b — Gaussian Blur (2-D convolution)
//
// -----------------------------------------------------------------------------

__global__ void gaussian_blur_kernel(
    const float* in,
    float* out,
    int width,
    int height,
    int ksize)
{
    int num_pixels = width * height;
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int half = ksize / 2;

    for (int idx = pixel_idx; idx < num_pixels; idx += stride) {
        int y = idx / width;
        int x = idx % width;

        float sum = 0.0f;

        for (int ky = 0; ky < ksize; ++ky) {
            for (int kx = 0; kx < ksize; ++kx) {
                int sy = y - half + ky;
                int sx = x - half + kx;

                sy = max(0, min(sy, height - 1));
                sx = max(0, min(sx, width - 1));

                float pixel  = in[sy * width + sx];
                float weight = d_gaussian_kernel[ky * ksize + kx];

                sum += pixel * weight;
            }
        }

        out[idx] = sum;
    }
}


// -----------------------------------------------------------------------------
//FOR CUDA Impl.
//STEP 4a — Compute histogram
// -----------------------------------------------------------------------------
__global__ void histogram_kernel(
    const float* gray,
    uint32_t* hist,
    int num_pixels)
{
    __shared__ uint32_t local_hist[HIST_BINS];

    for (int i = threadIdx.x; i < HIST_BINS; i += blockDim.x)
        local_hist[i] = 0;

    __syncthreads();

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = global_id; i < num_pixels; i += stride) {
        int bin = static_cast<int>(gray[i]);
        if (bin < 0) bin = 0;
        if (bin > 255) bin = 255;

        atomicAdd(&local_hist[bin], 1u);
    }

    __syncthreads();

    for (int i = threadIdx.x; i < HIST_BINS; i += blockDim.x)
        atomicAdd(&hist[i], local_hist[i]);
}




// -----------------------------------------------------------------------------
//FOR CUDA Impl. 
//STEP 4b — Otsu threshold
// -----------------------------------------------------------------------------
__global__ void otsu_threshold_kernel(
    const uint32_t* hist,
    int total_pixels,
    float* threshold_out)
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

    for (int t = 0; t < HIST_BINS; ++t) {
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

        if (var > max_var) {
            max_var = var;
            threshold = t;
        }
    }

    *threshold_out = static_cast<float>(threshold);
}


// -----------------------------------------------------------------------------
// FOR CUDA Impl.
// STEP 5 — Binarization
// -----------------------------------------------------------------------------
__global__ void binarize_kernel(
    const float* gray,
    const float* threshold,
    uint8_t* binary,
    int num_pixels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = i; idx < num_pixels; idx += stride) {
        binary[idx] = (gray[idx] > *threshold) ? 255u : 0u;
    }
}


// -----------------------------------------------------------------------------
// Main pipeline — combines all steps
// -----------------------------------------------------------------------------

ImageU8 preprocess_CUDA(
        const uint8_t* rgb,
        int width,
        int height,
        int ksize,
        float sigma)
{
    if (width <= 0 || height <= 0)
        throw std::invalid_argument("width and height must be > 0");
    if (ksize < 1 || ksize % 2 == 0 || ksize > MAX_KERNEL)
        throw std::invalid_argument("ksize must be odd and <= 15");

    ImageU8 result;
    result.width = width;
    result.height = height;

    int num_pixels = width * height;
    size_t rgb_bytes = static_cast<size_t>(num_pixels) * 3 * sizeof(uint8_t);
    size_t float_bytes = static_cast<size_t>(num_pixels) * sizeof(float);
    size_t binary_bytes = static_cast<size_t>(num_pixels) * sizeof(uint8_t);

    result.data.resize(static_cast<size_t>(num_pixels));

    std::vector<float> kernel = make_gaussian_kernel(ksize, sigma);

    uint8_t* d_rgb = nullptr;
    float* d_gray = nullptr;
    float* d_blurred = nullptr;
    uint8_t* d_binary = nullptr;
    uint32_t* d_hist = nullptr;
    float* d_threshold = nullptr;
    float* d_minmax = nullptr;

    CUDA_CHECK(cudaMalloc(&d_rgb, rgb_bytes));
    CUDA_CHECK(cudaMalloc(&d_gray, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_blurred, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_binary, binary_bytes));
    CUDA_CHECK(cudaMalloc(&d_hist, HIST_BINS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_threshold, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_minmax, 2 * sizeof(float)));
    
    

    CUDA_CHECK(cudaMemcpy(d_rgb, rgb, rgb_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(
        d_gaussian_kernel,
        kernel.data(),
        kernel.size() * sizeof(float)));

    int block1d = 256;
    int grid1d = (num_pixels + block1d - 1) / block1d;
    //int grid1d = 1024;

    // 1 — RGB -> Grayscale
    rgb_to_grayscale_kernel<<<grid1d, block1d>>>(
        d_rgb,
        d_gray,
        width,
        height);
    CUDA_CHECK(cudaGetLastError());

    // 2a — Min/Max on GPU using Thrust
    thrust::device_ptr<float> gray_begin(d_gray);
    thrust::device_ptr<float> gray_end(d_gray + num_pixels);
    auto minmax_pair = thrust::minmax_element(
        thrust::device,
        gray_begin,
        gray_end);

    // Important:
    // minmax_pair points into d_gray. We copy the two values to a separate GPU
    // buffer before normalizing d_gray in-place.
    copy_minmax_kernel<<<1, 1>>>(
        thrust::raw_pointer_cast(minmax_pair.first),
        thrust::raw_pointer_cast(minmax_pair.second),
        d_minmax);
    CUDA_CHECK(cudaGetLastError());

    // 2b — Normalize
    normalize_kernel<<<grid1d, block1d>>>(
        d_gray,
        d_minmax,
        num_pixels);
    CUDA_CHECK(cudaGetLastError());

    // 3 — Gaussian Blur
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    gaussian_blur_kernel<<<grid1d, block1d>>>(
        d_gray,
        d_blurred,
        width,
        height,
        ksize);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "Gaussian Blur Time: " << milliseconds << " ms\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 4a — Histogram
    CUDA_CHECK(cudaMemset(d_hist, 0, HIST_BINS * sizeof(uint32_t)));

    int hist_grid = std::min(grid1d, 1024);
    histogram_kernel<<<hist_grid, block1d>>>(
        d_blurred,
        d_hist,
        num_pixels);
    CUDA_CHECK(cudaGetLastError());

    // 4b — Otsu threshold
    otsu_threshold_kernel<<<1, 1>>>(
        d_hist,
        num_pixels,
        d_threshold);
    CUDA_CHECK(cudaGetLastError());

    // 5 — Binarization
    binarize_kernel<<<grid1d, block1d>>>(
        d_blurred,
        d_threshold,
        d_binary,
        num_pixels);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(
        result.data.data(),
        d_binary,
        binary_bytes,
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_binary));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaFree(d_threshold));
    CUDA_CHECK(cudaFree(d_minmax));


    return result;
}





//TEST for Debugging


// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    const char* input_path  = (argc > 1) ? argv[1] : "../../data/single.JPG";
    // Build output path by prefixing the input filename with "binary_" and using .png
    std::string input_str = input_path;
    std::string output_path;
    {
        size_t pos = input_str.find_last_of("/\\");
        std::string dir = (pos == std::string::npos) ? std::string() : input_str.substr(0, pos + 1);
        std::string filename = (pos == std::string::npos) ? input_str : input_str.substr(pos + 1);
        size_t dot = filename.find_last_of('.');
        std::string base = (dot == std::string::npos) ? filename : filename.substr(0, dot);
        output_path = dir + "binary_" + base + ".png";
    }

    // Load image
    int W, H, channels;
    uint8_t* rgb = stbi_load(input_path, &W, &H, &channels, 3);
    if (!rgb) {
        std::cerr << "Error: Could not load image: " << input_path
                  << "\n" << stbi_failure_reason() << "\n";
        return EXIT_FAILURE;
    }
    std::cout << "Loaded image: " << input_path
              << "  (" << W << " x " << H << ")\n";



    
    // CUDA preprocessing
    auto res = preprocess_CUDA(rgb, W, H, 15, 1.0f);

    stbi_image_free(rgb);

    // Statistik
    int white = 0;
    for (uint8_t v : res.data) white += (v == 255);
    int total = W * H;

    std::cout << "\n-- Result ---------------------------------\n";
    std::cout << "Foreground        : " << white
              << " pixels  (" << 100.f * white / total << " %)\n";
    std::cout << "Background        : " << (total - white)
              << " pixels  (" << 100.f * (total - white) / total << " %)\n";

    // Save binary image
    if (stbi_write_png(output_path.c_str(), W, H, 1, res.data.data(), W))
        std::cout << "\nSaved: " << output_path << "\n";
    else
        std::cerr << "Error saving: " << output_path << "\n";

    return EXIT_SUCCESS;
}
