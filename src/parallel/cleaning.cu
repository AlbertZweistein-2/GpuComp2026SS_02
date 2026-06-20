#include "parallel/cleaning.cuh"
#include "helpers.hpp"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace
{
constexpr int MORPH_KERNEL_WIDTH = 5;
constexpr int MORPH_KERNEL_HEIGHT = 5;
constexpr int MORPH_RADIUS_X = MORPH_KERNEL_WIDTH / 2;
constexpr int MORPH_RADIUS_Y = MORPH_KERNEL_HEIGHT / 2;
} // namespace

__global__ void binarize_erode_tiled_kernel(
    const float *input,
    const float *threshold,
    uint8_t *output,
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tile_width = blockDim.x + 2 * MORPH_RADIUS_X;
    const int tile_height = blockDim.y + 2 * MORPH_RADIUS_Y;
    const int tile_area = tile_width * tile_height;
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y;
    const float threshold_value = *threshold;

    extern __shared__ uint8_t tile[];
    for (int idx = thread_id; idx < tile_area; idx += threads_per_block)
    {
        const int tile_y = idx / tile_width;
        const int tile_x = idx - tile_y * tile_width;
        const int global_x = blockIdx.x * blockDim.x + tile_x - MORPH_RADIUS_X;
        const int global_y = blockIdx.y * blockDim.y + tile_y - MORPH_RADIUS_Y;

        uint8_t value = 0;
        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height)
        {
            value = input[global_y * width + global_x] > threshold_value ? 255 : 0;
        }
        tile[idx] = value;
    }

    __syncthreads();

    if (x >= width || y >= height)
    {
        return;
    }

    const int local_x = threadIdx.x + MORPH_RADIUS_X;
    const int local_y = threadIdx.y + MORPH_RADIUS_Y;
    bool all_foreground = true;

    for (int ky = 0; ky < MORPH_KERNEL_HEIGHT && all_foreground; ++ky)
    {
        for (int kx = 0; kx < MORPH_KERNEL_WIDTH; ++kx)
        {
            const int tile_idx = (local_y + ky - MORPH_RADIUS_Y) * tile_width +
                                 (local_x + kx - MORPH_RADIUS_X);
            if (tile[tile_idx] != 255)
            {
                all_foreground = false;
                break;
            }
        }
    }

    output[y * width + x] = all_foreground ? 255 : 0;
}

__global__ void dilate_tiled_kernel(
    const uint8_t *input,
    uint8_t *output,
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tile_width = blockDim.x + 2 * MORPH_RADIUS_X;
    const int tile_height = blockDim.y + 2 * MORPH_RADIUS_Y;
    const int tile_area = tile_width * tile_height;
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y;

    extern __shared__ uint8_t tile[];
    for (int idx = thread_id; idx < tile_area; idx += threads_per_block)
    {
        const int tile_y = idx / tile_width;
        const int tile_x = idx - tile_y * tile_width;
        const int global_x = blockIdx.x * blockDim.x + tile_x - MORPH_RADIUS_X;
        const int global_y = blockIdx.y * blockDim.y + tile_y - MORPH_RADIUS_Y;

        uint8_t value = 0;
        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height)
        {
            value = input[global_y * width + global_x];
        }
        tile[idx] = value;
    }

    __syncthreads();

    if (x >= width || y >= height)
    {
        return;
    }

    const int local_x = threadIdx.x + MORPH_RADIUS_X;
    const int local_y = threadIdx.y + MORPH_RADIUS_Y;
    bool any_foreground = false;

    for (int ky = 0; ky < MORPH_KERNEL_HEIGHT && !any_foreground; ++ky)
    {
        for (int kx = 0; kx < MORPH_KERNEL_WIDTH; ++kx)
        {
            const int tile_idx = (local_y + ky - MORPH_RADIUS_Y) * tile_width +
                                 (local_x + kx - MORPH_RADIUS_X);
            if (tile[tile_idx] == 255)
            {
                any_foreground = true;
                break;
            }
        }
    }

    output[y * width + x] = any_foreground ? 255 : 0;
}

static void binarize_erode_cuda_device(
    const float *d_input,
    const float *d_threshold,
    uint8_t *d_output,
    int width,
    int height)
{
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);
    const size_t shared_bytes = static_cast<size_t>(block.x + 2 * MORPH_RADIUS_X) *
                                static_cast<size_t>(block.y + 2 * MORPH_RADIUS_Y) *
                                sizeof(uint8_t);

    binarize_erode_tiled_kernel<<<grid, block, shared_bytes>>>(
        d_input,
        d_threshold,
        d_output,
        width,
        height);
    CUDA_CHECK(cudaGetLastError());
}

static void dilate_cuda_device(
    const uint8_t *d_input,
    uint8_t *d_output,
    int width,
    int height)
{
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);
    const size_t shared_bytes = static_cast<size_t>(block.x + 2 * MORPH_RADIUS_X) *
                                static_cast<size_t>(block.y + 2 * MORPH_RADIUS_Y) *
                                sizeof(uint8_t);

    dilate_tiled_kernel<<<grid, block, shared_bytes>>>(
        d_input,
        d_output,
        width,
        height);
    CUDA_CHECK(cudaGetLastError());
}

void binarize_morphological_open_cuda_device(
    const float *d_input,
    const float *d_threshold,
    uint8_t *d_temp,
    uint8_t *d_output,
    int width,
    int height)
{
    binarize_erode_cuda_device(
        d_input,
        d_threshold,
        d_temp,
        width,
        height);
    dilate_cuda_device(
        d_temp,
        d_output,
        width,
        height);

}
