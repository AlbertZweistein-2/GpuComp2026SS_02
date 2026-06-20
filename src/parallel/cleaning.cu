#include "parallel/cleaning.cuh"
#include "helpers.hpp"

#include <cstdint>

#include <cuda_runtime.h>

namespace
{
constexpr int MORPH_KERNEL_WIDTH = 5;
constexpr int MORPH_KERNEL_HEIGHT = 5;
} // namespace

__global__ void erode_kernel(
    const uint8_t *input,
    uint8_t *output,
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    const int pad_w = MORPH_KERNEL_WIDTH / 2;
    const int pad_h = MORPH_KERNEL_HEIGHT / 2;
    bool all_foreground = true;

    for (int ky = 0; ky < MORPH_KERNEL_HEIGHT && all_foreground; ++ky)
    {
        for (int kx = 0; kx < MORPH_KERNEL_WIDTH; ++kx)
        {
            const int ix = x + kx - pad_w;
            const int iy = y + ky - pad_h;

            uint8_t pixel = 0;
            if (ix >= 0 && ix < width && iy >= 0 && iy < height)
            {
                pixel = input[iy * width + ix];
            }

            if (pixel != 255)
            {
                all_foreground = false;
                break;
            }
        }
    }

    output[y * width + x] = all_foreground ? 255 : 0;
}

__global__ void dilate_kernel(
    const uint8_t *input,
    uint8_t *output,
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    const int pad_w = MORPH_KERNEL_WIDTH / 2;
    const int pad_h = MORPH_KERNEL_HEIGHT / 2;
    bool any_foreground = false;

    for (int ky = 0; ky < MORPH_KERNEL_HEIGHT && !any_foreground; ++ky)
    {
        for (int kx = 0; kx < MORPH_KERNEL_WIDTH; ++kx)
        {
            const int ix = x + kx - pad_w;
            const int iy = y + ky - pad_h;

            if (ix >= 0 && ix < width && iy >= 0 && iy < height &&
                input[iy * width + ix] == 255)
            {
                any_foreground = true;
                break;
            }
        }
    }

    output[y * width + x] = any_foreground ? 255 : 0;
}

static void erode_cuda_device(
    const uint8_t *d_input,
    uint8_t *d_output,
    int width,
    int height)
{
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    erode_kernel<<<grid, block>>>(
        d_input,
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

    dilate_kernel<<<grid, block>>>(
        d_input,
        d_output,
        width,
        height);
    CUDA_CHECK(cudaGetLastError());
}

void morphological_open_cuda_device(
    const uint8_t *d_input,
    uint8_t *d_temp,
    uint8_t *d_output,
    int width,
    int height)
{
    erode_cuda_device(
        d_input,
        d_temp,
        width,
        height);

    dilate_cuda_device(
        d_temp,
        d_output,
        width,
        height);
}
