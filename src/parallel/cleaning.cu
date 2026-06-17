#include "parallel/cleaning.cuh"

#include <cstdint>

#include <cuda_runtime.h>

__global__ void erode_kernel(
    const uint8_t* input,
    const uint8_t* kernel,
    uint8_t* output,
    int width,
    int height,
    int kw,
    int kh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int pad_w = kw / 2;
    const int pad_h = kh / 2;
    bool all_foreground = true;

    for (int ky = 0; ky < kh && all_foreground; ++ky) {
        for (int kx = 0; kx < kw; ++kx) {
            if (kernel[ky * kw + kx] != 1) {
                continue;
            }

            const int ix = x + kx - pad_w;
            const int iy = y + ky - pad_h;

            uint8_t pixel = 0;
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                pixel = input[iy * width + ix];
            }

            if (pixel != 255) {
                all_foreground = false;
                break;
            }
        }
    }

    output[y * width + x] = all_foreground ? 255 : 0;
}

__global__ void dilate_kernel(
    const uint8_t* input,
    const uint8_t* kernel,
    uint8_t* output,
    int width,
    int height,
    int kw,
    int kh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int pad_w = kw / 2;
    const int pad_h = kh / 2;
    bool any_foreground = false;

    for (int ky = 0; ky < kh && !any_foreground; ++ky) {
        for (int kx = 0; kx < kw; ++kx) {
            if (kernel[ky * kw + kx] != 1) {
                continue;
            }

            const int ix = x + kx - pad_w;
            const int iy = y + ky - pad_h;

            if (ix >= 0 && ix < width && iy >= 0 && iy < height &&
                input[iy * width + ix] == 255) {
                any_foreground = true;
                break;
            }
        }
    }

    output[y * width + x] = any_foreground ? 255 : 0;
}

void erode_cuda_device(
    const uint8_t* d_input,
    const uint8_t* d_kernel,
    uint8_t* d_output,
    int width,
    int height,
    int kernel_width,
    int kernel_height)
{
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    erode_kernel<<<grid, block>>>(
        d_input,
        d_kernel,
        d_output,
        width,
        height,
        kernel_width,
        kernel_height
    );
}

void dilate_cuda_device(
    const uint8_t* d_input,
    const uint8_t* d_kernel,
    uint8_t* d_output,
    int width,
    int height,
    int kernel_width,
    int kernel_height)
{
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    dilate_kernel<<<grid, block>>>(
        d_input,
        d_kernel,
        d_output,
        width,
        height,
        kernel_width,
        kernel_height
    );
}

void morphological_open_cuda_device(
    const uint8_t* d_input,
    const uint8_t* d_kernel,
    uint8_t* d_temp,
    uint8_t* d_output,
    int width,
    int height,
    int kernel_width,
    int kernel_height,
    int iterations)
{
    if (iterations <= 0) {
        cudaMemcpy(
            d_output,
            d_input,
            static_cast<size_t>(width) * height * sizeof(uint8_t),
            cudaMemcpyDeviceToDevice
        );
        return;
    }

    const uint8_t* src = d_input;
    uint8_t* dst = d_temp;

    for (int i = 0; i < iterations; ++i) {
        erode_cuda_device(
            src,
            d_kernel,
            dst,
            width,
            height,
            kernel_width,
            kernel_height
        );

        cudaDeviceSynchronize();

        src = dst;
        dst = (dst == d_temp) ? d_output : d_temp;
    }

    src = (iterations % 2 == 1) ? d_temp : d_output;
    dst = (src == d_temp) ? d_output : d_temp;

    for (int i = 0; i < iterations; ++i) {
        dilate_cuda_device(
            src,
            d_kernel,
            dst,
            width,
            height,
            kernel_width,
            kernel_height
        );

        cudaDeviceSynchronize();

        src = dst;
        dst = (dst == d_temp) ? d_output : d_temp;
    }

    if (src != d_output) {
        cudaMemcpy(
            d_output,
            src,
            static_cast<size_t>(width) * height * sizeof(uint8_t),
            cudaMemcpyDeviceToDevice
        );
    }
}
