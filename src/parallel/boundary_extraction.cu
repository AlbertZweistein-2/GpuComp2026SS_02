#include "parallel/boundary_extraction.cuh"

#include <cstdint>

#include <cuda_runtime.h>

__global__ void extract_piece_mask_kernel(
    const int* labels,
    uint8_t* mask,
    int target_label,
    int total_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    mask[idx] = (labels[idx] == target_label) ? 255 : 0;
}

__global__ void boundary_kernel(
    const uint8_t* input,
    uint8_t* boundary,
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int idx = y * width + x;

    if (input[idx] == 0) {
        boundary[idx] = 0;
        return;
    }

    bool is_boundary = false;

    for (int dy = -1; dy <= 1 && !is_boundary; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }

            const int nx = x + dx;
            const int ny = y + dy;

            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                is_boundary = true;
                break;
            }

            if (input[ny * width + nx] == 0) {
                is_boundary = true;
                break;
            }
        }
    }

    boundary[idx] = is_boundary ? 255 : 0;
}

void extract_piece_mask_cuda_device(
    const int* d_labels,
    uint8_t* d_mask,
    int target_label,
    int total_pixels)
{
    int block = 256;
    int grid = (total_pixels + block - 1) / block;

    extract_piece_mask_kernel<<<grid, block>>>(d_labels, d_mask, target_label, total_pixels);
}

void get_external_boundary_mask_cuda_device(
    const uint8_t* d_input,
    uint8_t* d_boundary,
    int width,
    int height)
{
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    boundary_kernel<<<grid, block>>>(d_input, d_boundary, width, height);
}
