#include "parallel/boundary_extraction.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace {

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(                                        \
                std::string("CUDA error: ") + cudaGetErrorString(err) +       \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
        }                                                                    \
    } while (0)

}

__global__ void piece_boundary_from_labels_kernel(
    const int* labels,
    uint8_t* boundary,
    int image_width,
    int image_height,
    int target_label,
    int x0,
    int y0,
    int boundary_width,
    int boundary_height)
{
    const int local_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (local_x >= boundary_width || local_y >= boundary_height) {
        return;
    }

    const int x = x0 + local_x;
    const int y = y0 + local_y;
    const int boundary_idx = local_y * boundary_width + local_x;
    const int image_idx = y * image_width + x;

    if (labels[image_idx] != target_label) {
        boundary[boundary_idx] = 0;
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
            if (nx < 0 || nx >= image_width || ny < 0 || ny >= image_height ||
                labels[ny * image_width + nx] != target_label) {
                is_boundary = true;
                break;
            }
        }
    }

    boundary[boundary_idx] = is_boundary ? 255 : 0;
}

void get_piece_boundary_mask_from_labels_cuda(
    const int* d_labels,
    const Region& region,
    uint8_t* d_boundary,
    int image_width,
    int image_height,
    ImageU8& boundary,
    Coordinate<int>& offset)
{
    const int x0 = std::max(0, region.x - 1);
    const int y0 = std::max(0, region.y - 1);
    const int x1 = std::min(image_width, region.x + region.width + 1);
    const int y1 = std::min(image_height, region.y + region.height + 1);

    const int boundary_width = x1 - x0;
    const int boundary_height = y1 - y0;
    const std::size_t boundary_size = static_cast<std::size_t>(boundary_width) * boundary_height;

    offset = Coordinate<int>{x0, y0};
    boundary.width = boundary_width;
    boundary.height = boundary_height;
    boundary.data.resize(boundary_size);

    dim3 block(16, 16);
    dim3 grid(
        (boundary_width + block.x - 1) / block.x,
        (boundary_height + block.y - 1) / block.y
    );

    piece_boundary_from_labels_kernel<<<grid, block>>>(
        d_labels,
        d_boundary,
        image_width,
        image_height,
        region.label,
        x0,
        y0,
        boundary_width,
        boundary_height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        boundary.data.data(),
        d_boundary,
        boundary_size * sizeof(uint8_t),
        cudaMemcpyDeviceToHost));
}
