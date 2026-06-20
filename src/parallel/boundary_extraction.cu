#include "parallel/boundary_extraction.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "helpers.hpp"

namespace
{
struct BoundaryBox
{
    int label;
    int x0;
    int y0;
    int width;
    int height;
    std::size_t data_offset;
};

__global__ void piece_boundaries_from_labels_kernel(
    const int *labels,
    const BoundaryBox *boxes,
    uint8_t *boundary_data,
    int image_width,
    int image_height)
{
    const int box_idx = blockIdx.y;
    const BoundaryBox box = boxes[box_idx];
    const std::size_t local_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_idx >= static_cast<std::size_t>(box.width) * box.height)
    {
        return;
    }

    const std::size_t packed_idx = box.data_offset + local_idx;
    const int local_x = static_cast<int>(local_idx % box.width);
    const int local_y = static_cast<int>(local_idx / box.width);

    const int x = box.x0 + local_x;
    const int y = box.y0 + local_y;
    const int image_idx = y * image_width + x;

    if (labels[image_idx] != box.label)
    {
        boundary_data[packed_idx] = 0;
        return;
    }

    bool is_boundary = false;
    for (int dy = -1; dy <= 1 && !is_boundary; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0)
            {
                continue;
            }

            const int nx = x + dx;
            const int ny = y + dy;
            if (nx < 0 || nx >= image_width || ny < 0 || ny >= image_height ||
                labels[ny * image_width + nx] != box.label)
            {
                is_boundary = true;
                break;
            }
        }
    }

    boundary_data[packed_idx] = is_boundary ? 255 : 0;
}
} // namespace

void get_piece_boundary_masks_from_labels_cuda(
    const int *d_labels,
    const std::vector<Region> &regions,
    int image_width,
    int image_height,
    std::vector<uint8_t> &boundary_data,
    std::vector<PieceBoundaryMask> &boundaries)
{
    std::vector<BoundaryBox> boxes;
    boxes.reserve(regions.size());
    boundaries.clear();
    boundaries.reserve(regions.size());

    std::size_t total_boundary_pixels = 0;
    std::size_t max_boundary_pixels = 0;
    for (const Region &region : regions)
    {
        const int x0 = std::max(0, region.x - 1);
        const int y0 = std::max(0, region.y - 1);
        const int x1 = std::min(image_width, region.x + region.width + 1);
        const int y1 = std::min(image_height, region.y + region.height + 1);

        const int boundary_width = x1 - x0;
        const int boundary_height = y1 - y0;
        const std::size_t boundary_size = static_cast<std::size_t>(boundary_width) * boundary_height;

        boxes.push_back(BoundaryBox{
            region.label,
            x0,
            y0,
            boundary_width,
            boundary_height,
            total_boundary_pixels});
        boundaries.push_back(PieceBoundaryMask{
            boundary_width,
            boundary_height,
            total_boundary_pixels,
            Coordinate<int>{x0, y0}});
        total_boundary_pixels += boundary_size;
        max_boundary_pixels = std::max(max_boundary_pixels, boundary_size);
    }

    boundary_data.resize(total_boundary_pixels);
    if (total_boundary_pixels == 0)
    {
        return;
    }

    thrust::device_vector<BoundaryBox> d_boxes(boxes.begin(), boxes.end());
    thrust::device_vector<uint8_t> d_boundary_data(total_boundary_pixels);

    const int block = 256;
    const dim3 grid(
        static_cast<unsigned int>((max_boundary_pixels + block - 1) / block),
        static_cast<unsigned int>(boxes.size()));
    piece_boundaries_from_labels_kernel<<<grid, block>>>(
        d_labels,
        thrust::raw_pointer_cast(d_boxes.data()),
        thrust::raw_pointer_cast(d_boundary_data.data()),
        image_width,
        image_height);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(
        boundary_data.data(),
        thrust::raw_pointer_cast(d_boundary_data.data()),
        total_boundary_pixels * sizeof(uint8_t),
        cudaMemcpyDeviceToHost));
}
