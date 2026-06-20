#include "parallel/boundary_extraction.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "helpers.hpp"

namespace
{
struct BoundaryBox
{
    // Cropped bounding box around one region.
    // The kernels use this to avoid scanning the full image once per piece.
    int label;
    int x0;
    int y0;
    int width;
    int height;
    std::size_t data_offset;
};

// Checks whether a pixel belongs to the requested component and touches
// background or another component in its 8-neighborhood.
// Such pixels form the outer boundary mask for that piece.
__device__ bool is_piece_boundary_pixel(
    const int *labels,
    int image_width,
    int image_height,
    int label,
    int x,
    int y)
{
    const int image_idx = y * image_width + x;
    if (labels[image_idx] != label)
    {
        return false;
    }

    for (int dy = -1; dy <= 1; ++dy)
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
                labels[ny * image_width + nx] != label)
            {
                return true;
            }
        }
    }

    return false;
}

// Batched boundary extraction.
// blockIdx.y selects the piece; blockIdx.x/threadIdx.x walk that piece's
// cropped bounding box. Boundary pixels are written into the flat mask buffer
// at the precomputed offset for the selected piece.
__global__ void fill_piece_boundary_masks_kernel(
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

    const int local_x = static_cast<int>(local_idx % box.width);
    const int local_y = static_cast<int>(local_idx / box.width);
    const int x = box.x0 + local_x;
    const int y = box.y0 + local_y;
    if (is_piece_boundary_pixel(labels, image_width, image_height, box.label, x, y))
    {
        boundary_data[box.data_offset + local_idx] = 255;
    }
}
} // namespace

void extract_piece_boundary_masks_from_labels_cuda(
    const int *d_labels,
    const std::vector<Region> &regions,
    int image_width,
    int image_height,
    std::vector<PieceBoundaryMask> &boundaries,
    std::vector<uint8_t> &boundary_data)
{
    // Build one small host-side descriptor per region. This is cheap compared
    // to scanning the full image and lets the device kernels process all pieces
    // in one batched launch.
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

    boundary_data.clear();
    if (boxes.empty() || max_boundary_pixels == 0)
    {
        return;
    }

    thrust::device_vector<BoundaryBox> d_boxes(boxes.begin(), boxes.end());
    thrust::device_vector<uint8_t> d_boundary_data(total_boundary_pixels, 0);

    // We launch one grid row per piece. The x dimension is sized by the largest
    // cropped box, so smaller boxes exit early once local_idx exceeds their area.
    const int block = 256;
    const dim3 grid(
        static_cast<unsigned int>((max_boundary_pixels + block - 1) / block),
        static_cast<unsigned int>(boxes.size()));
    fill_piece_boundary_masks_kernel<<<grid, block>>>(
        d_labels,
        thrust::raw_pointer_cast(d_boxes.data()),
        thrust::raw_pointer_cast(d_boundary_data.data()),
        image_width,
        image_height);
    CUDA_CHECK(cudaGetLastError());

    boundary_data.resize(total_boundary_pixels);
    thrust::copy(d_boundary_data.begin(), d_boundary_data.end(), boundary_data.begin());
}
