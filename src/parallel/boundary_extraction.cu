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

BoundaryExtractionPlan build_boundary_extraction_plan(
    const std::vector<Region> &regions,
    int image_width,
    int image_height,
    std::vector<PieceBoundaryMask> &boundaries)
{
    BoundaryExtractionPlan plan;
    plan.boxes.reserve(regions.size());
    boundaries.clear();
    boundaries.reserve(regions.size());

    for (const Region &region : regions)
    {
        const int x0 = std::max(0, region.x - 1);
        const int y0 = std::max(0, region.y - 1);
        const int x1 = std::min(image_width, region.x + region.width + 1);
        const int y1 = std::min(image_height, region.y + region.height + 1);

        const int boundary_width = x1 - x0;
        const int boundary_height = y1 - y0;
        const std::size_t boundary_size = static_cast<std::size_t>(boundary_width) * boundary_height;

        plan.boxes.push_back(BoundaryBox{
            region.label,
            x0,
            y0,
            boundary_width,
            boundary_height,
            plan.total_boundary_pixels});
        boundaries.push_back(PieceBoundaryMask{
            boundary_width,
            boundary_height,
            plan.total_boundary_pixels,
            Coordinate<int>{x0, y0}});
        plan.total_boundary_pixels += boundary_size;
        plan.max_boundary_pixels = std::max(plan.max_boundary_pixels, boundary_size);
    }

    return plan;
}

void allocate_boundary_extraction_cuda_scratch(
    const BoundaryExtractionPlan &plan,
    BoundaryExtractionCudaScratch &scratch)
{
    scratch.boxes = plan.boxes;
    scratch.boundary_data.resize(plan.total_boundary_pixels);
}

void extract_piece_boundary_masks_from_labels_cuda(
    const int *d_labels,
    const BoundaryExtractionPlan &plan,
    BoundaryExtractionCudaScratch &scratch,
    int image_width,
    int image_height,
    std::vector<uint8_t> &boundary_data)
{
    boundary_data.clear();
    if (plan.boxes.empty() || plan.max_boundary_pixels == 0)
    {
        return;
    }

    CUDA_CHECK(cudaMemset(
        thrust::raw_pointer_cast(scratch.boundary_data.data()),
        0,
        plan.total_boundary_pixels * sizeof(uint8_t)));

    // We launch one grid row per piece. The x dimension is sized by the largest
    // cropped box, so smaller boxes exit early once local_idx exceeds their area.
    const int block = 256;
    const dim3 grid(
        static_cast<unsigned int>((plan.max_boundary_pixels + block - 1) / block),
        static_cast<unsigned int>(plan.boxes.size()));
    fill_piece_boundary_masks_kernel<<<grid, block>>>(
        d_labels,
        thrust::raw_pointer_cast(scratch.boxes.data()),
        thrust::raw_pointer_cast(scratch.boundary_data.data()),
        image_width,
        image_height);
    CUDA_CHECK(cudaGetLastError());

    boundary_data.resize(plan.total_boundary_pixels);
    thrust::copy(scratch.boundary_data.begin(), scratch.boundary_data.end(), boundary_data.begin());
}
