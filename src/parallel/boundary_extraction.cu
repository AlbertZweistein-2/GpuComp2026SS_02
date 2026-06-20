#include "parallel/boundary_extraction.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

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

__global__ void count_piece_boundary_points_kernel(
    const int *labels,
    const BoundaryBox *boxes,
    int *point_counts,
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
        atomicAdd(&point_counts[box_idx], 1);
    }
}

__global__ void fill_piece_boundary_points_kernel(
    const int *labels,
    const BoundaryBox *boxes,
    const int *point_offsets,
    int *write_offsets,
    Coordinate<int> *points,
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
    if (!is_piece_boundary_pixel(labels, image_width, image_height, box.label, x, y))
    {
        return;
    }

    const int local_point_idx = atomicAdd(&write_offsets[box_idx], 1);
    points[point_offsets[box_idx] + local_point_idx] = Coordinate<int>{x, y};
}
} // namespace

void extract_piece_boundary_points_from_labels_cuda(
    const int *d_labels,
    const std::vector<Region> &regions,
    int image_width,
    int image_height,
    PieceBoundaryPointBatch &batch)
{
    std::vector<BoundaryBox> boxes;
    boxes.reserve(regions.size());
    batch.boundaries.clear();
    batch.boundaries.reserve(regions.size());

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
        batch.boundaries.push_back(PieceBoundaryMask{
            boundary_width,
            boundary_height,
            total_boundary_pixels,
            Coordinate<int>{x0, y0}});
        total_boundary_pixels += boundary_size;
        max_boundary_pixels = std::max(max_boundary_pixels, boundary_size);
    }

    const std::size_t region_count = regions.size();
    batch.point_counts.resize(region_count);
    batch.point_offsets.resize(region_count);
    batch.points.clear();

    if (region_count == 0 || total_boundary_pixels == 0)
    {
        return;
    }

    thrust::device_vector<BoundaryBox> d_boxes(boxes.begin(), boxes.end());
    thrust::fill(batch.point_counts.begin(), batch.point_counts.end(), 0);

    const int block = 256;
    const dim3 grid(
        static_cast<unsigned int>((max_boundary_pixels + block - 1) / block),
        static_cast<unsigned int>(boxes.size()));
    count_piece_boundary_points_kernel<<<grid, block>>>(
        d_labels,
        thrust::raw_pointer_cast(d_boxes.data()),
        thrust::raw_pointer_cast(batch.point_counts.data()),
        image_width,
        image_height);
    CUDA_CHECK(cudaGetLastError());

    thrust::exclusive_scan(
        batch.point_counts.begin(),
        batch.point_counts.end(),
        batch.point_offsets.begin());

    const int last_count = batch.point_counts.back();
    const int last_offset = batch.point_offsets.back();
    const int total_points = last_offset + last_count;
    batch.points.resize(static_cast<std::size_t>(total_points));
    if (total_points == 0)
    {
        return;
    }

    thrust::device_vector<int> write_offsets(region_count);
    thrust::fill(write_offsets.begin(), write_offsets.end(), 0);
    fill_piece_boundary_points_kernel<<<grid, block>>>(
        d_labels,
        thrust::raw_pointer_cast(d_boxes.data()),
        thrust::raw_pointer_cast(batch.point_offsets.data()),
        thrust::raw_pointer_cast(write_offsets.data()),
        thrust::raw_pointer_cast(batch.points.data()),
        image_width,
        image_height);
    CUDA_CHECK(cudaGetLastError());
}

void copy_boundary_points_to_masks_host(
    const PieceBoundaryPointBatch &batch,
    std::vector<uint8_t> &boundary_data)
{
    std::size_t total_boundary_pixels = 0;
    for (const PieceBoundaryMask &boundary : batch.boundaries)
    {
        total_boundary_pixels = std::max(
            total_boundary_pixels,
            boundary.data_offset + static_cast<std::size_t>(boundary.width) * boundary.height);
    }

    boundary_data.assign(total_boundary_pixels, 0);
    if (batch.boundaries.empty() || batch.points.empty())
    {
        return;
    }

    const thrust::host_vector<int> point_offsets = batch.point_offsets;
    const thrust::host_vector<int> point_counts = batch.point_counts;
    const thrust::host_vector<Coordinate<int>> points = batch.points;

    for (std::size_t piece_idx = 0; piece_idx < batch.boundaries.size(); ++piece_idx)
    {
        const PieceBoundaryMask &boundary = batch.boundaries[piece_idx];
        const int point_offset = point_offsets[piece_idx];
        const int point_count = point_counts[piece_idx];

        for (int i = 0; i < point_count; ++i)
        {
            const Coordinate<int> point = points[static_cast<std::size_t>(point_offset + i)];
            const int local_x = point.a - boundary.image_offset.a;
            const int local_y = point.b - boundary.image_offset.b;
            if (local_x < 0 || local_x >= boundary.width || local_y < 0 || local_y >= boundary.height)
            {
                continue;
            }

            const std::size_t mask_idx = boundary.data_offset +
                                         static_cast<std::size_t>(local_y) * boundary.width +
                                         local_x;
            boundary_data[mask_idx] = 255;
        }
    }
}
