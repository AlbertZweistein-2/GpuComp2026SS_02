#include "parallel/component_labeling.cuh"
#include "helpers.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

__global__ void init_labels_kernel(
    const uint8_t *binary,
    int *labels,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    labels[idx] = (binary[idx] > 0) ? idx + 1 : 0;
}

__global__ void propagate_labels_kernel(
    const uint8_t *binary,
    int *labels,
    int *changed,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    if (binary[idx] == 0 || labels[idx] == 0)
        return;

    int current = labels[idx];
    int min_label = current;

    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0)
                continue;

            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                continue;

            int nidx = ny * width + nx;
            if (binary[nidx] > 0 && labels[nidx] > 0)
            {
                min_label = min_label < labels[nidx] ? min_label : labels[nidx];
            }
        }
    }

    if (min_label < current)
    {
        labels[idx] = min_label;
        *changed = 1;
    }
}

__global__ void count_component_area_kernel(
    const int *labels,
    int *areas,
    int total_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels)
        return;

    int label = labels[idx];
    if (label > 0)
    {
        atomicAdd(&areas[label], 1);
    }
}

__global__ void filter_small_components_kernel(
    int *labels,
    const int *areas,
    int min_area,
    int total_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels)
        return;

    int label = labels[idx];
    if (label > 0 && areas[label] < min_area)
    {
        labels[idx] = 0;
    }
}

__global__ void compact_labels_kernel(
    const int *raw_labels,
    const int *unique_labels,
    int *compact_labels,
    int num_unique,
    int total_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels)
        return;

    int label = raw_labels[idx];
    if (label == 0)
    {
        compact_labels[idx] = 0;
        return;
    }

    int left = 0;
    int right = num_unique - 1;

    while (left <= right)
    {
        int mid = (left + right) / 2;
        int mid_label = unique_labels[mid];

        if (mid_label == label)
        {
            compact_labels[idx] = mid + 1;
            return;
        }

        if (mid_label < label)
        {
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }

    compact_labels[idx] = 0;
}

void init_labels_cuda_device(
    const uint8_t *d_binary,
    int *d_labels,
    int width,
    int height)
{
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    init_labels_kernel<<<grid, block>>>(d_binary, d_labels, width, height);
}

void propagate_labels_cuda_device(
    const uint8_t *d_binary,
    int *d_labels,
    int *d_changed,
    int width,
    int height)
{
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    propagate_labels_kernel<<<grid, block>>>(d_binary, d_labels, d_changed, width, height);
}

void count_component_area_cuda_device(
    const int *d_labels,
    int *d_areas,
    int total_pixels)
{
    int block = 256;
    int grid = (total_pixels + block - 1) / block;
    count_component_area_kernel<<<grid, block>>>(d_labels, d_areas, total_pixels);
}

void filter_small_components_cuda_device(
    int *d_labels,
    const int *d_areas,
    int min_area,
    int total_pixels)
{
    int block = 256;
    int grid = (total_pixels + block - 1) / block;
    filter_small_components_kernel<<<grid, block>>>(d_labels, d_areas, min_area, total_pixels);
}

void connected_components_cuda_device_raw(
    const uint8_t *d_binary,
    int *d_labels,
    int *d_changed,
    int *d_areas,
    int width,
    int height,
    int min_area,
    int max_label_count)
{
    const int total_pixels = width * height;

    init_labels_cuda_device(d_binary, d_labels, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_changed = 1;
    int iterations = 0;
    const int max_iterations = width + height;

    while (h_changed && iterations < max_iterations)
    {
        h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));

        propagate_labels_cuda_device(d_binary, d_labels, d_changed, width, height);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        ++iterations;
    }

    cudaMemset(d_areas, 0, static_cast<size_t>(max_label_count) * sizeof(int));
    count_component_area_cuda_device(d_labels, d_areas, total_pixels);
    CUDA_CHECK(cudaDeviceSynchronize());

    filter_small_components_cuda_device(d_labels, d_areas, min_area, total_pixels);
    CUDA_CHECK(cudaDeviceSynchronize());
}

int connected_components_cuda_device(
    const uint8_t *d_binary,
    int *d_compact_labels,
    int width,
    int height,
    int min_area)
{
    const int total_pixels = width * height;

    thrust::device_vector<int> d_labels(static_cast<size_t>(total_pixels));
    thrust::device_vector<int> d_changed(1);
    thrust::device_vector<int> d_areas(static_cast<size_t>(total_pixels) + 1);

    connected_components_cuda_device_raw(
        d_binary,
        thrust::raw_pointer_cast(d_labels.data()),
        thrust::raw_pointer_cast(d_changed.data()),
        thrust::raw_pointer_cast(d_areas.data()),
        width,
        height,
        min_area,
        total_pixels + 1);

    return compact_labels_cuda_device(
        thrust::raw_pointer_cast(d_labels.data()),
        d_compact_labels,
        total_pixels);
}

int compact_labels_cuda_device(
    const int *d_labels,
    int *d_compact_labels,
    int total_pixels)
{
    thrust::device_vector<int> unique_labels(total_pixels);

    thrust::copy(
        thrust::device,
        d_labels,
        d_labels + total_pixels,
        unique_labels.begin());

    thrust::sort(thrust::device, unique_labels.begin(), unique_labels.end());

    auto nonzero_end = thrust::remove(
        thrust::device,
        unique_labels.begin(),
        unique_labels.end(),
        0);
    unique_labels.erase(nonzero_end, unique_labels.end());

    auto unique_end = thrust::unique(
        thrust::device,
        unique_labels.begin(),
        unique_labels.end());
    unique_labels.erase(unique_end, unique_labels.end());

    int num_unique = static_cast<int>(unique_labels.size());

    int block = 256;
    int grid = (total_pixels + block - 1) / block;

    compact_labels_kernel<<<grid, block>>>(
        d_labels,
        thrust::raw_pointer_cast(unique_labels.data()),
        d_compact_labels,
        num_unique,
        total_pixels);
    CUDA_CHECK(cudaDeviceSynchronize());

    return num_unique;
}

std::vector<Region> build_regions_from_labels_cuda(
    const int *d_compact_labels,
    int width,
    int height,
    int num_components)
{
    const int total_pixels = width * height;
    std::vector<int> h_labels(static_cast<size_t>(total_pixels));

    CUDA_CHECK(cudaMemcpy(
        h_labels.data(),
        d_compact_labels,
        static_cast<size_t>(total_pixels) * sizeof(int),
        cudaMemcpyDeviceToHost));

    std::vector<Region> regions;
    regions.reserve(static_cast<size_t>(num_components));

    for (int label = 1; label <= num_components; ++label)
    {
        Region r;
        r.label = label;
        r.area = 0;
        r.x = width;
        r.y = height;
        r.width = 0;
        r.height = 0;

        int max_x = -1;
        int max_y = -1;

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const int idx = y * width + x;
                if (h_labels[idx] != label)
                {
                    continue;
                }

                ++r.area;
                if (x < r.x)
                    r.x = x;
                if (y < r.y)
                    r.y = y;
                if (x > max_x)
                    max_x = x;
                if (y > max_y)
                    max_y = y;
            }
        }

        if (r.area > 0)
        {
            r.width = max_x - r.x + 1;
            r.height = max_y - r.y + 1;
            regions.push_back(r);
        }
    }

    return regions;
}

__global__ void extract_piece_mask_kernel(
    const int *labels,
    uint8_t *mask,
    int target_label,
    int total_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels)
        return;

    mask[idx] = (labels[idx] == target_label) ? 255 : 0;
}

void extract_piece_mask_cuda_device(
    const int *d_labels,
    uint8_t *d_mask,
    int target_label,
    int total_pixels)
{
    int block = 256;
    int grid = (total_pixels + block - 1) / block;
    extract_piece_mask_kernel<<<grid, block>>>(d_labels, d_mask, target_label, total_pixels);
}
