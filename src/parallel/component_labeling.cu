#include "parallel/component_labeling.cuh"
#include "helpers.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
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

__device__ __forceinline__ int find_root_label(
    int *labels,
    int label)
{
    int parent = labels[label - 1];
    while (parent != label)
    {
        int grandparent = labels[parent - 1];
        labels[label - 1] = grandparent;
        label = parent;
        parent = grandparent;
    }

    return label;
}

__device__ void union_label_roots(
    int *labels,
    int a,
    int b)
{
    while (true)
    {
        a = find_root_label(labels, a);
        b = find_root_label(labels, b);

        if (a == b)
            return;

        int high = a > b ? a : b;
        int low = a < b ? a : b;

        int old = atomicCAS(&labels[high - 1], high, low);
        if (old == high)
            return;
    }
}

__global__ void union_neighbor_labels_kernel(
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
    if (binary[idx] == 0 || labels[idx] == 0)
        return;

    int label = labels[idx];

    if (x + 1 < width && binary[idx + 1] > 0)
    {
        union_label_roots(labels, label, labels[idx + 1]);
    }

    if (y + 1 < height)
    {
        int down = idx + width;
        if (binary[down] > 0)
        {
            union_label_roots(labels, label, labels[down]);
        }

        if (x > 0 && binary[down - 1] > 0)
        {
            union_label_roots(labels, label, labels[down - 1]);
        }

        if (x + 1 < width && binary[down + 1] > 0)
        {
            union_label_roots(labels, label, labels[down + 1]);
        }
    }
}

__global__ void compress_labels_kernel(
    int *labels,
    int total_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels)
        return;

    int label = labels[idx];
    if (label > 0)
    {
        labels[idx] = find_root_label(labels, label);
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

void union_labels_cuda_device(
    const uint8_t *d_binary,
    int *d_labels,
    int width,
    int height)
{
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    union_neighbor_labels_kernel<<<grid, block>>>(d_binary, d_labels, width, height);
}

void compress_labels_cuda_device(
    int *d_labels,
    int total_pixels)
{
    int block = 256;
    int grid = (total_pixels + block - 1) / block;
    compress_labels_kernel<<<grid, block>>>(d_labels, total_pixels);
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
    (void)d_changed;
    (void)d_areas;
    (void)min_area;
    (void)max_label_count;
    const int total_pixels = width * height;

    init_labels_cuda_device(d_binary, d_labels, width, height);
    union_labels_cuda_device(d_binary, d_labels, width, height);
    compress_labels_cuda_device(d_labels, total_pixels);
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

    connected_components_cuda_device_raw(
        d_binary,
        thrust::raw_pointer_cast(d_labels.data()),
        nullptr,
        nullptr,
        width,
        height,
        min_area,
        total_pixels + 1);

    CUDA_CHECK(cudaMemcpy(
        d_compact_labels,
        thrust::raw_pointer_cast(d_labels.data()),
        static_cast<size_t>(total_pixels) * sizeof(int),
        cudaMemcpyDeviceToDevice));

    return 0;
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

void build_regions_from_labels_cuda(
    const int *d_compact_labels,
    int width,
    int height,
    int min_area,
    std::vector<Region> &regions)
{
    const int total_pixels = width * height;
    std::vector<int> h_labels(static_cast<size_t>(total_pixels));

    CUDA_CHECK(cudaMemcpy(
        h_labels.data(),
        d_compact_labels,
        static_cast<size_t>(total_pixels) * sizeof(int),
        cudaMemcpyDeviceToHost));

    regions.clear();

    std::unordered_map<int, size_t> label_to_region;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int idx = y * width + x;
            const int label = h_labels[idx];
            if (label == 0)
            {
                continue;
            }

            auto [it, inserted] = label_to_region.emplace(label, regions.size());
            if (inserted)
            {
                regions.push_back(Region{label, 0, x, y, 1, 1});
            }

            Region &region = regions[it->second];
            ++region.area;

            const int max_x = region.x + region.width - 1;
            const int max_y = region.y + region.height - 1;

            if (x < region.x)
            {
                region.width = max_x - x + 1;
                region.x = x;
            }
            else if (x > max_x)
            {
                region.width = x - region.x + 1;
            }

            if (y < region.y)
            {
                region.height = max_y - y + 1;
                region.y = y;
            }
            else if (y > max_y)
            {
                region.height = y - region.y + 1;
            }
        }
    }

    std::sort(regions.begin(), regions.end(), [](const Region &a, const Region &b)
              {
                  if (a.y != b.y)
                      return a.y < b.y;
                  return a.x < b.x;
              });

    regions.erase(
        std::remove_if(regions.begin(), regions.end(), [min_area](const Region &region)
                       { return region.area < min_area; }),
        regions.end());
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
