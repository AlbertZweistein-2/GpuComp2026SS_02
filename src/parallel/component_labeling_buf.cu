
// @Anantha
// Yes, but cite the paper as the main justification, not GitHub alone.
// Use:
// Stefano Allegretti, Federico Bolelli, Michele Cancilla, Costantino Grana, “A Block-Based Union-Find Algorithm to Label Connected Components on GPUs”, ICIAP 2019.
// YACCLAB GitHub repository as reference implementation / benchmark framework.
// Good wording for your report:
// We replaced iterative label propagation with a Block-Based Union-Find (BUF) connected-component labeling algorithm following Allegretti et al. BUF is designed for GPU-based 8-connectivity CCL and uses 2x2 pixel blocks instead of per-pixel labels, reducing temporary labels, memory accesses, and union-find work. The authors report improved performance over previous GPU CCL methods, and the implementation is available in the YACCLAB benchmark suite.

// Important nuance: our current component_labeling_buf.cu is based on the BUF idea, but it is not yet a literal copy of Algorithm 2’s optimized bitmask merge. So describe it as:
// “a BUF-inspired block-based union-find implementation”

// unless we fully match their bitset neighborhood mask logic. That is still valid, but more honest technically.

// But we need a way to reason how we came up with this shit

// Yes. The reasoning chain is defensible:
// The profiler showed propagate_labels_kernel was called many times.

// Looking at the old implementation, each foreground pixel started with its own label, then each kernel launch only propagated the minimum label to immediate neighbors. That means labels spread across a component roughly one pixel per iteration. For large puzzle pieces, this becomes many full-image passes.

// This matches the paper’s related-work discussion: label-equivalence / propagation-style GPU CCL methods repeatedly propagate labels or resolve equivalences, which is exactly the bad behavior we saw.

// The natural fix is union-find: instead of letting labels slowly diffuse through the image, directly merge equivalent labels into component trees, then compress the trees.

// The first union-find version still worked per pixel, so it reduced launch count but still had many labels, many memory accesses, and many atomic operations.

// Allegretti et al. propose Block-Based Union-Find for 8-connectivity, using the fact that all foreground pixels inside a 2x2 block are connected. Therefore, the algorithm can operate mostly on block labels instead of pixel labels.

// Our problem also uses binary image CCL with 8-connectivity, so BUF fits the exact structure of the task.

// Report wording:
// Profiling showed that connected-component labeling was dominated by repeated label propagation. The original implementation used an iterative relaxation scheme where labels moved only through local neighborhoods, requiring many full-image kernel launches for large components. We therefore looked for GPU CCL algorithms that avoid iterative propagation. Union-find based labeling directly resolves label equivalences, and Allegretti et al. further optimize this idea with Block-Based Union-Find for 8-connected images. Since in 8-connectivity all foreground pixels inside a 2x2 block belong to the same component, BUF reduces the number of labels and memory accesses by processing blocks instead of individual pixels. This made it a suitable replacement for our propagation-based approach.

// Then cite the paper and mention YACCLAB as benchmark/source-code evidence.

//Paper: https://www.federicobolelli.it/media/publications/pdfs/2019iciap_labeling.pdf
//Github: https://github.com/prittt/YACCLAB/tree/master


#include "parallel/component_labeling.cuh"
#include "helpers.hpp"

#include <algorithm>
#include <cstddef>
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

namespace
{
    __device__ __forceinline__ bool in_bounds(
        int x,
        int y,
        int width,
        int height)
    {
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    __device__ __forceinline__ bool is_foreground(
        const uint8_t *binary,
        int x,
        int y,
        int width,
        int height)
    {
        return in_bounds(x, y, width, height) && binary[y * width + x] > 0;
    }

    __device__ __forceinline__ int block_top_left_index(
        int block_x,
        int block_y,
        int width)
    {
        return (2 * block_y) * width + 2 * block_x;
    }

    __device__ __forceinline__ int find_root_label(
        const int *parents,
        int label)
    {
        int parent = parents[label - 1];
        while (parent != label)
        {
            label = parent;
            parent = parents[label - 1];
        }

        return label;
    }

    __device__ void union_label_roots(
        int *parents,
        int a,
        int b)
    {
        while (true)
        {
            a = find_root_label(parents, a);
            b = find_root_label(parents, b);

            if (a == b)
            {
                return;
            }

            const int high = a > b ? a : b;
            const int low = a < b ? a : b;
            const int old = atomicCAS(&parents[high - 1], high, low);

            if (old == high)
            {
                return;
            }
        }
    }

    __global__ void init_block_labels_kernel(
        const uint8_t *binary,
        int *parents,
        int width,
        int height)
    {
        const int block_x = blockIdx.x * blockDim.x + threadIdx.x;
        const int block_y = blockIdx.y * blockDim.y + threadIdx.y;
        const int x0 = 2 * block_x;
        const int y0 = 2 * block_y;

        if (x0 >= width || y0 >= height)
        {
            return;
        }

        const int top_left = y0 * width + x0;
        parents[top_left] = 0;
        if (x0 + 1 < width)
        {
            parents[top_left + 1] = 0;
        }
        if (y0 + 1 < height)
        {
            parents[top_left + width] = 0;
            if (x0 + 1 < width)
            {
                parents[top_left + width + 1] = 0;
            }
        }

        const bool active =
            is_foreground(binary, x0, y0, width, height) ||
            is_foreground(binary, x0 + 1, y0, width, height) ||
            is_foreground(binary, x0, y0 + 1, width, height) ||
            is_foreground(binary, x0 + 1, y0 + 1, width, height);

        if (active)
        {
            parents[top_left] = top_left + 1;
        }
    }

    __global__ void merge_block_labels_kernel(
        const uint8_t *binary,
        int *parents,
        int width,
        int height)
    {
        const int block_x = blockIdx.x * blockDim.x + threadIdx.x;
        const int block_y = blockIdx.y * blockDim.y + threadIdx.y;
        const int x0 = 2 * block_x;
        const int y0 = 2 * block_y;

        if (x0 >= width || y0 >= height)
        {
            return;
        }

        const int top_left = block_top_left_index(block_x, block_y, width);
        const int label = parents[top_left];
        if (label == 0)
        {
            return;
        }

        if (x0 + 2 < width)
        {
            const bool right_column =
                is_foreground(binary, x0 + 1, y0, width, height) ||
                is_foreground(binary, x0 + 1, y0 + 1, width, height);
            const bool neighbor_left_column =
                is_foreground(binary, x0 + 2, y0, width, height) ||
                is_foreground(binary, x0 + 2, y0 + 1, width, height);

            if (right_column && neighbor_left_column)
            {
                const int neighbor = y0 * width + x0 + 2;
                const int neighbor_label = parents[neighbor];
                if (neighbor_label > 0)
                {
                    union_label_roots(parents, label, neighbor_label);
                }
            }
        }

        if (y0 + 2 < height)
        {
            const bool bottom_row =
                is_foreground(binary, x0, y0 + 1, width, height) ||
                is_foreground(binary, x0 + 1, y0 + 1, width, height);
            const bool neighbor_top_row =
                is_foreground(binary, x0, y0 + 2, width, height) ||
                is_foreground(binary, x0 + 1, y0 + 2, width, height);

            if (bottom_row && neighbor_top_row)
            {
                const int neighbor = (y0 + 2) * width + x0;
                const int neighbor_label = parents[neighbor];
                if (neighbor_label > 0)
                {
                    union_label_roots(parents, label, neighbor_label);
                }
            }

            if (x0 + 2 < width &&
                is_foreground(binary, x0 + 1, y0 + 1, width, height) &&
                is_foreground(binary, x0 + 2, y0 + 2, width, height))
            {
                const int neighbor = (y0 + 2) * width + x0 + 2;
                const int neighbor_label = parents[neighbor];
                if (neighbor_label > 0)
                {
                    union_label_roots(parents, label, neighbor_label);
                }
            }

            if (x0 > 0 &&
                is_foreground(binary, x0, y0 + 1, width, height) &&
                is_foreground(binary, x0 - 1, y0 + 2, width, height))
            {
                const int neighbor = (y0 + 2) * width + x0 - 2;
                const int neighbor_label = parents[neighbor];
                if (neighbor_label > 0)
                {
                    union_label_roots(parents, label, neighbor_label);
                }
            }
        }
    }

    __global__ void compress_block_labels_kernel(
        int *parents,
        int total_pixels)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_pixels)
        {
            return;
        }

        const int label = parents[idx];
        if (label > 0)
        {
            parents[idx] = find_root_label(parents, label);
        }
    }

    __global__ void final_labels_from_blocks_kernel(
        const uint8_t *binary,
        const int *parents,
        int *labels,
        int width,
        int height)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
        {
            return;
        }

        const int idx = y * width + x;
        if (binary[idx] == 0)
        {
            labels[idx] = 0;
            return;
        }

        const int block_x = x / 2;
        const int block_y = y / 2;
        const int top_left = block_top_left_index(block_x, block_y, width);
        const int block_label = parents[top_left];
        labels[idx] = block_label > 0 ? find_root_label(parents, top_left + 1) : 0;
    }

    __global__ void count_component_area_kernel(
        const int *labels,
        int *areas,
        int total_pixels)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_pixels)
        {
            return;
        }

        const int label = labels[idx];
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
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_pixels)
        {
            return;
        }

        const int label = labels[idx];
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
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_pixels)
        {
            return;
        }

        const int label = raw_labels[idx];
        if (label == 0)
        {
            compact_labels[idx] = 0;
            return;
        }

        int left = 0;
        int right = num_unique - 1;

        while (left <= right)
        {
            const int mid = (left + right) / 2;
            const int mid_label = unique_labels[mid];

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

    __global__ void extract_piece_mask_kernel(
        const int *labels,
        uint8_t *mask,
        int target_label,
        int total_pixels)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_pixels)
        {
            return;
        }

        mask[idx] = (labels[idx] == target_label) ? 255 : 0;
    }

    dim3 block_grid(int width, int height, dim3 block)
    {
        return dim3(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);
    }
}

void init_labels_buf_cuda_device(
    const uint8_t *d_binary,
    int *d_labels,
    int width,
    int height)
{
    const dim3 block(16, 16);
    const dim3 grid = block_grid((width + 1) / 2, (height + 1) / 2, block);
    init_block_labels_kernel<<<grid, block>>>(d_binary, d_labels, width, height);
}

void union_labels_buf_cuda_device(
    const uint8_t *d_binary,
    int *d_labels,
    int width,
    int height)
{
    const dim3 block(16, 16);
    const dim3 grid = block_grid((width + 1) / 2, (height + 1) / 2, block);
    merge_block_labels_kernel<<<grid, block>>>(d_binary, d_labels, width, height);
}

void compress_labels_buf_cuda_device(
    int *d_labels,
    int total_pixels)
{
    const int block = 256;
    const int grid = (total_pixels + block - 1) / block;
    compress_block_labels_kernel<<<grid, block>>>(d_labels, total_pixels);
}

void count_component_area_buf_cuda_device(
    const int *d_labels,
    int *d_areas,
    int total_pixels)
{
    const int block = 256;
    const int grid = (total_pixels + block - 1) / block;
    count_component_area_kernel<<<grid, block>>>(d_labels, d_areas, total_pixels);
}

void filter_small_components_buf_cuda_device(
    int *d_labels,
    const int *d_areas,
    int min_area,
    int total_pixels)
{
    const int block = 256;
    const int grid = (total_pixels + block - 1) / block;
    filter_small_components_kernel<<<grid, block>>>(d_labels, d_areas, min_area, total_pixels);
}

void connected_components_buf_cuda_device_raw(
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
    thrust::device_vector<int> d_parents(static_cast<size_t>(total_pixels));
    int *d_parent_ptr = thrust::raw_pointer_cast(d_parents.data());

    init_labels_buf_cuda_device(d_binary, d_parent_ptr, width, height);
    union_labels_buf_cuda_device(d_binary, d_parent_ptr, width, height);
    compress_labels_buf_cuda_device(d_parent_ptr, total_pixels);

    const dim3 block(16, 16);
    const dim3 grid = block_grid(width, height, block);
    final_labels_from_blocks_kernel<<<grid, block>>>(
        d_binary,
        d_parent_ptr,
        d_labels,
        width,
        height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

int connected_components_buf_cuda_device(
    const uint8_t *d_binary,
    int *d_compact_labels,
    int width,
    int height,
    int min_area)
{
    connected_components_buf_cuda_device_raw(
        d_binary,
        d_compact_labels,
        nullptr,
        nullptr,
        width,
        height,
        min_area,
        width * height + 1);

    return 0;
}

int compact_labels_buf_cuda_device(
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

    const int num_unique = static_cast<int>(unique_labels.size());

    const int block = 256;
    const int grid = (total_pixels + block - 1) / block;
    compact_labels_kernel<<<grid, block>>>(
        d_labels,
        thrust::raw_pointer_cast(unique_labels.data()),
        d_compact_labels,
        num_unique,
        total_pixels);
    CUDA_CHECK(cudaDeviceSynchronize());

    return num_unique;
}

void build_regions_from_labels_buf_cuda(
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
                  {
                      return a.y < b.y;
                  }
                  return a.x < b.x;
              });

    regions.erase(
        std::remove_if(regions.begin(), regions.end(), [min_area](const Region &region)
                       { return region.area < min_area; }),
        regions.end());
}

void extract_piece_mask_buf_cuda_device(
    const int *d_labels,
    uint8_t *d_mask,
    int target_label,
    int total_pixels)
{
    const int block = 256;
    const int grid = (total_pixels + block - 1) / block;
    extract_piece_mask_kernel<<<grid, block>>>(d_labels, d_mask, target_label, total_pixels);
}
