
#include <cstdint>
#include <vector>
#include <utility>
#include <map>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/binary_search.h>

// Shared datatypes
// ______________________________________________________________________________________________


/*
Binary or grayscale image stored as 2D vector
0 = background, 255 = puzzle piece
*/
using ImageU8 = std::vector<std::vector<uint8_t>>;

/*
Integer label image stored as 2D vector
0 = background
1, 2, .. = connected component labels
*/
using ImageI32 = std::vector<std::vector<int>>;

// Stores data for each detected connected component. 1 Region = 1 puzzle piece
struct Region {
    int label;  // component label in the label image
    int area;   // number of foreground picels
    int x;      // left coordinate of bounding box
    int y;      // top coordinate of bb
    int width;  // bb width
    int height; // bb height
};


// CUDA kernels
// ______________________________________________________________________________________________

/*
Each CUDA thread computes one output pixel. Erosion keeps a pixel as foreground only if active kernel positions 
around it are also foreground. Pixels outside the image are treated as background.
*/

__global__ void erode_kernel(const uint8_t* input, const uint8_t* kernel,
    uint8_t* output, int width, int height, int kw, int kh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return; // threads outside the image do no work

    int pad_w = kw / 2;
    int pad_h = kh / 2;

    bool all_foreground = true;

    // Check all kernel positions around the current pixel
    for (int ky = 0; ky < kh; ++ky) {
        for (int kx = 0; kx < kw; ++kx) {
            //  ignore inactive kernel cells
            if (kernel[ky * kw + kx] != 1) continue;

            int ix = x + kx - pad_w;
            int iy = y + ky - pad_h;
            
            // Out of bounds pixels are treated as bacckground
            uint8_t pixel = 0;

            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                pixel = input[iy * width + ix];
            }

            // if a neighbour is not foreground, the output pixel is eroded away
            if (pixel != 255) {
                all_foreground = false;
                break;
            }
        }

        if (!all_foreground) break;
    }

    output[y * width + x] = all_foreground ? 255 : 0;
}

/*
Each thread computed one output pixel. 
Dilation keeps pixel as foreground if at least one kernel positon overlaps a foreground pixel.
This expands foregorund regions and can fill small gaps between enighbouring pixels.
*/
__global__ void dilate_kernel(const uint8_t* input, const uint8_t* kernel, uint8_t* output,
    int width, int height, int kw, int kh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return; // threads outside the image do no work

    int pad_w = kw / 2;
    int pad_h = kh / 2;

    bool any_foreground = false;

    for (int ky = 0; ky < kh; ++ky) {
        for (int kx = 0; kx < kw; ++kx) {
            if (kernel[ky * kw + kx] != 1) continue;

            int ix = x + kx - pad_w;
            int iy = y + ky - pad_h;

            // If any neighbouring foreground pixel is found, the output pixel becomes foreground.
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                if (input[iy * width + ix] == 255) {
                    any_foreground = true;
                    break;
                }
            }
        }
        if (any_foreground) break;
    }

    output[y * width + x] = any_foreground ? 255 : 0;
}

/*
Foreground pixels are assigned unique initial label based on the position in image.
Done until all pixels belonging to same connected component has the same label.
*/
__global__ void init_labels_kernel(const uint8_t* binary,
    int* labels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Each foreground pixel start with own unqiue label,
    // Background is unlabeled
    if (binary[idx] > 0)
        labels[idx] = idx + 1;
    else
        labels[idx] = 0;
}

/*
Each thread works on one foreground pixel.

Kernel examines all 8 neighbouring pixels and finds the smallest connected label in the local neighbourhood.
If the smaller label is found, the current pixel uses that label.
Repeated kernel launches propogates this label throughout each connected component 
until all picels share the same label in the component.
*/

__global__ void propagate_labels_kernel(const uint8_t* binary,
    int* labels, int* changed, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // background pixels are skipped
    if (binary[idx] == 0) return;

    int current = labels[idx];
    int min_label = current;

    // search the 8 connected neighbourhood for a smaller label
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue; // skip the current pixel

            int nx = x + dx;
            int ny = y + dy;

            // Ignore labels outside the image
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            int nidx = ny * width + nx;

            // consider only labeled foreground neighbours
            if (binary[nidx] > 0 && labels[nidx] > 0) {
                min_label = min(min_label, labels[nidx]);
            }
        }
    }

    // if smaller neighbouring label is found, update current pixel and mark that a change occurred.
    if (min_label < current) {
        labels[idx] = min_label;
        *changed = 1;
    }
}

/*
kernel to extract one connected component as a binary mask.
one thread handles one pixel in label image.

Pixels whose label matches target label become foreground. All other pixels become backgrond
*/
__global__ void extract_piece_mask_kernel(const int* labels, uint8_t* mask, 
    int target_label, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) return;

    mask[idx] = (labels[idx] == target_label) ? 255 : 0;
}

/*
Kernel to count the area of each connected component.
Since many pixel can belong to the same compoentn at the same time, atomicAdd is used to incremenet area
*/
__global__ void count_component_area_kernel(const int* labels, int* areas, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) return;

    int label = labels[idx];

    if (label > 0) {
        atomicAdd(&areas[label], 1);
    }
}

/*
kernel to remove small connected components.
If pixel belongs to a component whose area is smaller than min_area, its label is set to 0 and becomes background
*/
__global__ void filter_small_components_kernel(int* labels, const int* areas, 
    int min_area, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) return;

    int label = labels[idx];

    if (label > 0 && areas[label] < min_area) {
        labels[idx] = 0;
    }
}

/*
kernel for label compaction.
connected-component labeling produces sparse labels based on pixel indices:
for example: 8, 17, 24, ...

This kernel converts them into consecutive labels:
1, 2, 3, ...

Done by searching a sorted list of noque labels. Binary search is used so that each thread
can efficiently find the compact label corresponding to its original component label.
*/

__global__ void compact_labels_kernel(const int* raw_labels, const int* unique_labels,
    int* compact_labels, int num_unique, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) return;

    int label = raw_labels[idx];

    // Background remains background
    if (label == 0) {
        compact_labels[idx] = 0;
        return;
    }

    int left = 0;
    int right = num_unique - 1;

    // Binary search through sorted unqiue label list
    while (left <= right) {
        int mid = (left + right) / 2;
        int mid_label = unique_labels[mid];

        if (mid_label == label) {
            // label becomes consecutive:
            // unique_labels[0] -> 1, ...
            compact_labels[idx] = mid + 1;
            return;
        }

        if (mid_label < label) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    compact_labels[idx] = 0;
}

/*
kernel for external boundary extraction. 
A pixel is classified as a boundary pixel if at least one its 8 neighbouring pixel is background, 
or pixel lies on the image border.

The output is a binary boundary image where 255 = boundary pixel, 0 = non boundary pixel
*/
__global__ void boundary_kernel(const uint8_t* input, uint8_t* boundary,
    int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    if (input[idx] == 0) {
        boundary[idx] = 0;
        return;
    }

    bool is_boundary = false;

    // Examine the 8 connected neighbourhood
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue; // skip current pixel

            int nx = x + dx;
            int ny = y + dy;

            // Pixels touching the image border are considered boundary pixels
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                is_boundary = true;
                continue;
            }

            int nidx = ny * width + nx;

            // If any neighbour is background, the current pixel belongs to the external boundary
            if (input[nidx] == 0) {
                is_boundary = true;
            }
        }
    }

    boundary[idx] = is_boundary ? 255 : 0;
}

// Wrapper functions
// ______________________________________________________________________________________________

/*
since image type is a 2D vector, the image and kernel are first flattened into 1D arrays
before being copied into device memory,
kernel computed one output pixel per thread
*/
ImageU8 erode_cuda(const ImageU8& image, const ImageU8& kernel, int iterations) 
{
    const int height = image.size();
    const int width = image[0].size();

    const int kh = kernel.size();
    const int kw = kernel[0].size();

    // flatten 2D inut image and kernel into 1D arrays
    // CUDA memory is linear, so using flat arrays makes indexing simpler
    std::vector<uint8_t> h_input(width * height);
    std::vector<uint8_t> h_output(width * height);
    std::vector<uint8_t> h_kernel(kw * kh);

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            h_input[y * width + x] = image[y][x];

    for (int y = 0; y < kh; ++y)
        for (int x = 0; x < kw; ++x)
            h_kernel[y * kw + x] = kernel[y][x];

    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    uint8_t* d_kernel = nullptr;

    cudaMalloc(&d_input, width * height * sizeof(uint8_t));
    cudaMalloc(&d_output, width * height * sizeof(uint8_t));
    cudaMalloc(&d_kernel, kw * kh * sizeof(uint8_t));

    cudaMemcpy(d_input, h_input.data(), width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), kw * kh * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,(height + block.y - 1) / block.y);

    // Apply erosion repeatedly if multiple iterations are requested, after each iteration, input and output buffers are swapper, 
    // so the next iteration uses the previous result.
    for (int iter = 0; iter < iterations; ++iter) {
        erode_kernel<<<grid, block>>>(d_input, d_kernel, d_output, width, height, kw, kh);

        cudaDeviceSynchronize();

        std::swap(d_input, d_output);
    }

    cudaMemcpy(h_output.data(), d_input, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    // convert flat output array back into 2D format
    ImageU8 result(height, std::vector<uint8_t>(width, 0));

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            result[y][x] = h_output[y * width + x];

    return result;
} 

/*
Image and kernel flattened again into 1D array.
*/

ImageU8 dilate_cuda(const ImageU8& image, const ImageU8& kernel, int iterations) {
    const int height = image.size();
    const int width = image[0].size();

    const int kh = kernel.size();
    const int kw = kernel[0].size();

    std::vector<uint8_t> h_input(width * height);
    std::vector<uint8_t> h_output(width * height);
    std::vector<uint8_t> h_kernel(kw * kh);

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            h_input[y * width + x] = image[y][x];

    for (int y = 0; y < kh; ++y)
        for (int x = 0; x < kw; ++x)
            h_kernel[y * kw + x] = kernel[y][x];

    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    uint8_t* d_kernel = nullptr;

    cudaMalloc(&d_input, width * height * sizeof(uint8_t));
    cudaMalloc(&d_output, width * height * sizeof(uint8_t));
    cudaMalloc(&d_kernel, kw * kh * sizeof(uint8_t));

    cudaMemcpy(d_input, h_input.data(), width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), kw * kh * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,(height + block.y - 1) / block.y);

    // Apply dilation repeatedly if multiple iterations are requested
    for (int iter = 0; iter < iterations; ++iter) {
        dilate_kernel<<<grid, block>>>(d_input,d_kernel, d_output, width, height, kw, kh);

        cudaDeviceSynchronize();
        std::swap(d_input, d_output);
    }

    cudaMemcpy(h_output.data(), d_input, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    ImageU8 result(height, std::vector<uint8_t>(width, 0));

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            result[y][x] = h_output[y * width + x];

    return result;
}

/*
This removed small foreground noise while preserving larger puzzle piece regions.

Step 1: erode() removes small foreground artifacts
Step 2: dilate() restores size of the remaining foreground objects
*/
ImageU8 morphological_open_cuda(const ImageU8& image, const ImageU8& kernel, int iterations) {
    ImageU8 eroded = erode_cuda(image, kernel, iterations);
    ImageU8 opened = dilate_cuda(eroded, kernel, iterations);
    return opened;
}

/*
performs parallel connected-component labelling stage.

Each foreground pixel is initially assigned a unqiue label.
The propogate_labels_kernel is then launched repeatedly until no label changes occur.
During each iteration, pixel adops the smallest label found in their 8 connected neighbourhood.

At convergence, all pixels from the same connected component share a common label.

The labels returned by this function are raw labels, meaning that they are not uniquely consective yet (like not 1, 2, 3..)
*/

ImageI32 connected_components_cuda_raw(const ImageU8& binary) {
    const int height = binary.size();
    const int width = binary[0].size();
    const int total = width * height;

    std::vector<uint8_t> h_binary(total);

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            h_binary[y * width + x] = binary[y][x];

    uint8_t* d_binary = nullptr;
    int* d_labels = nullptr;
    int* d_changed = nullptr;

    cudaMalloc(&d_binary, total * sizeof(uint8_t));
    cudaMalloc(&d_labels, total * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));

    cudaMemcpy(d_binary, h_binary.data(), total * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,(height + block.y - 1) / block.y);

    init_labels_kernel<<<grid, block>>>(d_binary, d_labels, width, height);
    cudaDeviceSynchronize();

    // iteratively propogate labels until convergence.
    int h_changed = 1;
    int iterations = 0;
    const int max_iterations = width + height; // prevent infinite loops

    while (h_changed && iterations < max_iterations) {
        h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

        // propagate smallest neighbouring label
        propagate_labels_kernel<<<grid, block>>>(d_binary, d_labels, d_changed, width, height);

        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);

        iterations++;
    }

    std::vector<int> h_labels(total);
    cudaMemcpy(h_labels.data(), d_labels, total * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_binary);
    cudaFree(d_labels);
    cudaFree(d_changed);

    ImageI32 labels(height, std::vector<int>(width, 0));

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            labels[y][x] = h_labels[y * width + x];

    return labels;
}

/*
Removes connected components whose area is smaller than min area

input labels are raw labels from connected_components_cuda_raw(),
Function counts the number of pixels belonging to each label on GPU, then launches 
another kernel to reove lavels who area is small.
*/
ImageI32 filter_small_components_cuda_raw(const ImageI32& raw_labels, int min_area) {
    const int height = raw_labels.size();
    const int width = raw_labels[0].size();
    const int total = width * height;

    std::vector<int> h_labels(total);

    int max_label = 0;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int label = raw_labels[y][x];
            h_labels[y * width + x] = label;
            max_label = std::max(max_label, label);
        }
    }

    int* d_labels = nullptr;
    int* d_areas = nullptr;

    cudaMalloc(&d_labels, total * sizeof(int));
    cudaMalloc(&d_areas, (max_label + 1) * sizeof(int));

    cudaMemcpy(d_labels, h_labels.data(), total * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(d_areas, 0, (max_label + 1) * sizeof(int));

    int block = 256;
    int grid = (total + block - 1) / block;

    // count number of pixels for each component in parallel.

    count_component_area_kernel<<<grid, block>>>(d_labels, d_areas, total);

    cudaDeviceSynchronize();

    // remove components whose area is smaller than min_area
    filter_small_components_kernel<<<grid, block>>>(d_labels, d_areas, min_area, total);

    cudaDeviceSynchronize();

    cudaMemcpy(h_labels.data(), d_labels, total * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_labels);
    cudaFree(d_areas);

    ImageI32 filtered(height, std::vector<int>(width, 0));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            filtered[y][x] = h_labels[y * width + x];
        }
    }

    return filtered;
}

/*
Compacts sparse connected-component labels into consecutive labels.

The raw connected-component labels are based on pixel indices, so they may
look like 8, 17, 24, ... instead of 1, 2, 3, ...

This function uses Thrust to:
- copy all labels to GPU
- sort them
- removes background label 0
- keep only unqiue component labels
- remap each raw label to complact consecutive label

Example:
raw_labels: 0, 8, 8, 24
compact labels: 0, 1, 1, 2
*/
ImageI32 compact_labels_thrust(const ImageI32& raw_labels) {
    const int height = raw_labels.size();
    const int width = raw_labels[0].size();
    const int total = width * height;

    std::vector<int> h_raw(total);

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            h_raw[y * width + x] = raw_labels[y][x];

    // copy raw labels to the GPU using a thrust device vector
    thrust::device_vector<int> d_raw = h_raw;

    // create a copy that will be sorted and reduced to unqiue labels
    thrust::device_vector<int> d_unique = d_raw;

    // sort labels so equal component labels become adjacent.
    thrust::sort(d_unique.begin(), d_unique.end());

    // remove background labels
    d_unique.erase(thrust::remove(d_unique.begin(), d_unique.end(), 0), d_unique.end());
    // keep only one copy of each remaining component label
    auto unique_end = thrust::unique( d_unique.begin(), d_unique.end());

    d_unique.erase(unique_end, d_unique.end());

    int num_unique = d_unique.size();

    thrust::device_vector<int> d_compact(total, 0);

    int block = 256;
    int grid = (total + block - 1) / block;

    // remap every raw label to its compact label in parallel.
    compact_labels_kernel<<<grid, block>>>(thrust::raw_pointer_cast(d_raw.data()),thrust::raw_pointer_cast(d_unique.data()),
        thrust::raw_pointer_cast(d_compact.data()), num_unique, total);

    cudaDeviceSynchronize();

    // copy compact labels back to the cpu
    thrust::host_vector<int> h_compact = d_compact;

    ImageI32 compact(height, std::vector<int>(width, 0));

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            compact[y][x] = h_compact[y * width + x];

    return compact;
}

/*
connected-components pipeline

 1. connected_components_cuda_raw()
    assigns raw component labels using iterative GPU label propagation.

 2. filter_small_components_cuda_raw()
    removes small noisy regions using GPU area counting.

 3. compact_labels_thrust()
    converts sparse raw labels into consecutive labels.

 4. CPU region metadata extraction
    computes area and bounding boxes for each detected component.

 The returned label image uses:
 0        = background
 1,2,3... = detected puzzle pieces
*/

std::pair<ImageI32, std::vector<Region>>
connected_components_cuda(const ImageU8& binary, int min_area) {
    ImageI32 raw_labels = connected_components_cuda_raw(binary);
    raw_labels = filter_small_components_cuda_raw(raw_labels, min_area);

    ImageI32 compact_labels = compact_labels_thrust(raw_labels);

    const int height = compact_labels.size();
    const int width = compact_labels[0].size();

    std::map<int, Region> region_map; // build region list used later

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int label = compact_labels[y][x];

            if (label == 0) continue;

            // first time seeing label? create new region
            if (region_map.find(label) == region_map.end()) {
                region_map[label] = {label, 0, x, y, 1, 1};
            }

            Region& r = region_map[label];

            r.area++; // count pixels and update bounding box
            int min_x = std::min(r.x, x);
            int min_y = std::min(r.y, y);
            int max_x = std::max(r.x + r.width - 1, x);
            int max_y = std::max(r.y + r.height - 1, y);

            r.x = min_x;
            r.y = min_y;
            r.width = max_x - min_x + 1;
            r.height = max_y - min_y + 1;
        }
    }

    std::vector<Region> regions;

    for (auto& [label, region] : region_map) {
        regions.push_back(region);
    }

    return {compact_labels, regions};
}

/*
Extracts a single puzzle piece from a compact label image

The input labels image contains:
0       = background
1, 2, 3, .. = detected connected components

this function creates a binaru mask for one selected component. 
Pixels with target_label become foreground, and all other pixels become background
*/

ImageU8 extract_piece_mask_cuda(const ImageI32& labels, int target_label) {
    const int height = labels.size();
    const int width = labels[0].size();
    const int total = width * height;

    std::vector<int> h_labels(total);
    std::vector<uint8_t> h_mask(total);

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            h_labels[y * width + x] = labels[y][x];

    int* d_labels = nullptr;
    uint8_t* d_mask = nullptr;

    cudaMalloc(&d_labels, total * sizeof(int));
    cudaMalloc(&d_mask, total * sizeof(uint8_t));

    cudaMemcpy(d_labels, h_labels.data(), total * sizeof(int), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (total + block - 1) / block;

    // one thread checks on pixel label
    extract_piece_mask_kernel<<<grid, block>>>(d_labels, d_mask, target_label, total);

    cudaDeviceSynchronize();

    cudaMemcpy(h_mask.data(), d_mask, total * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_labels);
    cudaFree(d_mask);
    
    // copy binary pice mask back to the CPU
    ImageU8 mask(height, std::vector<uint8_t>(width, 0));

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            mask[y][x] = h_mask[y * width + x];

    return mask;
}

/* 
Extracts boundary pixels pf a binary piece mask on the GPU

The input is expected to be a single puzzle piece mask:
0 = backgeound
255 = foreground piece

A foreground pixel becomes a boundary pixel if at least one of its 8 neighbours is background or outside image.
resulting boundary image is used a =s input for contour tracing.
*/
ImageU8 get_external_boundary_mask_cuda(const ImageU8& input) {
    const int height = input.size();
    const int width = input[0].size();
    const int total = width * height;

    std::vector<uint8_t> h_input(total);
    std::vector<uint8_t> h_boundary(total);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_input[y * width + x] = input[y][x] > 0 ? 255 : 0;
        }
    }

    uint8_t* d_input = nullptr;
    uint8_t* d_boundary = nullptr;

    cudaMalloc(&d_input, total * sizeof(uint8_t));
    cudaMalloc(&d_boundary, total * sizeof(uint8_t));

    cudaMemcpy(d_input, h_input.data(), total * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    boundary_kernel<<<grid, block>>>(d_input, d_boundary, width, height);

    cudaDeviceSynchronize();

    cudaMemcpy(h_boundary.data(), d_boundary, total * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_boundary);

    ImageU8 boundary(height, std::vector<uint8_t>(width, 0));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            boundary[y][x] = h_boundary[y * width + x];
        }
    }

    return boundary;
}