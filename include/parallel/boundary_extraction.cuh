#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include <thrust/device_vector.h>

#include "types.hpp"

struct PieceBoundaryMask
{
    int width;
    int height;
    std::size_t data_offset;
    Coordinate<int> image_offset;
};

struct BoundaryBox
{
    int label;
    int x0;
    int y0;
    int width;
    int height;
    std::size_t data_offset;
};

struct BoundaryExtractionPlan
{
    std::vector<BoundaryBox> boxes;
    std::size_t total_boundary_pixels = 0;
    std::size_t max_boundary_pixels = 0;
};

struct BoundaryExtractionCudaScratch
{
    thrust::device_vector<BoundaryBox> boxes;
    thrust::device_vector<uint8_t> boundary_data;
};

BoundaryExtractionPlan build_boundary_extraction_plan(
    const std::vector<Region> &regions,
    int image_width,
    int image_height,
    std::vector<PieceBoundaryMask> &boundaries
);

void allocate_boundary_extraction_cuda_scratch(
    const BoundaryExtractionPlan &plan,
    BoundaryExtractionCudaScratch &scratch
);

// Extracts cropped boundary masks for all regions in one batched CUDA launch.
// boundary_data is a flat host buffer; PieceBoundaryMask::data_offset selects
// the mask belonging to each region.
void extract_piece_boundary_masks_from_labels_cuda(
    const int *d_labels,
    const BoundaryExtractionPlan &plan,
    BoundaryExtractionCudaScratch &scratch,
    int image_width,
    int image_height,
    std::vector<uint8_t> &boundary_data);
