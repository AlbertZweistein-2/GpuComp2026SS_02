#pragma once

// ----------------------------------------------------------------------
// STANDARD LIBRARY INCLUDES
#include <cstdint>
#include <cstddef>
#include <vector>
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// GPU SPECIFIC INCLUDES
#include <thrust/device_vector.h>
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// PROJECT INCLUDES
#include "types.hpp"
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// HOST-SIDE BOUNDARY METADATA
// Describes one cropped boundary mask in the flat host boundary buffer.
struct PieceBoundaryMask
{
    int width;
    int height;
    std::size_t data_offset;
    Coordinate<int> image_offset;
};

// Describes one region crop copied to device so the boundary kernel can process
// all pieces in a single batched launch.
struct BoundaryBox
{
    int label;
    int x0;
    int y0;
    int width;
    int height;
    std::size_t data_offset;
};

// Precomputed launch/packing data for boundary extraction. The largest cropped
// box determines the x-grid size; smaller boxes exit early in the kernel.
struct BoundaryExtractionPlan
{
    std::vector<BoundaryBox> boxes;
    std::size_t total_boundary_pixels = 0;
    std::size_t max_boundary_pixels = 0;
};
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// CUDA SCRATCH BUFFERS
struct BoundaryExtractionCudaScratch
{
    // Device copy of all per-piece crop boxes.
    thrust::device_vector<BoundaryBox> boxes;
    // Flat device buffer containing all cropped boundary masks back-to-back.
    thrust::device_vector<uint8_t> boundary_data;
};
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// PUBLIC BOUNDARY EXTRACTION API
// Build host-side crop metadata from connected-component regions.
BoundaryExtractionPlan build_boundary_extraction_plan(
    const std::vector<Region> &regions,
    int image_width,
    int image_height,
    std::vector<PieceBoundaryMask> &boundaries
);

// Allocate and upload the metadata needed by the batched boundary kernel.
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
// ----------------------------------------------------------------------
