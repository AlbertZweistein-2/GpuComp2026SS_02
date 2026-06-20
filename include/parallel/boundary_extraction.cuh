#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include "types.hpp"

struct PieceBoundaryMask
{
    int width;
    int height;
    std::size_t data_offset;
    Coordinate<int> image_offset;
};

// Extracts cropped boundary masks for all regions in one batched CUDA launch.
// boundary_data is a flat host buffer; PieceBoundaryMask::data_offset selects
// the mask belonging to each region.
void extract_piece_boundary_masks_from_labels_cuda(
    const int *d_labels,
    const std::vector<Region> &regions,
    int image_width,
    int image_height,
    std::vector<PieceBoundaryMask> &boundaries,
    std::vector<uint8_t> &boundary_data);
