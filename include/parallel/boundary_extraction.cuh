#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "types.hpp"

struct PieceBoundaryMask
{
    int width;
    int height;
    std::size_t data_offset;
    Coordinate<int> image_offset;
};

void get_piece_boundary_masks_from_labels_cuda(
    const int *d_labels,
    const std::vector<Region> &regions,
    int image_width,
    int image_height,
    std::vector<uint8_t> &boundary_data,
    std::vector<PieceBoundaryMask> &boundaries);
