#pragma once

#include <cstdint>

#include "types.hpp"

void get_piece_boundary_mask_from_labels_cuda(
    const int* d_labels,
    const Region& region,
    uint8_t* d_boundary,
    int image_width,
    int image_height,
    ImageU8& boundary,
    Coordinate<int>& offset
);
