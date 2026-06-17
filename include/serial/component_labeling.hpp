#pragma once

#include <utility>
#include <vector>

#include "types.hpp"

void connected_components(
    const ImageU8& binary,
    int min_area,
    ImageI32& labels,
    std::vector<Region>& regions
);

void extract_piece_mask(
    const ImageI32& labels,
    int target_label,
    ImageU8& mask
);