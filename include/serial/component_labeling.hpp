#pragma once

#include <utility>
#include <vector>

#include "types.hpp"

std::pair<ImageI32, std::vector<Region>> connected_components(
    const ImageU8& binary,
    int min_area = 1
);

ImageU8 extract_piece_mask(const ImageI32& labels, int target_label);
