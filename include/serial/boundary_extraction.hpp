#pragma once

// #include <cstdint>
// #include <utility>
// #include <vector>
#include "types.hpp"

void get_external_boundary_mask(
    const ImageU8& input,
    ImageU8 &boundary
);

void get_piece_boundary_mask_from_labels(
    const ImageI32& labels,
    const Region& region,
    ImageU8& boundary,
    Coordinate<int>& offset
);
