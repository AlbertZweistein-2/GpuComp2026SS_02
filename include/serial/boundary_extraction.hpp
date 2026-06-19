#pragma once

#include "types.hpp"

void get_piece_boundary_mask_from_labels(
    const ImageI32& labels,
    const Region& region,
    ImageU8& boundary,
    Coordinate<int>& offset
);
