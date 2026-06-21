#pragma once

#include "types.hpp"

// Builds a cropped binary boundary mask for one connected component label.
// `offset` maps local mask coordinates back to full-image coordinates.
void get_piece_boundary_mask_from_labels(
    const ImageI32& labels,
    const Region& region,
    ImageU8& boundary,
    Coordinate<int>& offset
);
