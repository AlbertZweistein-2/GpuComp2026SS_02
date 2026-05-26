#pragma once

#include <cstdint>
#include <utility>
#include <vector>

using ImageU8 = std::vector<std::vector<uint8_t>>;
using ImageI32 = std::vector<std::vector<int>>;

struct Region {
    int label;
    int area;
    int x;
    int y;
    int width;
    int height;
};

ImageU8 erode(
    const ImageU8& image,
    const ImageU8& kernel,
    int iterations = 1
);

ImageU8 dilate(
    const ImageU8& image,
    const ImageU8& kernel,
    int iterations = 1
);

ImageU8 morphological_open(
    const ImageU8& image,
    const ImageU8& kernel,
    int iterations = 1
);

ImageU8 get_external_boundary_mask(
    const ImageU8& input
);

std::pair<ImageI32, std::vector<Region>> connected_components(
    const ImageU8& binary,
    int min_area = 1
);

ImageU8 extract_piece_mask(const ImageI32& labels, int target_label);