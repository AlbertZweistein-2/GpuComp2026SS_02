#pragma once

#include <cstdint>
#include <vector>

#include "types.hpp"

void build_regions_from_labels_cuda(
    const int *d_compact_labels,
    int width,
    int height,
    int min_area,
    std::vector<Region> &regions);

void connected_components_buf_cuda_device(
    const uint8_t *d_binary,
    int *d_labels,
    int width,
    int height);
