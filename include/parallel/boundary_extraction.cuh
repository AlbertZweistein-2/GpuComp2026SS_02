#pragma once

#include <cstdint>

void extract_piece_mask_cuda_device(
    const int* d_labels,
    uint8_t* d_mask,
    int target_label,
    int total_pixels
);

void get_external_boundary_mask_cuda_device(
    const uint8_t* d_input,
    uint8_t* d_boundary,
    int width,
    int height
);
