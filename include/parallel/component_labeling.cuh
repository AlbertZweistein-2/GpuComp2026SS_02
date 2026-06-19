#pragma once

#include <cstdint>
#include <vector>

#include "types.hpp"

void init_labels_cuda_device(
    const uint8_t *d_binary,
    int *d_labels,
    int width,
    int height);

void propagate_labels_cuda_device(
    const uint8_t *d_binary,
    int *d_labels,
    int *d_changed,
    int width,
    int height);

void count_component_area_cuda_device(
    const int *d_labels,
    int *d_areas,
    int total_pixels);

void filter_small_components_cuda_device(
    int *d_labels,
    const int *d_areas,
    int min_area,
    int total_pixels);

void connected_components_cuda_device_raw(
    const uint8_t *d_binary,
    int *d_labels,
    int *d_changed,
    int *d_areas,
    int width,
    int height,
    int min_area,
    int max_label_count);

int connected_components_cuda_device(
    const uint8_t *d_binary,
    int *d_compact_labels,
    int width,
    int height,
    int min_area);

int compact_labels_cuda_device(
    const int *d_labels,
    int *d_compact_labels,
    int total_pixels);

std::vector<Region> build_regions_from_labels_cuda(
    const int *d_compact_labels,
    int width,
    int height,
    int num_components);



// extracts the binary mask for a single puzzle piece by label
// sets pixels to 255 where label matches, 0 elsewhere
void extract_piece_mask_cuda_device(
    const int *d_labels,
    uint8_t *d_mask,
    int target_label,
    int total_pixels);
