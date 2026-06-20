#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <thrust/device_vector.h>

#include "types.hpp"

struct PieceBoundaryMask
{
    int width;
    int height;
    std::size_t data_offset;
    Coordinate<int> image_offset;
};

struct PieceBoundaryPointBatch
{
    std::vector<PieceBoundaryMask> boundaries;
    thrust::device_vector<int> point_offsets;
    thrust::device_vector<int> point_counts;
    thrust::device_vector<Coordinate<int>> points;
};

void extract_piece_boundary_points_from_labels_cuda(
    const int *d_labels,
    const std::vector<Region> &regions,
    int image_width,
    int image_height,
    PieceBoundaryPointBatch &batch);

void copy_boundary_points_to_masks_host(
    const PieceBoundaryPointBatch &batch,
    std::vector<uint8_t> &boundary_data);
