#pragma once

#include <cstdint>
#include <vector>

#include <thrust/device_vector.h>

#include "types.hpp"

struct BatchedContours
{
    // All piece contours packed into one contiguous host buffer.
    std::vector<Coordinate<int>> points;
    // offsets[i] and lengths[i] describe the slice for piece i.
    std::vector<int> offsets;
    std::vector<int> lengths;
    int max_length = 0;
};

struct BatchedContourCudaScratch
{
    // Flat device-side contour input and smoothed output for all pieces.
    thrust::device_vector<Coordinate<int>> points;
    thrust::device_vector<Coordinate<int>> smoothed;
    thrust::device_vector<int> offsets;
    thrust::device_vector<int> lengths;
    thrust::device_vector<Coordinate<float>> centers;
    // Flat radial signal output matching the same offsets/lengths slices.
    thrust::device_vector<float> signals;
};

// Moore neighbor tracing is CPU-side and sequential, but it gives the connected
// contour order needed by the current batched radial-signal logic.
void find_contour_chain_approx_simple_cuda(
    const uint8_t *boundary,
    int width,
    int height,
    CoordinateVector<int> &result);

void build_batched_contours(
    const std::vector<PuzzlePiece> &pieces,
    BatchedContours &batch);

void smooth_contours_batched_cuda(
    const BatchedContours &batch,
    int window,
    BatchedContourCudaScratch &scratch);

void enclosing_circle_centers_batched_cuda(
    const BatchedContours &batch,
    BatchedContourCudaScratch &scratch);

void radial_signals_batched_cuda(
    const BatchedContours &batch,
    BatchedContourCudaScratch &scratch,
    Signal &signals);
