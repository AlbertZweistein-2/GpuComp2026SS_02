#pragma once

#include <cstdint>
#include <vector>

#include <thrust/device_vector.h>

#include "types.hpp"

struct BatchedContours
{
    // All piece contours packed into one contiguous host buffer. Kernels use
    // offsets/lengths to recover each piece without a vector-of-vectors layout.
    std::vector<Coordinate<int>> points;
    // offsets[i] and lengths[i] describe the flat slice for piece i.
    std::vector<int> offsets;
    std::vector<int> lengths;
    // Longest contour length, used as the x dimension for batched launches.
    int max_length = 0;
};

struct BatchedContourCudaScratch
{
    // Flat device-side contour input and smoothed output for all pieces.
    thrust::device_vector<Coordinate<int>> points;
    thrust::device_vector<Coordinate<int>> smoothed;
    // Device copies of the host slice metadata.
    thrust::device_vector<int> offsets;
    thrust::device_vector<int> lengths;
    // One approximate enclosing-circle center per piece; no host copy is needed.
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

// Smooths every contour in one 2D launch and keeps the result device-resident.
void smooth_contours_batched_cuda(
    const BatchedContours &batch,
    int window,
    BatchedContourCudaScratch &scratch);

// Reduces one contour per block to the bounding-box center approximation.
void enclosing_circle_centers_batched_cuda(
    const BatchedContours &batch,
    BatchedContourCudaScratch &scratch);

// Produces device radial signals and copies the raw signal back for CPU edge
// classification.
void radial_signals_batched_cuda(
    const BatchedContours &batch,
    BatchedContourCudaScratch &scratch,
    Signal &signals);
