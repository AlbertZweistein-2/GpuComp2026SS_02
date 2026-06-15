#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "types.hpp"

// CUDA pipeline extension points.
//
// Implement only the functions you want to select with the matching
// PIPELINE_USE_CUDA_* compile definition. For example:
//
//   -DPIPELINE_USE_CUDA_PREPROCESSING
//
// requires preprocess_cuda(...) to be provided by one of the linked CUDA
// translation units. Stages without a PIPELINE_USE_CUDA_* flag keep using the
// serial implementation.

void preprocess_cuda(
    const uint8_t* rgb,
    int width,
    int height,
    int ksize = 5,
    float sigma = 1.0f,
    ImageU8& result
);

void morphological_open_cuda(
    const ImageU8& image,
    const ImageU8& kernel,
    ImageU8& result,
    int iterations = 1
);

void get_external_boundary_mask_cuda(
    const ImageU8& input,
    ImageU8& boundary
);

std::vector<Coordinate> find_contour_chain_approx_simple_cuda(
    const std::vector<uint8_t>& boundary,
    int width,
    int height
);

std::vector<Coordinate> smooth_contour_cuda(
    const std::vector<Coordinate>& points,
    int window
);

std::pair<CoordinateFloat, float> enclosing_circle_approx_cuda(
    const std::vector<Coordinate>& points
);

std::vector<float> radial_signal_cuda(
    const std::vector<Coordinate>& points,
    CoordinateFloat center
);

std::vector<int> find_triangular_peaks_cuda(
    const std::vector<float>& signal,
    int min_distance,
    float min_prominence,
    int smoothing_window,
    float min_sharpness
);
