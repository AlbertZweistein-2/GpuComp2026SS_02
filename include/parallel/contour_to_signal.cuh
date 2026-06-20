#pragma once

#include <cstdint>
#include <vector>

#include "types.hpp"

// inherently sequential, identical to serial implementation
void trace_contour_cuda(
    const std::vector<uint8_t> &boundary,
    int width,
    int height,
    CoordinateVector<int> &result);

void trace_contour_cuda(
    const uint8_t *boundary,
    int width,
    int height,
    CoordinateVector<int> &result);

// inherently sequential, identical to serial implementation
void simplify_chain_approx_cuda(CoordinateVector<int> &contour);

// sequential tracing + parallel coordinate swap via thrust::transform
void find_contour_chain_approx_simple_cuda(
    const std::vector<uint8_t> &boundary,
    int width,
    int height,
    CoordinateVector<int> &result);

void find_contour_chain_approx_simple_cuda(
    const uint8_t *boundary,
    int width,
    int height,
    CoordinateVector<int> &result);

// parallelized via custom CUDA kernel with shared memory
void smooth_contour_cuda(CoordinateVector<int> &points, int window);

// parallelized via thrust minmax + transform_reduce
void enclosing_circle_approx_cuda(
    const CoordinateVector<int> &points,
    Coordinate<float> &center,
    float &radius);

// parallelized via custom CUDA kernel, one thread per contour point
void radial_signal_cuda(
    const CoordinateVector<int> &points,
    Coordinate<float> center,
    Signal &signal);
