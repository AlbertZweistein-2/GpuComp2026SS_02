#pragma once

#include <utility>
#include <vector>

#include "types.hpp"

// inherently sequential, not parallelized
void trace_contour(const std::vector<uint8_t> &boundary, int width, int height, CoordinateVector<int> &result);

// inherently sequential, not parallelized
void simplify_chain_approx(CoordinateVector<int> &contour);

// sequential tracing + parallel coordinate swap via thrust::transform
void find_contour_chain_approx_simple(const std::vector<uint8_t> &boundary, int width, int height, CoordinateVector<int> &result);

// parallel moving average via custom CUDA kernel with shared memory
void smooth_contour(CoordinateVector<int> &points, int window);

// parallel min/max + transform_reduce via thrust
void enclosing_circle_approx(const CoordinateVector<int> &points, Coordinate<float> &center, float &radius);

// custom CUDA kernel, one thread per contour point
void radial_signal(const CoordinateVector<int> &points, Coordinate<float> center, Signal &signal);
