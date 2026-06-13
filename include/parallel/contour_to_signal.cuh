#pragma once

#include <utility>
#include <vector>

#include "types.hpp"

// inherently sequential, not parallelized
CoordinateVector<int> trace_contour(const std::vector<uint8_t> &boundary, int width, int height);

// inherently sequential, not parallelized
CoordinateVector<int> simplify_chain_approx(const CoordinateVector<int> &contour);

// sequential tracing, parallel coordinate swap using thrust
CoordinateVector<int> find_contour_chain_approx_simple(const std::vector<uint8_t> &boundary, int width, int height);

// parallel moving average using thrust
CoordinateVector<int> smooth_contour(const CoordinateVector<int> &points, int window);

// parallel min/max and transform_reduce using thrust
std::pair<Coordinate<float>, float> enclosing_circle_approx(const CoordinateVector<int> &points);

// custom CUDA kernel, one thread per contour point
Signal radial_signal(const CoordinateVector<int> &points, Coordinate<float> center);
