#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "types.hpp"

CoordinateVector<int> trace_contour(const std::vector<uint8_t> &boundary, int width, int height);

CoordinateVector<int> simplify_chain_approx(const CoordinateVector<int> &contour);

CoordinateVector<int> find_contour_chain_approx_simple(const std::vector<uint8_t>& boundary, int width, int height);

CoordinateVector<int> smooth_contour(const CoordinateVector<int>& points, int window);

std::pair<Coordinate<float>, float> enclosing_circle_approx(const CoordinateVector<int>& points);

Signal radial_signal(const CoordinateVector<int>& points, Coordinate<float> center);
