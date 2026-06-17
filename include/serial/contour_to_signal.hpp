#pragma once

#include <cstdint>
#include <vector>

#include "types.hpp"

void trace_contour(const std::vector<uint8_t> &boundary, int width, int height, CoordinateVector<int>& result);

void simplify_chain_approx(CoordinateVector<int>& contour);

void find_contour_chain_approx_simple(const std::vector<uint8_t>& boundary, int width, int height, CoordinateVector<int>& result);

void smooth_contour(CoordinateVector<int>& points, int window);

void enclosing_circle_approx(const CoordinateVector<int>& points, Coordinate<float>& center, float& radius);

void radial_signal(const CoordinateVector<int>& points, Coordinate<float> center, Signal& signal);
