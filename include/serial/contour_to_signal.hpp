#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "types.hpp"

std::vector<Coordinate> trace_contour(const std::vector<uint8_t> &boundary, int width, int height);

std::vector<Coordinate> simplify_chain_approx(const std::vector<Coordinate> &contour);

std::vector<Coordinate> find_contour_chain_approx_simple(const std::vector<uint8_t>& boundary, int width, int height);

std::pair<CoordinateFloat, float> enclosing_circle_approx(const std::vector<Coordinate>& points);

std::vector<float> radial_signal(const std::vector<Coordinate>& points, CoordinateFloat center);
