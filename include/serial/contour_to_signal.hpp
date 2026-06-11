#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "types.hpp"

std::vector<Coordinate<int>> trace_contour(const std::vector<uint8_t> &boundary, int width, int height);

std::vector<Coordinate<int>> simplify_chain_approx(const std::vector<Coordinate<int>> &contour);

std::vector<Coordinate<int>> find_contour_chain_approx_simple(const std::vector<uint8_t>& boundary, int width, int height);

std::pair<Coordinate<float>, float> enclosing_circle_approx(const std::vector<Coordinate<int>>& points);

Signal radial_signal(const std::vector<Coordinate<int>>& points, Coordinate<float> center);
