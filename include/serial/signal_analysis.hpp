#pragma once

#include <array>
#include <string>
#include <vector>

#include "types.hpp"

enum class EdgeType { Straight, Knob, Hole };

Signal smooth_signal(
    const std::vector<float>& signal,
    int k
);

Signal compute_1d_sharpness(
    const std::vector<float>& radial_signal,
    int k
);

std::array<int, 4> find_triangular_peaks(
    const std::vector<float>& curvature,
    float min_prominence,
    int min_distance
);

char edge_char(EdgeType e);

std::array<EdgeType, 4> classify_edges(
    const std::vector<float>& signal,
    const std::array<int, 4>& corner_idx,
    float knob_factor = 1.08f,
    float hole_factor = 0.92f
);

std::string edges_to_string(const std::array<EdgeType, 4>& labels);
