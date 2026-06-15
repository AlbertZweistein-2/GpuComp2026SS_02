#pragma once

#include <array>
#include <string>
#include <vector>

#include "types.hpp"


Signal smooth_signal(
    const Signal& signal,
    int k
);

Corners find_triangular_peaks(
    const Signal& smooth,
    float min_prominence,
    float min_sharpness,
    int   min_distance
);

char edge_char(EdgeType e);

EdgeLabels classify_edges(
    const Signal& raw,
    const Corners& corners,
    float tol_factor
);

std::string edges_to_string(const EdgeLabels& labels);
