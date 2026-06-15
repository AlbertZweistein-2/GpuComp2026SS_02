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

class PuzzleLookupTable {
public:
    PuzzleLookupTable();

    std::string getClassLabel(const std::string& edges) const;

private:
    std::array<std::string, 81> table;

    int charToDigit(char c) const;
    int getBase3Index(const std::string& s) const;
    std::string rotate(const std::string& s) const;
    std::string getCategory(const std::string& s) const;
};
