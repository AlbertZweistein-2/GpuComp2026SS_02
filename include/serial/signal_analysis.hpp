#pragma once

#include <array>
#include <string>

#include "types.hpp"

// Smooths one radial signal with a centered circular moving average.
Signal smooth_signal(
    const Signal& signal,
    int k
);

// Finds up to four corner indices from the smoothed radial signal using local
// maxima, prominence/sharpness filters, and greedy spacing.
Corners find_triangular_peaks(
    const Signal& smooth,
    float min_prominence,
    float min_sharpness,
    int   min_distance
);

// Converts one edge enum to the compact L/V/C representation.
char edge_char(EdgeType e);

// Classifies the four edges between the detected corners from the raw radial
// signal.
EdgeLabels classify_edges(
    const Signal& raw,
    const Corners& corners,
    float tol_factor
);

// Converts four edge labels into the string consumed by PuzzleLookupTable.
std::string edges_to_string(const EdgeLabels& labels);

// Rotation-invariant mapping from edge strings to puzzle-piece class labels.
class PuzzleLookupTable {
public:
    PuzzleLookupTable();

    std::string getClassLabel(const std::string& edges) const;

private:
    // 3^4 possible strings because each of four edges has L/V/C state.
    std::array<std::string, 81> table;

    int charToDigit(char c) const;
    int getBase3Index(const std::string& s) const;
    std::string rotate(const std::string& s) const;
    std::string getCategory(const std::string& s) const;
};
