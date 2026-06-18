#pragma once

#include <array>
#include <string>

#include "types.hpp"

// inherently sequential, identical to serial implementation
Signal smooth_signal_cuda(
    const Signal &signal,
    int k);

// partially parallelized via thrust for local maxima finding
Corners find_triangular_peaks_cuda(
    const Signal &smooth,
    float min_prominence,
    float min_sharpness,
    int min_distance);

// inherently sequential, identical to serial implementation
char edge_char_cuda(EdgeType e);

// inherently sequential, identical to serial implementation
EdgeLabels classify_edges_cuda(
    const Signal &raw,
    const Corners &corners,
    float tol_factor);

// inherently sequential, identical to serial implementation
std::string edges_to_string_cuda(const EdgeLabels &labels);

class PuzzleLookupTable
{
public:
    PuzzleLookupTable();

    std::string getClassLabel(const std::string &edges) const;

private:
    std::array<std::string, 81> table;

    int charToDigit(char c) const;
    int getBase3Index(const std::string &s) const;
    std::string rotate(const std::string &s) const;
    std::string getCategory(const std::string &s) const;
};
