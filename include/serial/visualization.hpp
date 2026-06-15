#pragma once

#include <string>
#include <vector>

#include "types.hpp"

// Draws contour (red), bounding box (light green), and label + edge type string (blue)
// onto a copy of the input RGB image and writes it to output_path.
// Uses stb_truetype for text rendering (no OpenCV dependency).
// Font file must exist at data/fonts/DejaVuSans.ttf.
// label is the piece index, edge_type is a placeholder string e.g. "ABCD"
// until classification is implemented.
// If output_path is empty, the image is not saved (useful for testing).
void draw_piece_overlays(
    const std::vector<uint8_t>& rgb_data,
    int width,
    int height,
    const std::vector<Region>& regions,
    const std::vector<CoordinateVector<int>>& contours,
    const std::vector<std::string>& edge_labels,
    const std::string& output_path
);
