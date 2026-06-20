#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "serial/pipeline.hpp"

// Draws contour (red), bounding box (light green), and label + edge type string (blue)
// onto a copy of the input RGB image.
// Uses stb_truetype for text rendering (no OpenCV dependency).
// Font file must exist at data/fonts/DejaVuSans.ttf.
// label is the piece index, edge_type is a placeholder string e.g. "ABCD"
// until classification is implemented.
std::vector<uint8_t> render_piece_overlays(
    const uint8_t* rgb_data,
    int width,
    int height,
    const std::vector<PuzzlePiece>& pieces
);

int write_overlay_image(
    const std::string& output_path,
    int width,
    int height,
    const std::vector<uint8_t>& img
);
