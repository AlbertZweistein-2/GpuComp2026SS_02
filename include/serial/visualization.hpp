#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "serial/pipeline.hpp"

// Draws contour (red), bounding box (light green), and label + edge type string (blue)
// onto a copy of the input RGB image and writes it to output_path.
void draw_piece_overlays(
    const uint8_t* rgb_data,
    int width,
    int height,
    const std::vector<PuzzlePiece>& pieces,
    const std::string& output_path
);
