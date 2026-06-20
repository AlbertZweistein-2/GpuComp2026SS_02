#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "serial/pipeline.hpp"

namespace parallel_visualization {

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

} // namespace parallel_visualization
