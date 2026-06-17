#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "serial/pipeline.hpp"

namespace parallel_visualization {

void draw_piece_overlays(
    const uint8_t* rgb_data,
    int width,
    int height,
    const std::vector<PuzzlePiece>& pieces,
    const std::string& output_path
);

} // namespace parallel_visualization
