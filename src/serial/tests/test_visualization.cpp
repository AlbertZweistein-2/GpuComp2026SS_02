#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "types.hpp"
#include "serial/visualization.hpp"

// ______________________________________________________________________________________________

std::vector<uint8_t> make_white_image(int width, int height)
{
    return std::vector<uint8_t>(static_cast<size_t>(width * height * 3), 255);
}

// ______________________________________________________________________________________________

void test_draw_piece_overlays(bool save_output = false)
{
    std::cout << "Running test_draw_piece_overlays..." << std::endl;

    const int width = 400;
    const int height = 400;
    std::vector<uint8_t> rgb = make_white_image(width, height);

    // fake piece 1: small rectangle in top-left
    CoordinateVector<int> contour1 = {
        {50,50},{60,50},{70,50},{80,50},{90,50},{100,50},
        {100,60},{100,70},{100,80},{100,90},{100,100},
        {90,100},{80,100},{70,100},{60,100},{50,100},
        {50,90},{50,80},{50,70},{50,60}
    };
    Region region1;
    region1.label  = 1;
    region1.area   = 2500;
    region1.x      = 50;
    region1.y      = 50;
    region1.width  = 51;
    region1.height = 51;

    // fake piece 2: small rectangle in bottom-right
    CoordinateVector<int> contour2 = {
        {250,250},{260,250},{270,250},{280,250},{290,250},{300,250},
        {300,260},{300,270},{300,280},{300,290},{300,300},
        {290,300},{280,300},{270,300},{260,300},{250,300},
        {250,290},{250,280},{250,270},{250,260}
    };
    Region region2;
    region2.label  = 2;
    region2.area   = 2500;
    region2.x      = 250;
    region2.y      = 250;
    region2.width  = 51;
    region2.height = 51;

    PuzzlePiece piece1;
    piece1.region = region1;
    piece1.contour = contour1;
    piece1.class_label = "C_0: LLVC";

    PuzzlePiece piece2;
    piece2.region = region2;
    piece2.contour = contour2;
    piece2.class_label = "E_0: LVVC";

    std::vector<PuzzlePiece> pieces = {piece1, piece2};

    if (save_output)
    {
        const std::string output_path = "data/pipeline_output/test_visualization_output.png";
        draw_piece_overlays(rgb.data(), width, height, pieces, output_path);
        std::cout << "Output saved to: " << output_path << std::endl;
    }
    else
    {
        // run without saving, just verify it does not crash
        draw_piece_overlays(rgb.data(), width, height, pieces, "");
    }

    std::cout << "test_draw_piece_overlays passed!" << std::endl;
}

// ______________________________________________________________________________________________

int main()
{
    test_draw_piece_overlays(false);
    std::cout << "All visualization tests passed!" << std::endl;
    return 0;
}
