#include "serial/visualization.hpp"

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "types.hpp"

// ______________________________________________________________________________________________

// sets a single pixel in the RGB image
inline void set_pixel(std::vector<uint8_t>& img, int width, int height,
                      int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    size_t idx = static_cast<size_t>((y * width + x) * 3);
    img[idx]     = r;
    img[idx + 1] = g;
    img[idx + 2] = b;
}

// ______________________________________________________________________________________________

// Bresenham's line algorithm
void draw_line(std::vector<uint8_t>& img, int width, int height,
               int x0, int y0, int x1, int y1,
               uint8_t r, uint8_t g, uint8_t b)
{
    int dx =  std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    while (true)
    {
        set_pixel(img, width, height, x0, y0, r, g, b);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// ______________________________________________________________________________________________

// draws a rectangle outline with given thickness
void draw_rect(std::vector<uint8_t>& img, int width, int height,
               int x, int y, int w, int h,
               uint8_t r, uint8_t g, uint8_t b, int thickness = 2)
{
    for (int t = 0; t < thickness; ++t)
    {
        draw_line(img, width, height, x+t,   y+t,   x+w-t, y+t,   r, g, b);
        draw_line(img, width, height, x+w-t, y+t,   x+w-t, y+h-t, r, g, b);
        draw_line(img, width, height, x+w-t, y+h-t, x+t,   y+h-t, r, g, b);
        draw_line(img, width, height, x+t,   y+h-t, x+t,   y+t,   r, g, b);
    }
}

// ______________________________________________________________________________________________

// renders a string into the RGB image using stb_truetype
// silently skips text rendering if font file cannot be loaded
void draw_text(std::vector<uint8_t>& img, int img_width, int img_height,
               const std::string& text, int x, int y, float font_size,
               uint8_t r, uint8_t g, uint8_t b,
               const std::string& font_path = "data/fonts/DejaVuSans.ttf")
{
    // loading font file
    std::ifstream font_file(font_path, std::ios::binary | std::ios::ate);
    if (!font_file.is_open())
    {
        std::cerr << "[visualization] warning: font file not found: " << font_path << '\n';
        return;
    }
    std::streamsize size = font_file.tellg();
    font_file.seekg(0, std::ios::beg);
    std::vector<uint8_t> font_data(static_cast<size_t>(size));
    font_file.read(reinterpret_cast<char*>(font_data.data()), size);

    stbtt_fontinfo font;
    if (!stbtt_InitFont(&font, font_data.data(), 0))
    {
        std::cerr << "[visualization] warning: failed to init font\n";
        return;
    }

    // scale factor for desired pixel height
    float scale = stbtt_ScaleForPixelHeight(&font, font_size);

    // baseline position
    int ascent, descent, line_gap;
    stbtt_GetFontVMetrics(&font, &ascent, &descent, &line_gap);
    int baseline = y + static_cast<int>(ascent * scale);

    // rendering each character
    int cursor_x = x;
    for (char c : text)
    {
        int bw, bh, bx, by;
        unsigned char* bitmap = stbtt_GetCodepointBitmap(
            &font, 0, scale, c, &bw, &bh, &bx, &by);

        // blitting the glyph bitmap into the image with alpha blending
        for (int row = 0; row < bh; ++row)
        {
            for (int col = 0; col < bw; ++col)
            {
                float alpha = bitmap[row * bw + col] / 255.0f;
                if (alpha == 0.0f) continue;

                int px = cursor_x + bx + col;
                int py = baseline + by + row;
                if (px < 0 || px >= img_width || py < 0 || py >= img_height) continue;

                size_t idx = static_cast<size_t>((py * img_width + px) * 3);
                img[idx]     = static_cast<uint8_t>(alpha * r + (1.0f - alpha) * img[idx]);
                img[idx + 1] = static_cast<uint8_t>(alpha * g + (1.0f - alpha) * img[idx + 1]);
                img[idx + 2] = static_cast<uint8_t>(alpha * b + (1.0f - alpha) * img[idx + 2]);
            }
        }

        stbtt_FreeBitmap(bitmap, nullptr);

        // advancing cursor to next character
        int advance, lsb;
        stbtt_GetCodepointHMetrics(&font, c, &advance, &lsb);
        cursor_x += static_cast<int>(advance * scale);
    }
}

// ______________________________________________________________________________________________

void draw_piece_overlays(
    const uint8_t* rgb_data,
    int width,
    int height,
    const std::vector<PuzzlePiece>& pieces,
    const std::string& output_path)
{
    // working on a copy so we don't modify the input
    std::vector<uint8_t> img = rgb_data;

    for (size_t p = 0; p < regions.size(); ++p)
    {
        const Region& region   = regions[p];
        const CoordinateVector<int>& contour = contours[p];
        const std::string& label = edge_labels[p];

        // drawing the contour in red
        // contour points are (x, y) after the coordinate swap in find_contour_chain_approx_simple
        for (size_t i = 0; i < contour.size(); ++i)
        {
            size_t next = (i + 1) % contour.size();
            draw_line(img, width, height,
                      contour[i].a,    contour[i].b,
                      contour[next].a, contour[next].b,
                      255, 0, 0);
        }

        // drawing the bounding box in light green
        draw_rect(img, width, height,
                  region.x, region.y, region.width, region.height,
                  144, 238, 144, 2);

        // drawing the label text in blue above the bounding box
        // format: "E{label}: {edge_type}" e.g. "E1: ????"
        std::string text = "E" + std::to_string(region.label) + ": " + label;
        int text_y = std::max(0, region.y - 20);
        draw_text(img, width, height, text, region.x, text_y, 16.0f, 0, 0, 255);
    }

    // saving output image if path is provided
    if (!output_path.empty())
    {
        stbi_write_png(output_path.c_str(), width, height, 3, img.data(), width * 3);
    }
}
