#include "parallel/visualization.hpp"

#ifndef ENABLE_NVTX
#define ENABLE_NVTX 0
#endif

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <cstddef>
#include <string>
#include <vector>

#include "types.hpp"

#if ENABLE_NVTX >= 1
#include <nvtx3/nvToolsExt.h>
#endif

// Visualization is diagnostic output, not part of the core image-processing
// pipeline. CUDA would add launch/copy overhead for small, branchy line/text
// drawing work, so the parallel path only uses OpenMP at piece granularity.
#ifdef _OPENMP
#include <omp.h>
#endif

// ______________________________________________________________________________________________

namespace {

#if ENABLE_NVTX >= 1
class NvtxRange
{
public:
    explicit NvtxRange(const char* label)
    {
        nvtxRangePushA(label);
    }

    ~NvtxRange()
    {
        nvtxRangePop();
    }
};
#else
class NvtxRange
{
public:
    // Compile the same source with or without NVTX instrumentation.
    explicit NvtxRange(const char*) {}
};
#endif

// sets a single pixel in the RGB image
inline void set_pixel(uint8_t* img, int width, int height,
                      int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
    // if pixel is out of bounds, do nothing
    if (x < 0 || x >= width || y < 0 || y >= height)
    {
        return;
    }
    // image is stored as flat row-major array
    // with RGB triplets
    size_t idx = static_cast<size_t>((y * width + x) * 3);

    img[idx]     = r;
    img[idx + 1] = g;
    img[idx + 2] = b;
}

// ______________________________________________________________________________________________

// Bresenham's line algorithm
// draws a line between two points with minimal error
// inherently serial, but probably not necessary to parallelize for this application
// GPU overhead would likely dominate
void draw_line(uint8_t* img, int width, int height,
               int x0, int y0, int x1, int y1,
               uint8_t r, uint8_t g, uint8_t b)
{
    // dx and dy are the absolute distances between the two points in x and y
    // sx and sy are the step directions in x and y
    int dx = std::abs(x1 - x0);
    int sx;

    if (x0 < x1)
    {
        sx = 1;
    }
    else
    {
        sx = -1;
    }

    int dy = -std::abs(y1 - y0);
    int sy;

    if (y0 < y1)
    {
        sy = 1;
    }
    else
    {
        sy = -1;
    }

    // err accumulates how far we are from the ideal line
    int err = dx + dy;

    while (true)
    {
        // Clipping is handled in set_pixel, so lines may extend outside the
        // image without special-case logic here.
        set_pixel(img, width, height, x0, y0, r, g, b);

        // while we have not reached the end point
        if (x0 == x1 && y0 == y1)
        {
            break;
        }

        int e2 = 2 * err;

        // if the error is too large in x or y direction, we step in that direction
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }

        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
    }
}

// ______________________________________________________________________________________________

// draws a rectangle outline with given thickness
// just using four lines
void draw_rect(uint8_t* img, int width, int height,
               int x, int y, int w, int h,
               uint8_t r, uint8_t g, uint8_t b, int thickness = 2)
{
    for (int t = 0; t < thickness; ++t)
    {
        // Draw inward offsets for thickness instead of filling the rectangle.
        draw_line(img, width, height, x+t,   y+t,   x+w-t, y+t,   r, g, b);
        draw_line(img, width, height, x+w-t, y+t,   x+w-t, y+h-t, r, g, b);
        draw_line(img, width, height, x+w-t, y+h-t, x+t,   y+h-t, r, g, b);
        draw_line(img, width, height, x+t,   y+h-t, x+t,   y+t,   r, g, b);
    }
}

// ______________________________________________________________________________________________

// renders a string into the RGB image using stb_truetype
// silently skips text rendering if font file cannot be loaded
void draw_text(uint8_t* img, int img_width, int img_height,
               const std::string& text, int x, int y, float font_size,
               uint8_t r, uint8_t g, uint8_t b,
               const stbtt_fontinfo& font)
{
    // scale factor for desired pixel height
    float scale = stbtt_ScaleForPixelHeight(&font, font_size);

    // baseline is where the characters sit
    // and ascent is the distance from baseline to top of tallest character
    int ascent, descent, line_gap;
    stbtt_GetFontVMetrics(&font, &ascent, &descent, &line_gap);
    int baseline = y + static_cast<int>(ascent * scale);

    // rendering each character sequentially
    // cursor_x is the current x position for the next character
    int cursor_x = x;
    for (char c : text)
    {
        // stb_truetype turns the character into a bitmap
        int bw, bh, bx, by;
        unsigned char* bitmap = stbtt_GetCodepointBitmap(
            &font, 0, scale, c, &bw, &bh, &bx, &by);

        for (int row = 0; row < bh; ++row)
        {
            for (int col = 0; col < bw; ++col)
            {
                // Alpha-blend glyph coverage with the existing image so labels
                // remain readable without drawing an opaque text box.
                float alpha = bitmap[row * bw + col] / 255.0f;
                if (alpha == 0.0f)
                {
                    continue;
                }

                // px and py are the absolute pixel coordinates in the image
                int px = cursor_x + bx + col;
                int py = baseline + by + row;
                if (px < 0 || px >= img_width || py < 0 || py >= img_height)
                {
                    continue;
                }

                size_t idx = static_cast<size_t>((py * img_width + px) * 3);
                img[idx]     = static_cast<uint8_t>(alpha * r + (1.0f - alpha) * img[idx]);
                img[idx + 1] = static_cast<uint8_t>(alpha * g + (1.0f - alpha) * img[idx + 1]);
                img[idx + 2] = static_cast<uint8_t>(alpha * b + (1.0f - alpha) * img[idx + 2]);
            }
        }

        stbtt_FreeBitmap(bitmap, nullptr);

        // advance is the x distance to the next character
        // moving horizontally
        int advance, lsb;
        stbtt_GetCodepointHMetrics(&font, c, &advance, &lsb);
        cursor_x += static_cast<int>(advance * scale);
    }
}

bool has_extension(const std::string& path, const std::string& extension)
{
    // Case-insensitive suffix check keeps ".BMP" and ".bmp" equivalent.
    if (path.size() < extension.size())
    {
        return false;
    }

    return std::equal(extension.rbegin(), extension.rend(), path.rbegin(),
                      [](char expected, char actual)
                      {
                          return std::tolower(static_cast<unsigned char>(expected)) ==
                                 std::tolower(static_cast<unsigned char>(actual));
                      });
}

} // namespace

namespace parallel_visualization {

int write_overlay_image(const std::string& output_path, int width, int height, const uint8_t* img)
{
    // BMP writes dominate the profile on some runs, so keep a separate NVTX
    // range from PNG writes when instrumentation is enabled.
    if (has_extension(output_path, ".bmp"))
    {
        NvtxRange range("Image store: write BMP");
        return stbi_write_bmp(output_path.c_str(), width, height, 3, img);
    }

    NvtxRange range("Image store: write PNG");
    return stbi_write_png(output_path.c_str(), width, height, 3, img, width * 3);
}

void render_piece_overlays(
    uint8_t* rgb_data,
    int width,
    int height,
    const std::vector<PuzzlePiece>& pieces)
{
    // Draw directly into the loaded RGB image. The CUDA pipeline writes the
    // overlay immediately afterwards, so keeping an unmodified copy is wasted.
    // Scale font size and padding to the image resolution. The constants come
    // from the original high-resolution input image.
    const float label_font_size = 50.0f * (static_cast<float>(height) / 5100.0f); // 5100 is the height of the original image used for scaling
    const float label_padding = 8.0f * (static_cast<float>(height) / 5100.0f); // 5100 is the height of the original image used for scaling
    const float padding_left = 16.0f * (static_cast<float>(width) / 5100.0f); // 5100 is the height of the original image used for scaling

    // Load the font once and share the read-only stbtt_fontinfo across pieces.
    // If this fails, contours and boxes are still drawn.
    const std::string font_path = "data/fonts/DejaVuSans.ttf";
    std::ifstream font_file(font_path, std::ios::binary | std::ios::ate);
    std::vector<uint8_t> font_data;
    stbtt_fontinfo font_info;
    bool font_loaded = false;
    {
        NvtxRange range("Visualization: load font");
        if (font_file.is_open()) {
            std::streamsize size = font_file.tellg();
            font_file.seekg(0, std::ios::beg);
            font_data.resize(static_cast<size_t>(size));
            font_file.read(reinterpret_cast<char*>(font_data.data()), size);
            font_loaded = stbtt_InitFont(&font_info, font_data.data(), 0);
        }
    }
    if (!font_loaded)
        std::cerr << "[visualization] warning: font file not found: " << font_path << '\n';

    // Use OpenMP when enabled for drawing; the later file write remains serial.
    {
        NvtxRange range("Visualization: draw overlays");
#ifdef _OPENMP
// Piece-level parallelism is coarse but cheap. Writes are unsynchronized; this
// relies on puzzle-piece overlays being mostly disjoint, and minor overlap is
// acceptable for this diagnostic image.
#pragma omp parallel for
#endif
        for (std::ptrdiff_t piece_idx = 0; piece_idx < static_cast<std::ptrdiff_t>(pieces.size()); ++piece_idx)
        {
            const PuzzlePiece& piece = pieces[static_cast<size_t>(piece_idx)];
            const Region& region = piece.region;
            const CoordinateVector<int>& contour = piece.contour;

            // drawing the contour in red
            // contour points are (x, y) again after the coordinate swap in find_contour_chain_approx_simple
            for (size_t i = 0; i < contour.size(); ++i)
            {
                // connecting consecutive points in the contour with lines
                // wrapping around to close the contour
                size_t next = (i + 1) % contour.size();
                draw_line(rgb_data, width, height,
                          contour[i].a,    contour[i].b,
                          contour[next].a, contour[next].b,
                          255, 0, 0);
            }

            // drawing the bounding box in light green
            draw_rect(rgb_data, width, height,
                      region.x, region.y, region.width, region.height,
                      144, 238, 144, 2);

            // drawing the label text in white above the bounding box
            std::string piece_label;
            piece_label = piece.class_label;
            std::string text = "Piece " + std::to_string(piece_idx + 1) + " ; Class " + piece_label;
            // Position labels above the box when possible, clamping to the
            // image boundary so top-edge pieces still get visible labels.
            int text_y = std::max(0, region.y - static_cast<int>(label_font_size + label_padding));
            int text_x = std::max(0, region.x - static_cast<int>(padding_left));
            if (font_loaded)
                draw_text(rgb_data, width, height, text, text_x, text_y, label_font_size, 255, 255, 255, font_info);
        }
    }
}

} // namespace parallel_visualization
