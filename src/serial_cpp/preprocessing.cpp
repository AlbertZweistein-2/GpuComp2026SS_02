/**
 * Serial C++ — Step 1: Image Preprocessing
 * TU Wien — GPU Architecture & Computing SS2026
 *
 * Reference serial implementation of the full pipeline:
 *   1. RGB -> Grayscale      (ITU-R BT.601)
 *   2. Normalization         (Min/Max -> [0, 255])
 *   3. Gaussian Blur         (2-D convolution with separable kernel)
 *   4. Histogram + Otsu      (automatic threshold computation)
 *   5. Binarization
 *
 * Compile:
 *   g++ -O2 -std=c++17 -o preprocessing_serial preprocessing_serial.cpp -lm
 *
 * Usage:
 *   ./preprocessing_serial                        # loads ../../data/single.JPG
 *   ./preprocessing_serial ../../data/1_p1.jpg   # any image
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// stb_image - single-header image loader (place stb_image.h in the same folder)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "serial/preprocessing.hpp"
#include "helpers.hpp"
// ─── Constants (same as the CUDA implementation) ─────────────────────────────
static constexpr int MAX_KERNEL = 15;
static constexpr int HIST_BINS  = 256;

// -----------------------------------------------------------------------------
// STEP 1 — RGB -> Grayscale
//
// Formula: Y = 0.299·R + 0.587·G + 0.114·B  (ITU-R BT.601)
// Matches the CUDA kernel `rgb_to_grayscale`, but implemented sequentially.
// -----------------------------------------------------------------------------
static void rgb_to_grayscale(
    const uint8_t* rgb,
    int width, int height,
    std::vector<float>& gray)
{
    gray.resize(static_cast<size_t>(width) * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            float r = rgb[idx];
            float g = rgb[idx + 1];
            float b = rgb[idx + 2];
            gray[y * width + x] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }
}

// -----------------------------------------------------------------------------
// STEP 2 — Normalization
//
// Scales all pixel values linearly to [0, 255].
// Equivalent to the combined reduce_minmax + normalize operation in the CUDA version.
// -----------------------------------------------------------------------------
static void normalize(std::vector<float>& gray)
{
    float gmin = *std::min_element(gray.begin(), gray.end());
    float gmax = *std::max_element(gray.begin(), gray.end());

    float range = gmax - gmin;
    if (range < 1e-5f) range = 1.0f;   // avoid division by zero

    for (float& v : gray)
        v = (v - gmin) / range * 255.0f;
}

// -----------------------------------------------------------------------------
// STEP 3a — Create Gaussian kernel
//
// Produce a normalized (ksize × ksize) Gaussian kernel.
// Uses the same formula as the CUDA function `make_gaussian_kernel()`.
// -----------------------------------------------------------------------------
static std::vector<float> make_gaussian_kernel(int ksize, float sigma)
{
    if (ksize < 1 || ksize % 2 == 0 || ksize > MAX_KERNEL)
        throw std::invalid_argument(
            "ksize must be odd and <= " + std::to_string(MAX_KERNEL));
    if (sigma <= 0.f)
        throw std::invalid_argument("sigma must be > 0");

    std::vector<float> kernel(ksize * ksize);
    int half = ksize / 2;
    float sum = 0.f;

    for (int y = -half; y <= half; ++y) {
        for (int x = -half; x <= half; ++x) {
            float v = std::exp(-(x*x + y*y) / (2.f * sigma * sigma));
            kernel[(y + half) * ksize + (x + half)] = v;
            sum += v;
        }
    }

    // Normalize: sum of all weights = 1
    for (float& v : kernel)
        v /= sum;

    return kernel;
}

// -----------------------------------------------------------------------------
// STEP 3b — Gaussian Blur (2-D convolution)
//
// Serial version of the `gaussian_blur_shared` CUDA kernel.
// Border handling: clamp (edge pixels are repeated).
//
// Optimization note: the CUDA version separates the kernel (horizontal then vertical).
// That optimization could be applied here as well, but direct 2-D convolution
// is sufficient for the serial reference implementation.
// -----------------------------------------------------------------------------
static void gaussian_blur(
    const std::vector<float>& in,
    int width, int height,
    int ksize, const std::vector<float>& kernel,
    std::vector<float>& out)
{
    out.resize(static_cast<size_t>(width) * height);

    int half = ksize / 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.f;

            for (int ky = 0; ky < ksize; ++ky) {
                for (int kx = 0; kx < ksize; ++kx) {
                    // Clamp border handling (same as CUDA kernel)
                    int sy = clamp(y - half + ky, 0, height - 1);
                    int sx = clamp(x - half + kx, 0,  width - 1);
                    sum += in[sy * width + sx] * kernel[ky * ksize + kx];
                }
            }

            out[y * width + x] = sum;
        }
    }
}

// -----------------------------------------------------------------------------
// STEP 4a — Compute histogram
// -----------------------------------------------------------------------------
static void compute_histogram(
    const std::vector<float>& gray,
    std::vector<uint32_t>& hist)
{
    hist.assign(HIST_BINS, 0);

    for (float v : gray) {
        int bin = static_cast<int>(std::min(v, 255.f));
        ++hist[bin];
    }
}

// -----------------------------------------------------------------------------
// STEP 4b — Otsu threshold
//
// Maximize inter-class variance between foreground and background.
// Same logic as `otsu_from_histogram()` in the CUDA implementation
// and the Python reference (`otsu_threshold`).
// -----------------------------------------------------------------------------
static float otsu_threshold(
        const std::vector<uint32_t>& hist,
        int total_pixels)
{
    double sum_total = 0.0;
    for (int t = 0; t < HIST_BINS; ++t)
        sum_total += t * hist[t];

    double sum_bg = 0.0, w_bg = 0.0;
    double max_var = 0.0;
    int threshold = 0;

    for (int t = 0; t < HIST_BINS; ++t) {
        w_bg += hist[t];
        if (w_bg == 0.0) continue;

        double w_fg = total_pixels - w_bg;
        if (w_fg == 0.0) break;

        sum_bg += t * hist[t];
        double mean_bg = sum_bg / w_bg;
        double mean_fg = (sum_total - sum_bg) / w_fg;

        double var = w_bg * w_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
        if (var > max_var) {
            max_var = var;
            threshold = t;
        }
    }
    return static_cast<float>(threshold);
}

// -----------------------------------------------------------------------------
// STEP 5 - Binarization
// -----------------------------------------------------------------------------
static void binarize(
    const std::vector<float>& gray,
    float threshold,
    std::vector<uint8_t>& binary)
{
    binary.resize(gray.size());

    for (size_t i = 0; i < gray.size(); ++i)
        binary[i] = (gray[i] > threshold) ? 255u : 0u;

}

// -----------------------------------------------------------------------------
// Main pipeline — combines all steps
// -----------------------------------------------------------------------------

ImageU8 preprocess(
        const uint8_t* rgb,
        int width, 
        int height,
        int ksize,
        float sigma)
{
    ImageU8 result;
    result.width = width;
    result.height = height;

    // 1 — RGB -> Grayscale
    std::vector<float> gray;
    rgb_to_grayscale(rgb, width, height, gray);

    // 2 — Normalization
    normalize(gray);

    // 3 — Gaussian Blur
    std::vector<float> kernel = make_gaussian_kernel(ksize, sigma);
    std::vector<float> blurred;
    gaussian_blur(gray, width, height, ksize, kernel, blurred);

    // 4a — Histogram
    std::vector<uint32_t> hist;
    compute_histogram(blurred, hist);

    // 4b — Otsu
    auto threshold = otsu_threshold(hist, width * height);

    // 5 — Binarization
    binarize(blurred, threshold, result.data);

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
// int main(int argc, char** argv)
// {
//     const char* input_path  = (argc > 1) ? argv[1] : "../../data/single.JPG";
//     // Build output path by prefixing the input filename with "binary_" and using .png
//     std::string input_str = input_path;
//     std::string output_path;
//     {
//         size_t pos = input_str.find_last_of("/\\");
//         std::string dir = (pos == std::string::npos) ? std::string() : input_str.substr(0, pos + 1);
//         std::string filename = (pos == std::string::npos) ? input_str : input_str.substr(pos + 1);
//         size_t dot = filename.find_last_of('.');
//         std::string base = (dot == std::string::npos) ? filename : filename.substr(0, dot);
//         output_path = dir + "binary_" + base + ".png";
//     }

//     // Load image
//     int W, H, channels;
//     uint8_t* rgb = stbi_load(input_path, &W, &H, &channels, 3);
//     if (!rgb) {
//         std::cerr << "Error: Could not load image: " << input_path
//                   << "\n" << stbi_failure_reason() << "\n";
//         return EXIT_FAILURE;
//     }
//     std::cout << "Loaded image: " << input_path
//               << "  (" << W << " x " << H << ")\n";

//     // Run pipeline (ksize=5, sigma=1.0 — same as Python reference)
//     auto res = preprocess(rgb, W, H, 5, 1.0f);
//     stbi_image_free(rgb);

//     // Statistik
//     int white = 0;
//     for (uint8_t v : res.binary) white += (v == 255);
//     int total = W * H;

//     std::cout << "\n-- Result ---------------------------------\n";
//     std::cout << "Otsu threshold    : " << res.threshold        << "\n";
//     std::cout << "Foreground        : " << white
//               << " pixels  (" << 100.f * white / total << " %)\n";
//     std::cout << "Background        : " << (total - white)
//               << " pixels  (" << 100.f * (total - white) / total << " %)\n";

//     std::cout << "\n-- Timings (serial) ------------------------\n";
//     std::cout << "  RGB -> Grayscale : " << res.timing.grayscale << " ms\n";
//     std::cout << "  Normalization    : " << res.timing.normalize << " ms\n";
//     std::cout << "  Gaussian Blur    : " << res.timing.blur      << " ms\n";
//     std::cout << "  Histogram        : " << res.timing.histogram << " ms\n";
//     std::cout << "  Otsu             : " << res.timing.otsu      << " ms\n";
//     std::cout << "  Binarize         : " << res.timing.binarize  << " ms\n";
//     std::cout << "  -----------------------------------------\n";
//     std::cout << "  Total            : " << res.timing.total     << " ms\n";

//     // Save binary image
//     if (stbi_write_png(output_path.c_str(), W, H, 1, res.binary.data(), W))
//         std::cout << "\nSaved: " << output_path << "\n";
//     else
//         std::cerr << "Error saving: " << output_path << "\n";

//     return EXIT_SUCCESS;
// }
