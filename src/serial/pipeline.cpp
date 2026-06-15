#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

#ifndef SUB_TIMINGS
#define SUB_TIMINGS 0
#endif

#include "serial/pipeline.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "helpers.hpp"
#include "serial/boundary_extraction.hpp"
#include "serial/cleaning.hpp"
#include "serial/component_labeling.hpp"
#include "serial/contour_to_signal.hpp"
#include "serial/preprocessing.hpp"
#include "serial/signal_analysis.hpp"
#include "stb_image.h"

namespace fs = std::filesystem;

namespace {

fs::path project_path(const std::string& path)
{
    fs::path p(path);
    return p.is_absolute() ? p : fs::current_path() / p;
}

template <typename Fn>
void time_stage(double& seconds, Fn&& fn)
{
#if SUB_TIMINGS >= 1
    Timer timer;
    timer.reset();
    fn();
    seconds += timer.get();
#else
    fn();
#endif
}

#if DEBUG_LEVEL >= 1
void debug_print(std::string_view message)
{
    std::cout << message << '\n';
}
#else
#define debug_print(message) ((void)0)
#endif

uint8_t* load_rgb_image(const fs::path& path, int& width, int& height)
{
    int channels = 0;
    uint8_t* data = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
    if (data == nullptr) {
        throw std::runtime_error("Could not load image: " + path.string() + "\n" + stbi_failure_reason() + "\n");
    }
    return data;
}

} // namespace

PipelineResult run(const PipelineOptions& options)
{
    Timer total_timer;
    total_timer.reset();

    const fs::path input_image_path = project_path(options.input_image_path);
    PipelineResult result;

    debug_print("[pipeline] loading image: " + input_image_path.string());
    int width = 0;
    int height = 0;
    uint8_t* rgb = load_rgb_image(input_image_path, width, height);

    debug_print("[pipeline] preprocessing");
    time_stage(result.timings.preprocessing, [&]() {
        result.binary = preprocess(
            rgb,
            width,
            height,
            options.gaussian_kernel_size,
            options.gaussian_sigma);
    });
    stbi_image_free(rgb);

    debug_print("[pipeline] cleaning");
    time_stage(result.timings.cleaning, [&]() {
        ImageU8 kernel = make_kernel(options.morphology_kernel_width, options.morphology_kernel_height);
        morphological_open(result.binary, kernel, result.cleaned, options.morphology_iterations);
    });

    ImageI32 labels;
    std::vector<Region> regions;
    debug_print("[pipeline] connected components");
    time_stage(result.timings.connected_components, [&]() {
        auto components = connected_components(result.cleaned, options.min_region_area);
        labels = std::move(components.first);
        regions = std::move(components.second);
    });
    debug_print("[pipeline] found " + std::to_string(regions.size()) + " regions");

    result.pieces.reserve(regions.size());

    // Some debugging happening here
    std::cout << "Number of puzzle pieces detected: " << regions.size() << '\n';
    for (const auto& region : regions) {
        std::cout << "  Label: " << region.label
                  << ", Area: " << region.area
                  << ", Bounding box: (" << region.x << ", " << region.y
                  << "), Width: " << region.width
                  << ", Height: " << region.height
                  << '\n';
    }

    for (const auto& region : regions) {
        ImageU8 piece_mask;
        debug_print("[pipeline] extracting piece mask: " + std::to_string(region.label));
        time_stage(result.timings.mask_extraction, [&]() {
            piece_mask = extract_piece_mask(labels, region.label);
        });

        ImageU8 piece_boundary;
        debug_print("[pipeline] extracting piece boundary: " + std::to_string(region.label));
        time_stage(result.timings.boundary_extraction, [&]() {
            get_external_boundary_mask(piece_mask, piece_boundary);
        });

        PuzzlePiece piece;
        piece.label = region.label;

        debug_print("[pipeline] extracting contour: " + std::to_string(region.label));
        time_stage(result.timings.contour_extraction, [&]() {
            piece.contour = find_contour_chain_approx_simple(
                piece_boundary.data,
                piece_boundary.width,
                piece_boundary.height);
        });

        debug_print("[pipeline] smoothing contour: " + std::to_string(region.label));
        time_stage(result.timings.contour_smoothing, [&]() {
            piece.smoothed_contour = smooth_contour(piece.contour, options.contour_smoothing_window);
        });

        debug_print("[pipeline] approximating enclosing circle: " + std::to_string(region.label));
        time_stage(result.timings.enclosing_circle, [&]() {
            auto [center, radius] = enclosing_circle_approx(piece.smoothed_contour);
            piece.circle_center = center;
            piece.circle_radius = radius;
        });

        debug_print("[pipeline] calculating radial signal: " + std::to_string(region.label));
        time_stage(result.timings.radial_signal, [&]() {
            piece.radial_signal = radial_signal(piece.smoothed_contour, piece.circle_center);
        });

        debug_print("[pipeline] smoothing signal: " + std::to_string(region.label));
        time_stage(result.timings.signal_smoothing, [&]() {
            piece.radial_signal = smooth_signal(piece.radial_signal, options.peak_smoothing_window);
        });

        debug_print("[pipeline] detecting peaks: " + std::to_string(region.label));
        time_stage(result.timings.peak_detection, [&]() {
            piece.triangular_peak_indices = find_triangular_peaks(
                piece.radial_signal,
                options.peak_min_prominence,
                options.peak_min_sharpness,
                options.peak_min_distance);
        });

        debug_print("[pipeline] classifying edges: " + std::to_string(region.label));
        time_stage(result.timings.edge_classification, [&]() {
            piece.edge_labels = classify_edges(
                piece.radial_signal,
                piece.triangular_peak_indices,
                options.tol_factor);
        });

        if (regions.size() == 1) {
            result.boundary = piece_boundary;
        }
        result.pieces.push_back(std::move(piece));
    }

    result.timings.total_seconds = total_timer.get();

#if DEBUG_LEVEL >= 1
    print_summary(options, result);
#endif

    return result;
}

#if DEBUG_LEVEL >= 1
void print_summary(const PipelineOptions& options, const PipelineResult& results)
{
    const auto print_timing = [](const char* label, double seconds) {
        std::cout << "  " << std::left << std::setw(22) << label
                  << std::right << std::fixed << std::setprecision(3)
                  << seconds * 1000.0 << " ms\n";
    };

    std::cout << "\n-- Pipeline output -------------------------\n";
    std::cout << "Input image      : " << options.input_image_path << '\n';
    std::cout << "Output directory : " << options.output_dir << '\n';
    std::cout << "Pieces found     : " << results.pieces.size() << '\n';

    if (results.pieces.empty()) {
        std::cout << "No puzzle pieces were detected.\n";
    }

    for (size_t i = 0; i < results.pieces.size(); ++i) {
        const PuzzlePiece& piece = results.pieces[i];

        std::cout << '\n';
        if (results.pieces.size() > 1) {
            std::cout << "Piece " << (i + 1) << " (label " << piece.label << ")\n";
        } else {
            std::cout << "Piece label      : " << piece.label << '\n';
        }

        std::cout << "  Edge labels          : " << edges_to_string(piece.edge_labels) << '\n';
        std::cout << "  Corner indices       : ["
                  << piece.triangular_peak_indices[0] << ", "
                  << piece.triangular_peak_indices[1] << ", "
                  << piece.triangular_peak_indices[2] << ", "
                  << piece.triangular_peak_indices[3] << "]\n";
        std::cout << "  Contour points       : " << piece.contour.size() << '\n';
        std::cout << "  Smoothed contour     : " << piece.smoothed_contour.size() << '\n';
        std::cout << "  Radial signal samples: " << piece.radial_signal.size() << '\n';
        std::cout << "  Circle center        : (" << piece.circle_center.a
                  << ", " << piece.circle_center.b << ")\n";
        std::cout << "  Circle radius        : " << piece.circle_radius << '\n';
    }

    std::cout << "\n-- Timings -------------------------------\n";
    print_timing("Total wall time", results.timings.total_seconds);
#if SUB_TIMINGS >= 1
    print_timing("Preprocessing", results.timings.preprocessing);
    print_timing("Cleaning", results.timings.cleaning);
    print_timing("Connected components", results.timings.connected_components);
    print_timing("Mask extraction", results.timings.mask_extraction);
    print_timing("Boundary extraction", results.timings.boundary_extraction);
    print_timing("Contour extraction", results.timings.contour_extraction);
    print_timing("Contour smoothing", results.timings.contour_smoothing);
    print_timing("Enclosing circle", results.timings.enclosing_circle);
    print_timing("Radial signal", results.timings.radial_signal);
    print_timing("Signal smoothing", results.timings.signal_smoothing);
    print_timing("Peak detection", results.timings.peak_detection);
    print_timing("Edge classification", results.timings.edge_classification);
#else
    std::cout << "  Sub timings disabled; compile with SUB_TIMINGS >= 1 to measure stages.\n";
#endif
}
#endif

#ifdef PIPELINE_BUILD_STANDALONE
int main(int argc, char** argv)
{
    PipelineOptions options;
    if (argc > 1) {
        options.input_image_path = argv[1];
    }
    if (argc > 2) {
        options.output_dir = argv[2];
    }

    try {
        (void)run(options);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Pipeline failed: " << e.what() << '\n';
        return 1;
    }
}
#endif
