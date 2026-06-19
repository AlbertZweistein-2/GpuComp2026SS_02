#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

#ifndef SUB_TIMINGS
#define SUB_TIMINGS 0
#endif

#ifndef PERSIST_TIMINGS
#define PERSIST_TIMINGS 0
#endif

#include "serial/pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "helpers.hpp"
#include "serial/boundary_extraction.hpp"
#include "serial/component_labeling.hpp"
#include "serial/contour_to_signal.hpp"
#include "serial/preprocessing.hpp"
#include "serial/signal_analysis.hpp"
#include "serial/visualization.hpp"
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
#if SUB_TIMINGS >= 1 || PERSIST_TIMINGS >= 1
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

std::string lowercase_extension(const fs::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return extension;
}

bool is_supported_image_file(const fs::path& path)
{
    if (!fs::is_regular_file(path)) {
        return false;
    }

    const std::string extension = lowercase_extension(path);
    return extension == ".jpg" ||
           extension == ".jpeg" ||
           extension == ".png" ||
           extension == ".bmp" ||
           extension == ".tga";
}

std::vector<fs::path> collect_images(const fs::path& directory)
{
    std::vector<fs::path> images;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
        if (is_supported_image_file(entry.path())) {
            images.push_back(entry.path());
        }
    }

    std::sort(images.begin(), images.end());
    return images;
}

#if PERSIST_TIMINGS >= 1
std::string env_or_empty(const char* name)
{
    const char* value = std::getenv(name);
    return value == nullptr ? "" : value;
}

std::string csv_value_or_fallback(const std::string& value, const std::string& fallback)
{
    return value.empty() ? fallback : value;
}

fs::path timings_csv_path(const PipelineOptions& options)
{
    const std::string configured_path = env_or_empty("PIPELINE_TIMINGS_CSV");
    if (!configured_path.empty()) {
        return project_path(configured_path);
    }
    return project_path(options.output_dir) / "serial_timings.csv";
}

bool needs_csv_header(const fs::path& path)
{
    return !fs::exists(path) || fs::file_size(path) == 0;
}

double milliseconds(double seconds)
{
    return seconds * 1000.0;
}

std::string image_resolution(int width, int height)
{
    return std::to_string(width) + "x" + std::to_string(height);
}

void append_timings_csv(const PipelineOptions& options, const PipelineResult& result, int width, int height)
{
    const fs::path input_path = project_path(options.input_image_path);
    const fs::path csv_path = timings_csv_path(options);
    fs::create_directories(csv_path.parent_path());

    const bool write_header = needs_csv_header(csv_path);
    std::ofstream csv(csv_path, std::ios::app);
    if (!csv) {
        throw std::runtime_error("Could not open timings CSV: " + csv_path.string());
    }

    if (write_header) {
        csv << "resolution,image,pieces,run,total_ms,preprocessing_ms,"
            << "connected_components_ms,boundary_extraction_ms,contour_extraction_ms,"
            << "contour_smoothing_ms,enclosing_circle_ms,radial_signal_ms,"
            << "signal_smoothing_ms,peak_detection_ms,edge_classification_ms,"
            << "visualization_ms\n";
    }

    const std::string pieces = csv_value_or_fallback(
        env_or_empty("PIPELINE_TIMING_PIECES"),
        std::to_string(result.pieces.size()));
    const std::string run = csv_value_or_fallback(env_or_empty("PIPELINE_TIMING_RUN"), "1");

    csv << image_resolution(width, height) << ','
        << input_path.filename().string() << ','
        << pieces << ','
        << run << ','
        << std::fixed << std::setprecision(3)
        << milliseconds(result.timings.total_seconds) << ','
        << milliseconds(result.timings.preprocessing) << ','
        << milliseconds(result.timings.connected_components) << ','
        << milliseconds(result.timings.boundary_extraction) << ','
        << milliseconds(result.timings.contour_extraction) << ','
        << milliseconds(result.timings.contour_smoothing) << ','
        << milliseconds(result.timings.enclosing_circle) << ','
        << milliseconds(result.timings.radial_signal) << ','
        << milliseconds(result.timings.signal_smoothing) << ','
        << milliseconds(result.timings.peak_detection) << ','
        << milliseconds(result.timings.edge_classification) << ','
        << milliseconds(result.timings.visualization) << '\n';
}
#endif

} // namespace

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
            std::cout << "Piece " << (i + 1) << " (label " << piece.region.label << ")\n";
        } else {
            std::cout << "Piece label      : " << piece.region.label << '\n';
        }

        std::cout << "  Edge labels          : " << edges_to_string(piece.edge_labels) << '\n';
        std::cout << "  Class label          : " << piece.class_label << '\n';
        std::cout << "  Corner indices       : ["
                  << piece.corner_indices[0] << ", "
                  << piece.corner_indices[1] << ", "
                  << piece.corner_indices[2] << ", "
                  << piece.corner_indices[3] << "]\n";
        std::cout << "  Contour points       : " << piece.contour.size() << '\n';
    }

    std::cout << "\n-- Timings -------------------------------\n";
    print_timing("Total wall time", results.timings.total_seconds);
#if SUB_TIMINGS >= 1 || PERSIST_TIMINGS >= 1
    std::vector<std::pair<std::string_view, double>> stage_timings = {
        {"Preprocessing", results.timings.preprocessing},
        {"Connected components", results.timings.connected_components},
        {"Boundary extraction", results.timings.boundary_extraction},
        {"Contour extraction", results.timings.contour_extraction},
        {"Contour smoothing", results.timings.contour_smoothing},
        {"Enclosing circle", results.timings.enclosing_circle},
        {"Radial signal", results.timings.radial_signal},
        {"Signal smoothing", results.timings.signal_smoothing},
        {"Peak detection", results.timings.peak_detection},
        {"Edge classification", results.timings.edge_classification},
        {"Visualization", results.timings.visualization},
    };

    std::sort(stage_timings.begin(), stage_timings.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.second > rhs.second;
              });

    for (const auto& [label, seconds] : stage_timings) {
        print_timing(label.data(), seconds);
    }
#else
    std::cout << "  Sub timings disabled; compile with SUB_TIMINGS >= 1 to measure stages.\n";
#endif
}
#endif

PipelineResult run(const PipelineOptions& options)
{
    Timer total_timer;
    total_timer.reset();

    const fs::path input_image_path = project_path(options.input_image_path);
    PipelineResult result;

    // STEP 1: Load input image
    debug_print("[pipeline] loading image: " + input_image_path.string());
    int width = 0;
    int height = 0;
    uint8_t* rgb = load_rgb_image(input_image_path, width, height);
    ImageU8 cleaned;

    // STEP 2: Preprocess RGB image into a cleaned binary mask
    debug_print("[pipeline] preprocessing");
    time_stage(result.timings.preprocessing, [&]() {
        preprocess(
            rgb,
            width,
            height,
            options.gaussian_kernel_size,
            options.gaussian_sigma,
            options.morphology_kernel_width,
            options.morphology_kernel_height,
            options.morphology_iterations,
            cleaned);
    });
    ImageI32 labels;
    std::vector<Region> regions;
    // STEP 3: Label connected components and collect piece regions
    debug_print("[pipeline] connected components");
    time_stage(result.timings.connected_components, [&]() {
        connected_components(cleaned, options.min_region_area, labels, regions);
    });
    debug_print("[pipeline] found " + std::to_string(regions.size()) + " regions");

    result.pieces.reserve(regions.size());
    const PuzzleLookupTable puzzle_lookup;


    for (const auto& region : regions) {
        ImageU8 piece_boundary;
        Coordinate<int> boundary_offset{0, 0};
        // STEP 4: Extract piece boundary directly from the label image
        debug_print("[pipeline] extracting piece boundary: " + std::to_string(region.label));
        time_stage(result.timings.boundary_extraction, [&]() {
            get_piece_boundary_mask_from_labels(labels, region, piece_boundary, boundary_offset);
        });

        PuzzlePiece piece;
        piece.region = region;

        // STEP 5: Convert boundary mask to contour points
        debug_print("[pipeline] extracting contour: " + std::to_string(region.label));
        time_stage(result.timings.contour_extraction, [&]() {
            find_contour_chain_approx_simple(
                piece_boundary.data,
                piece_boundary.width,
                piece_boundary.height,
                piece.contour);
            for (Coordinate<int>& point : piece.contour) {
                point.a += boundary_offset.a;
                point.b += boundary_offset.b;
            }
        });

        CoordinateVector<int> smoothed_contour;
        // STEP 6: Smooth contour before signal extraction
        debug_print("[pipeline] smoothing contour: " + std::to_string(region.label));
        time_stage(result.timings.contour_smoothing, [&]() {
            smoothed_contour = piece.contour;
            smooth_contour(smoothed_contour, options.contour_smoothing_window);
        });

        Coordinate<float> circle_center{0.0f, 0.0f};
        // STEP 7: Approximate enclosing circle center
        debug_print("[pipeline] approximating enclosing circle: " + std::to_string(region.label));
        time_stage(result.timings.enclosing_circle, [&]() {
            float radius = 0.0f;
            enclosing_circle_approx(smoothed_contour, circle_center, radius);
        });

        Signal raw_radial_signal;
        // STEP 8: Convert contour to radial signal
        debug_print("[pipeline] calculating radial signal: " + std::to_string(region.label));
        time_stage(result.timings.radial_signal, [&]() {
            radial_signal(smoothed_contour, circle_center, raw_radial_signal);
        });

        Signal smoothed_radial_signal;
        // STEP 9: Smooth radial signal for corner detection
        debug_print("[pipeline] smoothing signal: " + std::to_string(region.label));
        time_stage(result.timings.signal_smoothing, [&]() {
            smoothed_radial_signal = smooth_signal(raw_radial_signal, options.peak_smoothing_window);
        });

        // STEP 10: Detect the four corner peaks
        debug_print("[pipeline] detecting peaks: " + std::to_string(region.label));
        time_stage(result.timings.peak_detection, [&]() {
            piece.corner_indices = find_triangular_peaks(
                smoothed_radial_signal,
                options.peak_min_prominence,
                options.peak_min_sharpness,
                options.peak_min_distance);
        });

        // STEP 11: Classify edge types and lookup piece class
        debug_print("[pipeline] classifying edges: " + std::to_string(region.label));
        time_stage(result.timings.edge_classification, [&]() {
            piece.edge_labels = classify_edges(
                raw_radial_signal,
                piece.corner_indices,
                options.tol_factor);
            const std::string edge_label_string = edges_to_string(piece.edge_labels);
            piece.class_label = puzzle_lookup.getClassLabel(edge_label_string);
        });

        result.pieces.push_back(std::move(piece));
    }

    // STEP 12: Draw final contour and bounding-box overlay image
    time_stage(result.timings.visualization, [&]() {
        const fs::path output_dir = project_path(options.output_dir);
        fs::create_directories(output_dir);
        const fs::path overlay_path = output_dir / (input_image_path.stem().string() + "_overlays.png");
        draw_piece_overlays(rgb, width, height, result.pieces, overlay_path.string());
    });
    stbi_image_free(rgb);

    result.timings.total_seconds = total_timer.get();

#if PERSIST_TIMINGS >= 1
    append_timings_csv(options, result, width, height);
#endif

#if DEBUG_LEVEL >= 1
    print_summary(options, result);
#endif

    return result;
}

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
        const fs::path input_path = project_path(options.input_image_path);

        if (fs::is_directory(input_path)) {
            const std::vector<fs::path> images = collect_images(input_path);
            if (images.empty()) {
                throw std::runtime_error("No supported images found in folder: " + input_path.string());
            }

            debug_print("Running pipeline on " + std::to_string(images.size()) +
                        " image(s) in: " + input_path.string());
            for (const fs::path& image_path : images) {
                PipelineOptions image_options = options;
                image_options.input_image_path = image_path.string();
                (void)run(image_options);
            }
        } else if (fs::is_regular_file(input_path)) {
            (void)run(options);
        } else {
            throw std::runtime_error("Input path is not a file or folder: " + input_path.string());
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Pipeline failed: " << e.what() << '\n';
        return 1;
    }
}
#endif
