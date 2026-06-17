#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

#ifndef SUB_TIMINGS
#define SUB_TIMINGS 0
#endif

#include "parallel/pipeline.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "helpers.hpp"
#include "parallel/boundary_extraction.cuh"
#include "parallel/component_labeling.cuh"
#include "parallel/preprocessing.hpp"
#include "serial/contour_to_signal.hpp"
#include "serial/signal_analysis.hpp"
#include "serial/visualization.hpp"
#include "stb_image.h"

namespace fs = std::filesystem;

namespace {

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(                                        \
                std::string("CUDA error: ") + cudaGetErrorString(err) +       \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
        }                                                                    \
    } while (0)

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

void print_timing(const char* label, double seconds)
{
    std::cout << "  " << std::left << std::setw(22) << label
              << std::right << std::fixed << std::setprecision(3)
              << seconds * 1000.0 << " ms\n";
}

#if DEBUG_LEVEL >= 1
void print_summary_cuda(const PipelineOptions& options, const PipelineResult& results)
{
    std::cout << "\n-- CUDA Pipeline output --------------------\n";
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
#if SUB_TIMINGS >= 1
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

} // namespace

PipelineResult run_cuda(const PipelineOptions& options)
{
    Timer total_timer;
    total_timer.reset();

    const fs::path input_image_path = project_path(options.input_image_path);
    PipelineResult result;

    debug_print("[cuda pipeline] loading image: " + input_image_path.string());
    int width = 0;
    int height = 0;
    std::unique_ptr<uint8_t, decltype(&stbi_image_free)> rgb(
        load_rgb_image(input_image_path, width, height),
        stbi_image_free);

    const int total_pixels = width * height;

    thrust::device_vector<uint8_t> d_cleaned(static_cast<size_t>(total_pixels));
    thrust::device_vector<int> d_labels(static_cast<size_t>(total_pixels));
    thrust::device_vector<int> d_compact_labels(static_cast<size_t>(total_pixels));
    thrust::device_vector<int> d_changed(1);
    thrust::device_vector<int> d_areas(static_cast<size_t>(total_pixels) + 1);
    thrust::device_vector<uint8_t> d_piece_mask(static_cast<size_t>(total_pixels));
    thrust::device_vector<uint8_t> d_piece_boundary(static_cast<size_t>(total_pixels));

    uint8_t* d_cleaned_ptr = thrust::raw_pointer_cast(d_cleaned.data());
    int* d_labels_ptr = thrust::raw_pointer_cast(d_labels.data());
    int* d_compact_labels_ptr = thrust::raw_pointer_cast(d_compact_labels.data());
    int* d_changed_ptr = thrust::raw_pointer_cast(d_changed.data());
    int* d_areas_ptr = thrust::raw_pointer_cast(d_areas.data());
    uint8_t* d_piece_mask_ptr = thrust::raw_pointer_cast(d_piece_mask.data());
    uint8_t* d_piece_boundary_ptr = thrust::raw_pointer_cast(d_piece_boundary.data());

    debug_print("[cuda pipeline] preprocessing and cleaning");
    time_stage(result.timings.preprocessing, [&]() {
        preprocess_cuda_device(
            rgb.get(),
            d_cleaned_ptr,
            width,
            height,
            options.gaussian_kernel_size,
            options.gaussian_sigma,
            options.morphology_kernel_width,
            options.morphology_kernel_height,
            options.morphology_iterations);
    });

    std::vector<Region> regions;
    debug_print("[cuda pipeline] connected components");
    time_stage(result.timings.connected_components, [&]() {
        connected_components_cuda_device_raw(
            d_cleaned_ptr,
            d_labels_ptr,
            d_changed_ptr,
            d_areas_ptr,
            width,
            height,
            options.min_region_area,
            total_pixels + 1);
        const int num_components = compact_labels_cuda_device(
            d_labels_ptr,
            d_compact_labels_ptr,
            total_pixels);
        regions = build_regions_from_labels_cuda(
            d_compact_labels_ptr,
            width,
            height,
            num_components);
    });
    debug_print("[cuda pipeline] found " + std::to_string(regions.size()) + " regions");

    result.pieces.reserve(regions.size());
    const PuzzleLookupTable puzzle_lookup;

    std::cout << "Number of puzzle pieces detected: " << regions.size() << '\n';
    for (const auto& region : regions) {
        std::cout << "  Label: " << region.label
                  << ", Area: " << region.area
                  << ", Bounding box: (" << region.x << ", " << region.y
                  << "), Width: " << region.width
                  << ", Height: " << region.height
                  << '\n';
    }

    ImageU8 piece_boundary;
    piece_boundary.width = width;
    piece_boundary.height = height;
    piece_boundary.data.resize(static_cast<size_t>(total_pixels));

    for (const auto& region : regions) {
        debug_print("[cuda pipeline] extracting piece boundary: " + std::to_string(region.label));
        time_stage(result.timings.boundary_extraction, [&]() {
            extract_piece_mask_cuda_device(
                d_compact_labels_ptr,
                d_piece_mask_ptr,
                region.label,
                total_pixels);
            get_external_boundary_mask_cuda_device(
                d_piece_mask_ptr,
                d_piece_boundary_ptr,
                width,
                height);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(
                piece_boundary.data.data(),
                d_piece_boundary_ptr,
                static_cast<size_t>(total_pixels) * sizeof(uint8_t),
                cudaMemcpyDeviceToHost));
        });

        PuzzlePiece piece;
        piece.region = region;

        debug_print("[cuda pipeline] extracting contour: " + std::to_string(region.label));
        time_stage(result.timings.contour_extraction, [&]() {
            find_contour_chain_approx_simple(
                piece_boundary.data,
                piece_boundary.width,
                piece_boundary.height,
                piece.contour);
        });

        CoordinateVector<int> smoothed_contour;
        debug_print("[cuda pipeline] smoothing contour: " + std::to_string(region.label));
        time_stage(result.timings.contour_smoothing, [&]() {
            smoothed_contour = piece.contour;
            smooth_contour(smoothed_contour, options.contour_smoothing_window);
        });

        Coordinate<float> circle_center{0.0f, 0.0f};
        debug_print("[cuda pipeline] approximating enclosing circle: " + std::to_string(region.label));
        time_stage(result.timings.enclosing_circle, [&]() {
            float radius = 0.0f;
            enclosing_circle_approx(smoothed_contour, circle_center, radius);
        });

        Signal raw_radial_signal;
        debug_print("[cuda pipeline] calculating radial signal: " + std::to_string(region.label));
        time_stage(result.timings.radial_signal, [&]() {
            radial_signal(smoothed_contour, circle_center, raw_radial_signal);
        });

        Signal smoothed_radial_signal;
        debug_print("[cuda pipeline] smoothing signal: " + std::to_string(region.label));
        time_stage(result.timings.signal_smoothing, [&]() {
            smoothed_radial_signal = smooth_signal(raw_radial_signal, options.peak_smoothing_window);
        });

        debug_print("[cuda pipeline] detecting peaks: " + std::to_string(region.label));
        time_stage(result.timings.peak_detection, [&]() {
            piece.corner_indices = find_triangular_peaks(
                smoothed_radial_signal,
                options.peak_min_prominence,
                options.peak_min_sharpness,
                options.peak_min_distance);
        });

        debug_print("[cuda pipeline] classifying edges: " + std::to_string(region.label));
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

    time_stage(result.timings.visualization, [&]() {
        const fs::path output_dir = project_path(options.output_dir);
        fs::create_directories(output_dir);
        const fs::path overlay_path = output_dir / (input_image_path.stem().string() + "_cuda_overlays.png");
        draw_piece_overlays(rgb.get(), width, height, result.pieces, overlay_path.string());
    });

    result.timings.total_seconds = total_timer.get();

#if DEBUG_LEVEL >= 1
    print_summary_cuda(options, result);
#endif

    return result;
}

#ifdef CUDA_PIPELINE_BUILD_STANDALONE
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
        (void)run_cuda(options);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CUDA pipeline failed: " << e.what() << '\n';
        return 1;
    }
}
#endif
