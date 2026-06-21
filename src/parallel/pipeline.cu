// ----------------------------------------------------------------------
// COMPILE FLAGS
#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

#ifndef SUB_TIMINGS
#define SUB_TIMINGS 0
#endif

#ifndef PERSIST_TIMINGS
#define PERSIST_TIMINGS 0
#endif

#ifndef ENABLE_NVTX
#define ENABLE_NVTX 0
#endif
// ----------------------------------------------------------------------
// HEADER INCLUDES
#include "parallel/pipeline.cuh"
#include "helpers.hpp"
#include "parallel/boundary_extraction.cuh"
#include "parallel/component_labeling.cuh"
#include "parallel/contour_to_signal.cuh"
#include "parallel/preprocessing.cuh"
#include "parallel/signal_analysis.cuh"
#include "parallel/visualization.hpp"
#include "stb_image.h"
// ----------------------------------------------------------------------
// STANDARD LIBRARY INCLUDES
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>
// ----------------------------------------------------------------------
// GPU SPECIFIC INCLUDES
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#if ENABLE_NVTX >= 1
#include <nvtx3/nvToolsExt.h>
#endif // NVTX allows profiling of CUDA kernels in NVIDIA Nsight Systems and Nsight Compute.
// ----------------------------------------------------------------------


namespace fs = std::filesystem;

namespace
{
    // Algorithm parameters shared across the CUDA pipeline. Most of these
    // mirror the serial pipeline so performance changes are easier to compare.
    constexpr int GAUSSIAN_KERNEL_SIZE = 3; // KSize must be odd and >= 3!
    constexpr float GAUSSIAN_SIGMA = 1.0f;
    constexpr int MIN_REGION_AREA = 2000;
    constexpr int CONTOUR_SMOOTHING_WINDOW = 5;
    constexpr int PEAK_SMOOTHING_WINDOW = 5;
    constexpr float PEAK_MIN_PROMINENCE = 5.0f;
    constexpr float PEAK_MIN_SHARPNESS = 20.0f;
    constexpr int PEAK_MIN_DISTANCE = 5;
    constexpr float EDGE_TOL_FACTOR = 0.1f;
    constexpr double REFERENCE_IMAGE_HEIGHT = 5100.0;

    // Scale the minimum region area based on the input image height relative to a reference height.
    // Area scales quadratically with image height so filtering behaves similarly
    // on downscaled or upscaled inputs.
    int scaled_min_region_area(int height)
    {
        const double scale = static_cast<double>(height) / REFERENCE_IMAGE_HEIGHT;
        return std::max(1, static_cast<int>(MIN_REGION_AREA * scale * scale + 0.5));
    }

#if ENABLE_NVTX >= 1
    class NvtxRange
    {
    public:
        explicit NvtxRange(const char *label)
        {
            nvtxRangePushA(label);
        }

        ~NvtxRange()
        {
            nvtxRangePop();
        }
    };
#endif
    // Get absolute path of a file or directory
    fs::path project_path(const std::string &path)
    {
        // Keep CLI defaults relative to the current project directory while
        // still accepting absolute paths from scripts.
        fs::path p(path);
        return p.is_absolute() ? p : fs::current_path() / p;
    }

    // ---
    // Measure the execution time of a stage and accumulate it into the provided seconds reference.
    // Compile-flag controlled to enable/disable NVTX ranges and timing measurements.
    template <typename Fn>
    void time_stage(const char *label, double &seconds, Fn &&fn)
    {
#if ENABLE_NVTX >= 1
        // NVTX ranges make the same stage names visible in Nsight Systems.
        NvtxRange range(label);
#else
        (void)label;
#endif
#if SUB_TIMINGS >= 1 || PERSIST_TIMINGS >= 1
        // Stage timing is compiled out unless requested, avoiding timer noise
        // in the default build.
        Timer timer;
        timer.reset();
        fn();
        seconds += timer.get();
#else
        fn();
#endif
    }
    // ---

    // ---
    // Debug print function controlled by DEBUG_LEVEL compile flag.
#if DEBUG_LEVEL >= 1
    void debug_print(std::string_view message)
    {
        std::cout << message << '\n';
    }
#else
#define debug_print(message) ((void)0)
#endif
    // ---

    // Allocate device buffers whose sizes are only known after contour tracing.
    // Calling resize here centralizes allocation cost under the CUDA setup
    // bucket instead of charging it to later compute stages.
    void allocate_batched_cuda_scratch(
        const BatchedContours &batch,
        BatchedContourCudaScratch &contour_scratch,
        SignalCudaScratch &signal_scratch)
    {
        const std::size_t point_count = batch.points.size();
        const std::size_t piece_count = batch.offsets.size();

        contour_scratch.points.resize(point_count);
        contour_scratch.smoothed.resize(point_count);
        contour_scratch.offsets.resize(piece_count);
        contour_scratch.lengths.resize(piece_count);
        contour_scratch.centers.resize(piece_count);
        contour_scratch.signals.resize(point_count);

        signal_scratch.smoothed.resize(point_count);
        signal_scratch.peak_mask.resize(point_count);
    }

    // Load an RGB image from a file using stb_image.h and return a pointer to the pixel data.
    uint8_t *load_rgb_image(const fs::path &path, int &width, int &height)
    {
        int channels = 0;
        // Force RGB output so all downstream kernels can assume three channels.
        uint8_t *data = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
        if (data == nullptr)
        {
            throw std::runtime_error("Could not load image: " + path.string() + "\n" + stbi_failure_reason() + "\n");
        }
        return data;
    }

    // Convert the file extension of a path to lowercase for case-insensitive comparison.
    std::string lowercase_extension(const fs::path &path)
    {
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch)
                       { return static_cast<char>(std::tolower(ch)); });
        return extension;
    }

    // Check if a file has a supported image extension (jpg, jpeg, png, bmp, tga).
    bool is_supported_image_file(const fs::path &path)
    {
        if (!fs::is_regular_file(path))
        {
            return false;
        }

        const std::string extension = lowercase_extension(path);
        return extension == ".jpg" ||
               extension == ".jpeg" ||
               extension == ".png" ||
               extension == ".bmp" ||
               extension == ".tga";
    }

    // Collect all supported image files from a directory and return them in a sorted vector.
    std::vector<fs::path> collect_images(const fs::path &directory)
    {
        std::vector<fs::path> images;
        for (const fs::directory_entry &entry : fs::directory_iterator(directory))
        {
            if (is_supported_image_file(entry.path()))
            {
                images.push_back(entry.path());
            }
        }

        std::sort(images.begin(), images.end());
        // Deterministic order keeps benchmark CSV runs reproducible.
        return images;
    }
    
    // ---
    // Convert seconds to milliseconds for timing output.
#if DEBUG_LEVEL >= 1 || PERSIST_TIMINGS >= 1
    double milliseconds(double seconds)
    {
        return seconds * 1000.0;
    }
#endif
    // ---


    // ---
    // CSV output functions for persisting timing results to a file.
#if PERSIST_TIMINGS >= 1
    std::string env_or_empty(const char *name)
    {
        const char *value = std::getenv(name);
        return value == nullptr ? "" : value;
    }

    std::string csv_value_or_fallback(const std::string &value, const std::string &fallback)
    {
        return value.empty() ? fallback : value;
    }

    fs::path timings_csv_path(const PipelineOptions &options)
    {
        const std::string configured_path = env_or_empty("PIPELINE_TIMINGS_CSV");
        if (!configured_path.empty())
        {
            // Allows benchmark scripts to collect timings outside the normal
            // output directory.
            return project_path(configured_path);
        }
        return project_path(options.output_dir) / "cuda_timings.csv";
    }

    bool needs_csv_header(const fs::path &path)
    {
        return !fs::exists(path) || fs::file_size(path) == 0;
    }

    std::string image_resolution(int width, int height)
    {
        return std::to_string(width) + "x" + std::to_string(height);
    }

    void append_timings_csv(const PipelineOptions &options, const PipelineResult &result, int width, int height)
    {
        const fs::path input_path = project_path(options.input_image_path);
        const fs::path csv_path = timings_csv_path(options);
        fs::create_directories(csv_path.parent_path());

        const bool write_header = needs_csv_header(csv_path);
        std::ofstream csv(csv_path, std::ios::app);
        if (!csv)
        {
            throw std::runtime_error("Could not open timings CSV: " + csv_path.string());
        }

        if (write_header)
        {
            csv << "resolution,image,pieces,run,total_ms,cuda_setup_ms,preprocessing_ms,"
                << "connected_components_ms,boundary_extraction_ms,contour_extraction_ms,"
                << "contour_smoothing_ms,enclosing_circle_ms,radial_signal_ms,"
                << "signal_smoothing_ms,peak_detection_ms,edge_classification_ms,"
                << "visualization_ms\n";
        }

        const std::string pieces = csv_value_or_fallback(
            env_or_empty("PIPELINE_TIMING_PIECES"),
            std::to_string(result.pieces.size()));
        // `run` and `pieces` can be supplied by benchmark scripts so repeated
        // runs can be grouped without changing pipeline code.
        const std::string run = csv_value_or_fallback(env_or_empty("PIPELINE_TIMING_RUN"), "1");

        csv << image_resolution(width, height) << ','
            << input_path.filename().string() << ','
            << pieces << ','
            << run << ','
            << std::fixed << std::setprecision(3)
            << milliseconds(result.timings.total_seconds) << ','
            << milliseconds(result.timings.cuda_setup) << ','
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
    // ---

    // ---
    // Debug print function controlled by DEBUG_LEVEL compile flag.
#if DEBUG_LEVEL >= 1
    void print_timing(const char *label, double seconds)
    {
        std::cout << "  " << std::left << std::setw(22) << label
                  << std::right << std::fixed << std::setprecision(3)
                  << milliseconds(seconds) << " ms\n";
    }

    void print_summary_cuda(const PipelineOptions &options, const PipelineResult &results)
    {
        std::cout << "\n-- CUDA Pipeline output --------------------\n";
        std::cout << "Input image      : " << options.input_image_path << '\n';
        std::cout << "Output directory : " << options.output_dir << '\n';
        std::cout << "Pieces found     : " << results.pieces.size() << '\n';

        if (results.pieces.empty())
        {
            std::cout << "No puzzle pieces were detected.\n";
        }

        for (size_t i = 0; i < results.pieces.size(); ++i)
        {
            const PuzzlePiece &piece = results.pieces[i];

            std::cout << '\n';
            if (results.pieces.size() > 1)
            {
                std::cout << "Piece " << (i + 1) << " (label " << piece.region.label << ")\n";
            }
            else
            {
                std::cout << "Piece label      : " << piece.region.label << '\n';
            }

            std::cout << "  Edge labels          : " << edges_to_string_cuda(piece.edge_labels) << '\n';
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
        // Sort by stage cost so the slowest pipeline parts are immediately
        // visible in normal console output.
        std::vector<std::pair<std::string_view, double>> stage_timings = {
            {"CUDA setup", results.timings.cuda_setup},
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
                  [](const auto &lhs, const auto &rhs)
                  {
                      return lhs.second > rhs.second;
                  });

        for (const auto &[label, seconds] : stage_timings)
        {
            print_timing(label.data(), seconds);
        }

        double accounted_seconds = 0.0;
        for (const auto &[_, seconds] : stage_timings)
        {
            accounted_seconds += seconds;
        }
        // This includes host work inside the pipeline timer that is not wrapped
        // in an explicit stage, such as plan construction and batch packing.
        print_timing("Untracked overhead", results.timings.total_seconds - accounted_seconds);
#else
        std::cout << "  Sub timings disabled; compile with SUB_TIMINGS >= 1 to measure stages.\n";
#endif
    }
#endif
    // ---

} // namespace

// ----------------------------------------------------------------------
// MAIN PIPELINE FUNCTION
PipelineResult run_cuda(const PipelineOptions &options)
{
    const fs::path input_image_path = project_path(options.input_image_path);

    PipelineResult result;

    // STEP 1: Load input image
    debug_print("[cuda pipeline] loading image: " + input_image_path.string());
    int width = 0;
    int height = 0;
    std::unique_ptr<uint8_t, decltype(&stbi_image_free)> rgb(nullptr, stbi_image_free);
    rgb.reset(load_rgb_image(input_image_path, width, height));

    // Start wall-clock timing after image decode. The final overlay file write
    // also happens after this timer is captured, so pipeline timings focus on
    // processing and overlay rendering rather than I/O.
    Timer total_timer;
    total_timer.reset();

    const int total_pixels = width * height;
    const std::size_t pixel_count = static_cast<std::size_t>(total_pixels);
    const std::size_t rgb_value_count = pixel_count * 3;
    const int min_region_area = scaled_min_region_area(height);

    // Host-side pipeline state. Sizes that depend on detection results are
    // filled as the pipeline discovers regions and contours.
    // Raw/smoothed radial signals stay flat and use batched_contours offsets.
    std::vector<Region> regions;
    std::vector<PieceBoundaryMask> piece_boundaries;
    std::vector<uint8_t> piece_boundary_data;
    BatchedContours batched_contours;
    Signal flat_raw_radial_signal;
    Signal flat_smoothed_radial_signal;
    std::vector<Corners> batched_corner_indices;
    std::vector<EdgeLabels> batched_edge_labels;
    const PuzzleLookupTable puzzle_lookup;

    // CUDA buffers. Declaring thrust::device_vector does not allocate device
    // memory; the resize/copy calls inside CUDA setup below do.
    thrust::device_vector<uint8_t> d_rgb;
    thrust::device_vector<uint8_t> d_cleaned;
    thrust::device_vector<int> d_compact_labels;
    // Raw pointers are captured after resize because kernels take plain CUDA
    // pointers, while device_vector owns the allocation lifetime.
    uint8_t *d_rgb_ptr = nullptr;
    uint8_t *d_cleaned_ptr = nullptr;
    int *d_compact_labels_ptr = nullptr;

    PreprocessCudaScratch preprocess_scratch;
    ConnectedComponentsCudaScratch component_scratch;
    RegionBuildCudaScratch region_build_scratch;
    BoundaryExtractionCudaScratch boundary_scratch;
    BatchedContourCudaScratch batched_contour_scratch;
    SignalCudaScratch signal_scratch;

    // STEP 1b: Initialize CUDA, allocate image-size CUDA buffers, and upload
    // the input image. This is intentionally the first CUDA runtime call.
    debug_print("[cuda pipeline] allocating CUDA buffers and uploading image");
    time_stage("CUDA setup", result.timings.cuda_setup, [&]()
               {
        // These buffers are image-sized and can be allocated before any
        // content-dependent piece count is known.
        d_rgb.resize(rgb_value_count);
        d_cleaned.resize(pixel_count);
        d_compact_labels.resize(pixel_count);
        allocate_preprocess_cuda_scratch(width, height, preprocess_scratch);
        allocate_component_labeling_cuda_scratch(
            width,
            height,
            component_scratch,
            region_build_scratch);
        // Pageable host upload remains intentional; earlier pinned-memory tests
        // did not improve this single-image path.
        thrust::copy(rgb.get(), rgb.get() + rgb_value_count, d_rgb.begin());

        d_rgb_ptr = thrust::raw_pointer_cast(d_rgb.data());
        d_cleaned_ptr = thrust::raw_pointer_cast(d_cleaned.data());
        d_compact_labels_ptr = thrust::raw_pointer_cast(d_compact_labels.data()); });

    // STEP 2: Preprocess RGB image into a cleaned binary mask
    debug_print("[cuda pipeline] preprocessing and cleaning");
    time_stage("Preprocessing", result.timings.preprocessing, [&]()
               { preprocess_cuda(
                     d_rgb_ptr,
                     d_cleaned_ptr,
                     width,
                     height,
                     GAUSSIAN_KERNEL_SIZE,
                     GAUSSIAN_SIGMA,
                     preprocess_scratch); });

    // STEP 3: Label connected components and collect piece regions
    debug_print("[cuda pipeline] connected components");
    time_stage("Connected components", result.timings.connected_components, [&]()
               {
        // CCL produces dense labels on device; region construction filters
        // small components and copies back only Region metadata.
        connected_components_buf_cuda_device(
            d_cleaned_ptr,
            d_compact_labels_ptr,
            width,
            height,
            component_scratch);
        build_regions_from_labels_cuda(
            d_compact_labels_ptr,
            width,
            height,
            min_region_area,
            region_build_scratch,
            regions); });
    debug_print("[cuda pipeline] found " + std::to_string(regions.size()) + " regions");

    result.pieces.resize(regions.size());

    // Boundary extraction needs compact per-piece windows. The plan is built on
    // host from Region metadata before allocating the matching device buffers.
    BoundaryExtractionPlan boundary_plan = build_boundary_extraction_plan(
        regions,
        width,
        height,
        piece_boundaries);

    // Boundary buffers depend on the discovered regions, so this setup block
    // runs after CCL but still charges explicit CUDA allocation to CUDA setup.
    debug_print("[cuda pipeline] allocating boundary extraction CUDA buffers");
    time_stage("CUDA setup", result.timings.cuda_setup, [&]()
               { allocate_boundary_extraction_cuda_scratch(boundary_plan, boundary_scratch); });

    // STEP 4: Extract all piece boundary masks from the compact label image
    debug_print("[cuda pipeline] extracting piece boundaries");
    time_stage("Boundary extraction", result.timings.boundary_extraction, [&]()
               {
        // The output is copied back as packed per-piece masks because Moore
        // contour tracing is still CPU-side.
        extract_piece_boundary_masks_from_labels_cuda(
            d_compact_labels_ptr,
            boundary_plan,
            boundary_scratch,
            width,
            height,
            piece_boundary_data); });

    // STEP 5: Moore tracing stays on CPU because it produces connected contour
    // order. The later CUDA stages consume these contours as one flat batch.
    debug_print("[cuda pipeline] extracting contours");
    time_stage("Contour extraction", result.timings.contour_extraction, [&]()
               {
        const std::ptrdiff_t region_count = static_cast<std::ptrdiff_t>(regions.size());
#ifdef _OPENMP
// Each region owns a separate output slot and boundary mask, so CPU contour
// tracing can run independently per piece.
#pragma omp parallel for schedule(dynamic)
#endif
        for (std::ptrdiff_t region_idx = 0; region_idx < region_count; ++region_idx)
        {
            const std::size_t idx = static_cast<std::size_t>(region_idx);
            const Region &region = regions[idx];
            const PieceBoundaryMask &piece_boundary = piece_boundaries[idx];
            const uint8_t *piece_boundary_ptr = piece_boundary_data.data() + piece_boundary.data_offset;

            PuzzlePiece piece;
            piece.region = region;

            // Convert host boundary mask to a connected global contour.
            // This uses Moore tracing again because angle-sorted boundary pixels
            // were not a valid connected contour for concave puzzle pieces.
            find_contour_chain_approx_simple_cuda(
                piece_boundary_ptr,
                piece_boundary.width,
                piece_boundary.height,
                piece.contour);
            for (Coordinate<int> &point : piece.contour)
            {
                // Boundary masks are local to their cropped region. Shift the
                // contour back into full-image coordinates for visualization
                // and later geometric analysis.
                point.a += piece_boundary.image_offset.a;
                point.b += piece_boundary.image_offset.b;
            }

            result.pieces[idx] = std::move(piece);
        } });

    // Pack CPU contours before allocating the contour/signal CUDA scratch; the
    // required device sizes are only known after Moore tracing.
    build_batched_contours(result.pieces, batched_contours);

    debug_print("[cuda pipeline] allocating batched contour/signal CUDA buffers");
    time_stage("CUDA setup", result.timings.cuda_setup, [&]()
               {
        // This is a second setup range because contour counts and lengths are
        // only known after CPU Moore tracing.
        allocate_batched_cuda_scratch(
            batched_contours,
            batched_contour_scratch,
            signal_scratch); });

    // STEP 6: Smooth all contours in one batched GPU launch.
    debug_print("[cuda pipeline] smoothing contours");
    time_stage("Contour smoothing", result.timings.contour_smoothing, [&]()
               {
        smooth_contours_batched_cuda(
            batched_contours,
            CONTOUR_SMOOTHING_WINDOW,
            batched_contour_scratch); });

    // STEP 7: Approximate all enclosing circle centers in one batched GPU launch.
    debug_print("[cuda pipeline] approximating enclosing circles");
    time_stage("Enclosing circle", result.timings.enclosing_circle, [&]()
               { enclosing_circle_centers_batched_cuda(
                     batched_contours,
                     batched_contour_scratch); });

    // STEP 8: Convert all smoothed contours to radial signals in one batched GPU launch.
    debug_print("[cuda pipeline] calculating radial signals");
    time_stage("Radial signal", result.timings.radial_signal, [&]()
               { radial_signals_batched_cuda(
                     batched_contours,
                     batched_contour_scratch,
                     // Host copy is needed by CPU edge classification.
                     flat_raw_radial_signal); });

    // STEP 9: Smooth all radial signals in one batched GPU launch.
    debug_print("[cuda pipeline] smoothing signals");
    time_stage("Signal smoothing", result.timings.signal_smoothing, [&]()
               { smooth_signals_batched_cuda(
                     batched_contour_scratch.signals,
                     batched_contour_scratch.offsets,
                     batched_contour_scratch.lengths,
                     batched_contours.max_length,
                     PEAK_SMOOTHING_WINDOW,
                     flat_smoothed_radial_signal,
                     signal_scratch); });

    // STEP 10: Detect corner peaks for all pieces using one batched GPU maxima pass.
    debug_print("[cuda pipeline] detecting peaks");
    time_stage("Peak detection", result.timings.peak_detection, [&]()
               { find_triangular_peaks_batched_cuda(
                     // Host smoothed signal is used for CPU prominence scoring;
                     // device smoothed signal is used for the maxima mask.
                     flat_smoothed_radial_signal,
                     signal_scratch.smoothed,
                     batched_contours.offsets,
                     batched_contours.lengths,
                     batched_contour_scratch.offsets,
                     batched_contour_scratch.lengths,
                     batched_contours.max_length,
                     signal_scratch,
                     PEAK_MIN_PROMINENCE,
                     PEAK_MIN_SHARPNESS,
                     PEAK_MIN_DISTANCE,
                     batched_corner_indices); });

    // STEP 11: Classify all piece edges from the flat radial signal.
    debug_print("[cuda pipeline] classifying edges");
    time_stage("Edge classification", result.timings.edge_classification, [&]()
               {
        // Kept on CPU: profiling showed the isolated GPU classifier's launch
        // and tiny copy overhead outweighed the saved raw-signal copy.
        classify_edges_batched_cuda(
            flat_raw_radial_signal,
            batched_contours.offsets,
            batched_contours.lengths,
            batched_corner_indices,
            EDGE_TOL_FACTOR,
            batched_edge_labels);

        for (std::size_t piece_idx = 0; piece_idx < result.pieces.size(); ++piece_idx)
        {
            // Attach batched signal-analysis results back to the user-facing
            // per-piece structs.
            PuzzlePiece &piece = result.pieces[piece_idx];
            piece.corner_indices = batched_corner_indices[piece_idx];
            piece.edge_labels = batched_edge_labels[piece_idx];
            const std::string edge_label_string = edges_to_string_cuda(piece.edge_labels);
            piece.class_label = puzzle_lookup.getClassLabel(edge_label_string);
        } });

    // STEP 12: Draw final contour and bounding-box overlay image
    time_stage("Visualization", result.timings.visualization, [&]()
               {
        // Rendering mutates the loaded RGB buffer in-place and avoids another
        // full image copy before writing the overlay.
        parallel_visualization::render_piece_overlays(
            rgb.get(),
            width,
            height,
            result.pieces); });

    result.timings.total_seconds = total_timer.get();

    // Save overlay image to output directory with "_cuda_overlays" suffix.
    const fs::path output_dir = project_path(options.output_dir);
    fs::create_directories(output_dir);
    const fs::path overlay_path = output_dir / (input_image_path.stem().string() + "_cuda_overlays.bmp");
    parallel_visualization::write_overlay_image(overlay_path.string(), width, height, rgb.get());

    // Persist timings to CSV if enabled, and print summary to console if enabled.
#if PERSIST_TIMINGS >= 1
    append_timings_csv(options, result, width, height);
#endif
    // Debug print summary of results and timings to console.
#if DEBUG_LEVEL >= 1
    print_summary_cuda(options, result);
#endif

    return result;
}
// ----------------------------------------------------------------------

// MAIN FUNCTION FOR STANDALONE BUILD
#ifdef CUDA_PIPELINE_BUILD_STANDALONE
int main(int argc, char **argv)
{
    // Initialize Pipeline Options
    PipelineOptions options;
    if (argc > 1)
    {
        // Read Input Path (Image or Folder)
        options.input_image_path = argv[1];
    }
    if (argc > 2)
    {
        // Read Output Path
        options.output_dir = argv[2];
    }

    try
    {
        // Load Project Path
        const fs::path input_path = project_path(options.input_image_path);

        // If Input path is a folder containing images
        if (fs::is_directory(input_path))
        {
            const std::vector<fs::path> images = collect_images(input_path);
            if (images.empty())
            {
                throw std::runtime_error("No supported images found in folder: " + input_path.string());
            }

            debug_print("Running CUDA pipeline on " + std::to_string(images.size()) +
                        " image(s) in: " + input_path.string());
            // Run each image through the same one-image pipeline. The first
            // image pays CUDA context/module setup; later images usually see
            // warmer CUDA runtime state.
            for (const fs::path &image_path : images)
            {
                PipelineOptions image_options = options;
                image_options.input_image_path = image_path.string();
                (void)run_cuda(image_options);
            }
        }
        // For single image
        else if (fs::is_regular_file(input_path))
        {
            (void)run_cuda(options);
        }
        else
        {
            throw std::runtime_error("Input path is not a file or folder: " + input_path.string());
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "CUDA pipeline failed: " << e.what() << '\n';
        return 1;
    }
}
#endif
