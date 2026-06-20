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

#include "parallel/pipeline.cuh"

#include <algorithm>
#include <cctype>
#include <cstdint>
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

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#if ENABLE_NVTX >= 1
#include <nvtx3/nvToolsExt.h>
#endif

#include "helpers.hpp"
#include "parallel/boundary_extraction.cuh"
#include "parallel/component_labeling.cuh"
#include "parallel/contour_to_signal.cuh"
#include "parallel/preprocessing.cuh"
#include "parallel/signal_analysis.cuh"
#include "parallel/visualization.hpp"
#include "stb_image.h"

namespace fs = std::filesystem;

namespace
{
    // TODO: Experiment with constants to avoid misclassification if time
    constexpr int GAUSSIAN_KERNEL_SIZE = 3;
    constexpr float GAUSSIAN_SIGMA = 1.0f;
    constexpr int MIN_REGION_AREA = 2000;
    constexpr int CONTOUR_SMOOTHING_WINDOW = 5;
    constexpr int PEAK_SMOOTHING_WINDOW = 5;
    constexpr float PEAK_MIN_PROMINENCE = 5.0f;
    constexpr float PEAK_MIN_SHARPNESS = 20.0f;
    constexpr int PEAK_MIN_DISTANCE = 5;
    constexpr float EDGE_TOL_FACTOR = 0.1f;
    constexpr double REFERENCE_IMAGE_WIDTH = 5100.0;

    int scaled_min_region_area(int width)
    {
        const double scale = static_cast<double>(width) / REFERENCE_IMAGE_WIDTH;
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

    fs::path project_path(const std::string &path)
    {
        fs::path p(path);
        return p.is_absolute() ? p : fs::current_path() / p;
    }

    template <typename Fn>
    void time_stage(const char *label, double &seconds, Fn &&fn)
    {
#if ENABLE_NVTX >= 1
        NvtxRange range(label);
#else
        (void)label;
#endif
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

    uint8_t *load_rgb_image(const fs::path &path, int &width, int &height)
    {
        int channels = 0;
        uint8_t *data = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
        if (data == nullptr)
        {
            throw std::runtime_error("Could not load image: " + path.string() + "\n" + stbi_failure_reason() + "\n");
        }
        return data;
    }

    std::string lowercase_extension(const fs::path &path)
    {
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch)
                       { return static_cast<char>(std::tolower(ch)); });
        return extension;
    }

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
        return images;
    }

#if DEBUG_LEVEL >= 1 || PERSIST_TIMINGS >= 1
    double milliseconds(double seconds)
    {
        return seconds * 1000.0;
    }
#endif

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
                  [](const auto &lhs, const auto &rhs)
                  {
                      return lhs.second > rhs.second;
                  });

        for (const auto &[label, seconds] : stage_timings)
        {
            print_timing(label.data(), seconds);
        }
#else
        std::cout << "  Sub timings disabled; compile with SUB_TIMINGS >= 1 to measure stages.\n";
#endif
    }
#endif

} // namespace

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
    
    Timer total_timer;
    total_timer.reset();

    const int total_pixels = width * height;
    const int min_region_area = scaled_min_region_area(height);

    thrust::device_vector<uint8_t> d_cleaned(static_cast<size_t>(total_pixels));
    thrust::device_vector<int> d_compact_labels(static_cast<size_t>(total_pixels));
    thrust::device_vector<uint8_t> d_piece_boundary(static_cast<size_t>(total_pixels));

    uint8_t *d_cleaned_ptr = thrust::raw_pointer_cast(d_cleaned.data());
    int *d_compact_labels_ptr = thrust::raw_pointer_cast(d_compact_labels.data());
    uint8_t *d_piece_boundary_ptr = thrust::raw_pointer_cast(d_piece_boundary.data());

    // STEP 2: Preprocess RGB image into a cleaned binary mask
    debug_print("[cuda pipeline] preprocessing and cleaning");
    time_stage("Preprocessing", result.timings.preprocessing, [&]()
               { preprocess_cuda_device(
                     rgb.get(),
                     d_cleaned_ptr,
                     width,
                     height,
                     GAUSSIAN_KERNEL_SIZE,
                     GAUSSIAN_SIGMA); });

    std::vector<Region> regions;
    // STEP 3: Label connected components and collect piece regions
    debug_print("[cuda pipeline] connected components");
    time_stage("Connected components", result.timings.connected_components, [&]()
               {
        (void)connected_components_buf_cuda_device(
            d_cleaned_ptr,
            d_compact_labels_ptr,
            width,
            height,
            min_region_area);
        build_regions_from_labels_buf_cuda(
            d_compact_labels_ptr,
            width,
            height,
            min_region_area,
            regions); });
    debug_print("[cuda pipeline] found " + std::to_string(regions.size()) + " regions");

    result.pieces.reserve(regions.size());
    const PuzzleLookupTable puzzle_lookup;

    for (const auto &region : regions)
    {
        ImageU8 piece_boundary;
        Coordinate<int> boundary_offset{0, 0};

        // STEP 4: Extract piece boundary from the compact label image
        debug_print("[cuda pipeline] extracting piece boundary: " + std::to_string(region.label));
        time_stage("Boundary extraction", result.timings.boundary_extraction, [&]()
                   { get_piece_boundary_mask_from_labels_cuda(
                         d_compact_labels_ptr,
                         region,
                         d_piece_boundary_ptr,
                         width,
                         height,
                         piece_boundary,
                         boundary_offset); });

        PuzzlePiece piece;
        piece.region = region;

        // STEP 5: Convert host boundary mask to global contour points
        debug_print("[cuda pipeline] extracting contour: " + std::to_string(region.label));
        time_stage("Contour extraction", result.timings.contour_extraction, [&]()
                   {
            find_contour_chain_approx_simple_cuda(
                piece_boundary.data,
                piece_boundary.width,
                piece_boundary.height,
                piece.contour);
            for (Coordinate<int>& point : piece.contour) {
                point.a += boundary_offset.a;
                point.b += boundary_offset.b;
            } });

        CoordinateVector<int> smoothed_contour;
        // STEP 6: Smooth contour before signal extraction
        debug_print("[cuda pipeline] smoothing contour: " + std::to_string(region.label));
        time_stage("Contour smoothing", result.timings.contour_smoothing, [&]()
                   {
            smoothed_contour = piece.contour;
            smooth_contour_cuda(smoothed_contour, CONTOUR_SMOOTHING_WINDOW); });

        Coordinate<float> circle_center{0.0f, 0.0f};
        // STEP 7: Approximate enclosing circle center
        debug_print("[cuda pipeline] approximating enclosing circle: " + std::to_string(region.label));
        time_stage("Enclosing circle", result.timings.enclosing_circle, [&]()
                   {
            float radius = 0.0f;
            enclosing_circle_approx_cuda(smoothed_contour, circle_center, radius); });

        Signal raw_radial_signal;
        // STEP 8: Convert contour to radial signal
        debug_print("[cuda pipeline] calculating radial signal: " + std::to_string(region.label));
        time_stage("Radial signal", result.timings.radial_signal, [&]()
                   { radial_signal_cuda(smoothed_contour, circle_center, raw_radial_signal); });

        Signal smoothed_radial_signal;
        // STEP 9: Smooth radial signal for corner detection
        debug_print("[cuda pipeline] smoothing signal: " + std::to_string(region.label));
        time_stage("Signal smoothing", result.timings.signal_smoothing, [&]()
                   { smoothed_radial_signal = smooth_signal_cuda(
                         raw_radial_signal,
                         PEAK_SMOOTHING_WINDOW); });

        // STEP 10: Detect the four corner peaks
        debug_print("[cuda pipeline] detecting peaks: " + std::to_string(region.label));
        time_stage("Peak detection", result.timings.peak_detection, [&]()
                   { piece.corner_indices = find_triangular_peaks_cuda(
                         smoothed_radial_signal,
                         PEAK_MIN_PROMINENCE,
                         PEAK_MIN_SHARPNESS,
                         PEAK_MIN_DISTANCE); });

        // STEP 11: Classify edge types and lookup piece class
        debug_print("[cuda pipeline] classifying edges: " + std::to_string(region.label));
        time_stage("Edge classification", result.timings.edge_classification, [&]()
                   {
            piece.edge_labels = classify_edges_cuda(
                raw_radial_signal,
                piece.corner_indices,
                EDGE_TOL_FACTOR);
            const std::string edge_label_string = edges_to_string_cuda(piece.edge_labels);
            piece.class_label = puzzle_lookup.getClassLabel(edge_label_string); });

        result.pieces.push_back(std::move(piece));
    }

    // STEP 12: Draw final contour and bounding-box overlay image
    std::vector<uint8_t> overlay_image;
    time_stage("Visualization", result.timings.visualization, [&]()
               {
        overlay_image = parallel_visualization::render_piece_overlays(
            rgb.get(),
            width,
            height,
            result.pieces); });

    result.timings.total_seconds = total_timer.get();

    const fs::path output_dir = project_path(options.output_dir);
    fs::create_directories(output_dir);
    const fs::path overlay_path = output_dir / (input_image_path.stem().string() + "_cuda_overlays.bmp");
    parallel_visualization::write_overlay_image(overlay_path.string(), width, height, overlay_image);

#if PERSIST_TIMINGS >= 1
    append_timings_csv(options, result, width, height);
#endif

#if DEBUG_LEVEL >= 1
    print_summary_cuda(options, result);
#endif

    return result;
}

#ifdef CUDA_PIPELINE_BUILD_STANDALONE
int main(int argc, char **argv)
{
    PipelineOptions options;
    if (argc > 1)
    {
        options.input_image_path = argv[1];
    }
    if (argc > 2)
    {
        options.output_dir = argv[2];
    }

    try
    {
        const fs::path input_path = project_path(options.input_image_path);

        if (fs::is_directory(input_path))
        {
            const std::vector<fs::path> images = collect_images(input_path);
            if (images.empty())
            {
                throw std::runtime_error("No supported images found in folder: " + input_path.string());
            }

            debug_print("Running CUDA pipeline on " + std::to_string(images.size()) +
                        " image(s) in: " + input_path.string());
            for (const fs::path &image_path : images)
            {
                PipelineOptions image_options = options;
                image_options.input_image_path = image_path.string();
                (void)run_cuda(image_options);
            }
        }
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
