#include "serial/pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "serial/boundary_extraction.hpp"
#include "serial/cleaning.hpp"
#include "serial/contour_to_signal.hpp"
#include "serial/preprocessing.hpp"
#include "serial/signal_analysis.hpp"
#include "stb_image.h"
#include "stb_image_write.h"
#include "helpers.hpp"

namespace fs = std::filesystem;

namespace {

fs::path project_path(const std::string& path)
{
    fs::path p(path);
    return p.is_absolute() ? p : fs::current_path() / p;
}

ImageU8 make_kernel(int width, int height)
{
    ImageU8 kernel;
    kernel.width = width;
    kernel.height = height;
    kernel.data.assign(static_cast<size_t>(width) * height, 1);
    return kernel;
}

ImageU8 load_u8_image(const fs::path& path)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    uint8_t* data = stbi_load(path.string().c_str(), &width, &height, &channels, 1);
    if (data == nullptr) {
        throw std::runtime_error("Could not load image: " + path.string());
    }

    ImageU8 image;
    image.width = width;
    image.height = height;
    image.data.assign(data, data + static_cast<size_t>(width) * height);
    stbi_image_free(data);
    return image;
}

struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> data;
};

RgbImage load_rgb_image(const fs::path& path)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    uint8_t* rgb = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
    if (rgb == nullptr) {
        throw std::runtime_error("Could not load image: " + path.string());
    }

    RgbImage image;
    image.width = width;
    image.height = height;
    image.data.assign(rgb, rgb + static_cast<size_t>(width) * height * 3);
    stbi_image_free(rgb);
    return image;
}

void write_u8_image(const fs::path& path, const ImageU8& image)
{
    if (image.data.empty()) {
        return;
    }

    fs::create_directories(path.parent_path());
    const int stride = image.width;
    if (stbi_write_png(path.string().c_str(), image.width, image.height, 1, image.data.data(), stride) == 0) {
        throw std::runtime_error("Could not write image: " + path.string());
    }
}

std::vector<Coordinate<int>> load_coordinates_csv(const fs::path& path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open coordinate CSV: " + path.string());
    }

    std::vector<Coordinate<int>> coordinates;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string a;
        std::string b;
        if (!std::getline(ss, a, ',') || !std::getline(ss, b, ',')) {
            throw std::runtime_error("Malformed coordinate row in: " + path.string());
        }
        coordinates.push_back({std::stoi(a), std::stoi(b)});
    }

    return coordinates;
}

Signal load_signal_csv(const fs::path& path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open signal CSV: " + path.string());
    }

    Signal signal;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            signal.push_back(std::stof(line));
        }
    }
    return signal;
}

std::vector<int> load_indices_csv(const fs::path& path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open index CSV: " + path.string());
    }

    std::vector<int> indices;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            indices.push_back(std::stoi(line));
        }
    }
    return indices;
}

std::pair<Coordinate<float>, float> load_circle_txt(const fs::path& path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open circle data: " + path.string());
    }

    std::string center_line;
    std::string radius_line;
    if (!std::getline(file, center_line) || !std::getline(file, radius_line)) {
        throw std::runtime_error("Malformed circle data: " + path.string());
    }

    std::stringstream center_stream(center_line);
    std::string a;
    std::string b;
    if (!std::getline(center_stream, a, ',') || !std::getline(center_stream, b, ',')) {
        throw std::runtime_error("Malformed circle center: " + path.string());
    }

    return {{std::stof(a), std::stof(b)}, std::stof(radius_line)};
}

void write_coordinates_csv(const fs::path& path, const std::vector<Coordinate<int>>& coordinates)
{
    fs::create_directories(path.parent_path());
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not write coordinate CSV: " + path.string());
    }

    for (const Coordinate<int>& coordinate : coordinates) {
        file << coordinate.a << ',' << coordinate.b << '\n';
    }
}

void write_signal_csv(const fs::path& path, const Signal& signal)
{
    fs::create_directories(path.parent_path());
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not write signal CSV: " + path.string());
    }

    for (float value : signal) {
        file << value << '\n';
    }
}

void write_indices_csv(const fs::path& path, const std::vector<int>& indices)
{
    fs::create_directories(path.parent_path());
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not write index CSV: " + path.string());
    }

    for (int index : indices) {
        file << index << '\n';
    }
}

void write_circle_txt(const fs::path& path, Coordinate<float> center, float radius)
{
    fs::create_directories(path.parent_path());
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not write circle data: " + path.string());
    }
    file << center.a << ',' << center.b << '\n' << radius << '\n';
}

std::vector<Coordinate<int>> smooth_contour(const std::vector<Coordinate<int>>& points, int window)
{
    if (points.empty() || window <= 1) {
        return points;
    }

    const int n = static_cast<int>(points.size());
    const int pad = window / 2;
    std::vector<Coordinate<int>> smoothed;
    smoothed.reserve(points.size());

    for (int i = 0; i < n; ++i) {
        float sum_a = 0.0f;
        float sum_b = 0.0f;
        for (int offset = -pad; offset <= pad; ++offset) {
            int idx = (i + offset + n) % n;
            sum_a += static_cast<float>(points[static_cast<size_t>(idx)].a);
            sum_b += static_cast<float>(points[static_cast<size_t>(idx)].b);
        }

        const float divisor = static_cast<float>(2 * pad + 1);
        smoothed.push_back({
            static_cast<int>(std::lround(sum_a / divisor)),
            static_cast<int>(std::lround(sum_b / divisor))
        });
    }

    return smoothed;
}

std::vector<float> circular_valid_moving_average(const std::vector<float>& signal, int window)
{
    if (signal.empty() || window <= 1) {
        return signal;
    }

    const int n = static_cast<int>(signal.size());
    const int pad = window / 2;
    std::vector<float> padded;
    padded.reserve(signal.size() + static_cast<size_t>(2 * pad));

    for (int i = n - pad; i < n; ++i) {
        padded.push_back(signal[static_cast<size_t>((i + n) % n)]);
    }
    padded.insert(padded.end(), signal.begin(), signal.end());
    for (int i = 0; i < pad; ++i) {
        padded.push_back(signal[static_cast<size_t>(i % n)]);
    }

    const int output_size = static_cast<int>(padded.size()) - window + 1;
    std::vector<float> smoothed(static_cast<size_t>(output_size), 0.0f);

    for (int i = 0; i < output_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < window; ++j) {
            sum += padded[static_cast<size_t>(i + j)];
        }
        smoothed[static_cast<size_t>(i)] = sum / static_cast<float>(window);
    }

    return smoothed;
}

float local_prominence(const std::vector<float>& signal, int index)
{
    const int n = static_cast<int>(signal.size());
    const float peak = signal[static_cast<size_t>(index)];

    float left_min = peak;
    for (int i = index - 1; i >= 0 && signal[static_cast<size_t>(i)] < peak; --i) {
        left_min = std::min(left_min, signal[static_cast<size_t>(i)]);
    }

    float right_min = peak;
    for (int i = index + 1; i < n && signal[static_cast<size_t>(i)] < peak; ++i) {
        right_min = std::min(right_min, signal[static_cast<size_t>(i)]);
    }

    return peak - std::max(left_min, right_min);
}

std::vector<int> find_triangular_peaks(
    const std::vector<float>& signal,
    int min_distance,
    float min_prominence,
    int smoothing_window,
    float min_sharpness)
{
    if (signal.size() < 3) {
        return {};
    }

    const std::vector<float> smoothed = circular_valid_moving_average(signal, smoothing_window);
    const int n = static_cast<int>(smoothed.size());

    std::vector<int> candidates;
    for (int i = 1; i < n - 1; ++i) {
        int left = i - 1;
        int right = i + 1;
        if (smoothed[static_cast<size_t>(i)] <= smoothed[static_cast<size_t>(left)] ||
            smoothed[static_cast<size_t>(i)] <= smoothed[static_cast<size_t>(right)]) {
            continue;
        }

        if (local_prominence(smoothed, i) < min_prominence) {
            continue;
        }

        int sharp_left = std::max(0, i - 20);
        int sharp_right = std::min(n - 1, i + 20);
        float sharpness = smoothed[static_cast<size_t>(i)] -
            0.5f * (smoothed[static_cast<size_t>(sharp_left)] + smoothed[static_cast<size_t>(sharp_right)]);

        if (sharpness >= min_sharpness) {
            candidates.push_back(i);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [&](int lhs, int rhs) {
        return smoothed[static_cast<size_t>(lhs)] > smoothed[static_cast<size_t>(rhs)];
    });

    std::vector<int> selected;
    for (int candidate : candidates) {
        bool too_close = false;
        for (int existing : selected) {
            int distance = std::abs(candidate - existing);
            distance = std::min(distance, n - distance);
            if (distance < min_distance) {
                too_close = true;
                break;
            }
        }
        if (!too_close) {
            selected.push_back(candidate);
        }
    }

    std::sort(selected.begin(), selected.end());
    selected.erase(std::remove(selected.begin(), selected.end(), 1), selected.end());
    return selected;
}

fs::path debug_image(const fs::path& test_data_dir, const char* filename)
{
    return test_data_dir / "debug_images" / filename;
}

fs::path contour_data(const fs::path& test_data_dir, const char* filename)
{
    return test_data_dir / "contour_to_signal" / filename;
}

template <typename Fn>
void time_stage(pipeline::StageTiming& timing, const char* source, Fn&& fn)
{
    Timer timer;
    timer.reset();
    fn();
    timing.seconds = timer.get();
    timing.source = source;
}

} // namespace

namespace pipeline {

PipelineResult run(const PipelineOptions& options)
{
    Timer total_timer;
    total_timer.reset();

    const fs::path test_data_dir = project_path(options.test_data_dir);
    const fs::path output_dir = project_path(options.output_dir);

    PipelineResult result;

#ifdef PIPELINE_STEP_PREPROCESSING
    std::cout << "[pipeline] preprocessing: " << options.input_image_path << '\n';
    const RgbImage input_image = load_rgb_image(project_path(options.input_image_path));
    time_stage(result.timings.preprocessing, "serial", [&]() {
        result.binary = preprocess(
            input_image.data.data(),
            input_image.width,
            input_image.height,
            options.gaussian_kernel_size,
            options.gaussian_sigma);
    });
#else
    std::cout << "[pipeline] preprocessing disabled, loading test binary\n";
    time_stage(result.timings.preprocessing, "test_data", [&]() {
        result.binary = load_u8_image(debug_image(test_data_dir, "debug_03_binary.png"));
    });
#endif

#ifdef PIPELINE_STEP_CLEANING
    std::cout << "[pipeline] cleaning\n";
    const ImageU8 kernel = make_kernel(options.morphology_kernel_width, options.morphology_kernel_height);
    time_stage(result.timings.cleaning, "serial", [&]() {
        morphological_open(result.binary, kernel, result.cleaned, options.morphology_iterations);
    });
#else
    std::cout << "[pipeline] cleaning disabled, loading test cleaned image\n";
    time_stage(result.timings.cleaning, "test_data", [&]() {
        result.cleaned = load_u8_image(debug_image(test_data_dir, "debug_06_cleaned.png"));
    });
#endif

#ifdef PIPELINE_STEP_BOUNDARY_EXTRACTION
    std::cout << "[pipeline] boundary extraction\n";
    time_stage(result.timings.boundary_extraction, "serial", [&]() {
        get_external_boundary_mask(result.cleaned, result.boundary);
    });
#else
    std::cout << "[pipeline] boundary extraction disabled, loading test boundary\n";
    time_stage(result.timings.boundary_extraction, "test_data", [&]() {
        result.boundary = load_u8_image(debug_image(test_data_dir, "debug_07_boundary.png"));
    });
#endif

#ifdef PIPELINE_STEP_CONTOUR_EXTRACTION
    std::cout << "[pipeline] contour extraction\n";
    time_stage(result.timings.contour_extraction, "serial", [&]() {
        result.contour = find_contour_chain_approx_simple(
            result.boundary.data,
            result.boundary.width,
            result.boundary.height);
    });
#else
    std::cout << "[pipeline] contour extraction disabled, loading test contour\n";
    time_stage(result.timings.contour_extraction, "test_data", [&]() {
        result.contour = load_coordinates_csv(contour_data(test_data_dir, "find_contour_chain_approx_simple.csv"));
    });
#endif

#ifdef PIPELINE_STEP_CONTOUR_SMOOTHING
    std::cout << "[pipeline] contour smoothing\n";
    time_stage(result.timings.contour_smoothing, "serial", [&]() {
        result.smoothed_contour = smooth_contour(result.contour, options.contour_smoothing_window);
    });
#else
    std::cout << "[pipeline] contour smoothing disabled, loading test smoothed contour\n";
    time_stage(result.timings.contour_smoothing, "test_data", [&]() {
        result.smoothed_contour = load_coordinates_csv(contour_data(test_data_dir, "smooth_contour.csv"));
    });
#endif

#ifdef PIPELINE_STEP_ENCLOSING_CIRCLE
    std::cout << "[pipeline] enclosing circle\n";
    time_stage(result.timings.enclosing_circle, "serial", [&]() {
        auto [center, radius] = enclosing_circle_approx(result.smoothed_contour);
        result.circle_center = center;
        result.circle_radius = radius;
    });
#else
    std::cout << "[pipeline] enclosing circle disabled, loading test circle\n";
    time_stage(result.timings.enclosing_circle, "test_data", [&]() {
        auto [center, radius] = load_circle_txt(contour_data(test_data_dir, "enclosing_circle_approx.txt"));
        result.circle_center = center;
        result.circle_radius = radius;
    });
#endif

#ifdef PIPELINE_STEP_RADIAL_SIGNAL
    std::cout << "[pipeline] radial signal\n";
    time_stage(result.timings.radial_signal, "serial", [&]() {
        result.radial_signal = radial_signal(
            result.smoothed_contour,
            result.circle_center);
    });
#else
    std::cout << "[pipeline] radial signal disabled, loading test signal\n";
    time_stage(result.timings.radial_signal, "test_data", [&]() {
        result.radial_signal = load_signal_csv(contour_data(test_data_dir, "radial_signal.csv"));
    });
#endif

#ifdef PIPELINE_STEP_PEAK_DETECTION
    std::cout << "[pipeline] peak detection\n";
    time_stage(result.timings.peak_detection, "serial", [&]() {
        result.triangular_peak_indices = find_triangular_peaks(
            result.radial_signal,
            options.peak_min_distance,
            options.peak_min_prominence,
            options.peak_smoothing_window,
            options.peak_min_sharpness);
    });
#else
    std::cout << "[pipeline] peak detection disabled, loading test peak indices\n";
    time_stage(result.timings.peak_detection, "test_data", [&]() {
        result.triangular_peak_indices = load_indices_csv(contour_data(test_data_dir, "find_triangular_peaks.csv"));
    });
#endif

    if (options.write_debug_outputs) {
        write_u8_image(output_dir / "binary.png", result.binary);
        write_u8_image(output_dir / "cleaned.png", result.cleaned);
        write_u8_image(output_dir / "boundary.png", result.boundary);
        write_coordinates_csv(output_dir / "contour.csv", result.contour);
        write_coordinates_csv(output_dir / "smooth_contour.csv", result.smoothed_contour);
        write_circle_txt(output_dir / "enclosing_circle.txt", result.circle_center, result.circle_radius);
        write_signal_csv(output_dir / "radial_signal.csv", result.radial_signal);
        write_indices_csv(output_dir / "triangular_peaks.csv", result.triangular_peak_indices);
    }

    result.timings.total_seconds = total_timer.get();
    return result;
}

void print_summary(const PipelineResult& result)
{
    const auto foreground_count = [](const ImageU8& image) {
        return std::count(image.data.begin(), image.data.end(), static_cast<uint8_t>(255));
    };
    const auto print_timing = [](const char* label, const StageTiming& timing) {
        std::cout << "  " << label << " [" << timing.source << "]"
                  << " : " << timing.seconds * 1000.0 << " ms\n";
    };

    std::cout << "\n-- Pipeline summary ------------------------\n";
    std::cout << "Binary image       : " << result.binary.width << " x " << result.binary.height
              << ", foreground " << foreground_count(result.binary) << '\n';
    std::cout << "Cleaned image      : " << result.cleaned.width << " x " << result.cleaned.height
              << ", foreground " << foreground_count(result.cleaned) << '\n';
    std::cout << "Boundary image     : " << result.boundary.width << " x " << result.boundary.height
              << ", foreground " << foreground_count(result.boundary) << '\n';
    std::cout << "Contour points     : " << result.contour.size() << '\n';
    std::cout << "Smoothed contour   : " << result.smoothed_contour.size() << '\n';
    std::cout << "Circle center      : (" << result.circle_center.a << ", " << result.circle_center.b << ")\n";
    std::cout << "Circle radius      : " << result.circle_radius << '\n';
    std::cout << "Signal samples     : " << result.radial_signal.size() << '\n';
    std::cout << "Triangular peaks   : " << result.triangular_peak_indices.size() << '\n';
    std::cout << "\n-- Single-run timings ----------------------\n";
    print_timing("Preprocessing      ", result.timings.preprocessing);
    print_timing("Cleaning           ", result.timings.cleaning);
    print_timing("Boundary extraction", result.timings.boundary_extraction);
    print_timing("Contour extraction ", result.timings.contour_extraction);
    print_timing("Contour smoothing  ", result.timings.contour_smoothing);
    print_timing("Enclosing circle   ", result.timings.enclosing_circle);
    print_timing("Radial signal      ", result.timings.radial_signal);
    print_timing("Peak detection     ", result.timings.peak_detection);
    std::cout << "  Total wall time    : " << result.timings.total_seconds * 1000.0 << " ms\n";
}

} // namespace pipeline

int pipeline_main(int argc, char** argv)
{
    pipeline::PipelineOptions options;
    if (argc > 1) {
        options.input_image_path = argv[1];
    }
    if (argc > 2) {
        options.output_dir = argv[2];
    }

    try {
        const pipeline::PipelineResult result = pipeline::run(options);
        pipeline::print_summary(result);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Pipeline failed: " << e.what() << '\n';
        return 1;
    }
}

#ifdef PIPELINE_BUILD_STANDALONE
int main(int argc, char** argv)
{
    return pipeline_main(argc, argv);
}
#endif
