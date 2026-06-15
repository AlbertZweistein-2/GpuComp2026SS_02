#pragma once

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

#include <string>
#include <vector>

#include "types.hpp"

struct PipelineTimings {
    double preprocessing = 0.0;
    double cleaning = 0.0;
    double connected_components = 0.0;
    double mask_extraction = 0.0;
    double boundary_extraction = 0.0;
    double contour_extraction = 0.0;
    double contour_smoothing = 0.0;
    double enclosing_circle = 0.0;
    double radial_signal = 0.0;
    double signal_smoothing = 0.0;
    double peak_detection = 0.0;
    double edge_classification = 0.0;
    double total_seconds = 0.0;
};

struct PipelineOptions {
    std::string input_image_path = "data/single.JPG";
    std::string output_dir = "data/pipeline_output";

    int gaussian_kernel_size = 5;
    float gaussian_sigma = 1.0f;

    int morphology_kernel_width = 4;
    int morphology_kernel_height = 4;
    int morphology_iterations = 2;

    int min_region_area = 2000;

    int contour_smoothing_window = 5;

    int peak_smoothing_window = 5;
    float peak_min_prominence = 5.0f;
    float peak_min_sharpness = 20.0f;
    int peak_min_distance = 5;

    float tol_factor = 0.1f;
};

struct PuzzlePiece {
    int label = 0;
    CoordinateVector<int> contour;
    CoordinateVector<int> smoothed_contour;

    Coordinate<float> circle_center{0.0f, 0.0f};
    float circle_radius = 0.0f;

    Signal radial_signal;
    Corners triangular_peak_indices{0, 0, 0, 0};
    EdgeLabels edge_labels{EdgeType::Straight, EdgeType::Straight, EdgeType::Straight, EdgeType::Straight};
};

struct PipelineResult {
    ImageU8 binary;
    ImageU8 cleaned;
    ImageU8 boundary;

    std::vector<PuzzlePiece> pieces;
    PipelineTimings timings;
};

PipelineResult run(const PipelineOptions& options = PipelineOptions{});

#if DEBUG_LEVEL >= 1
void print_summary(const PipelineOptions& options, const PipelineResult& results);
#endif
