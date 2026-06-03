#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "types.hpp"

// If no explicit step selection is provided, build the complete pipeline.
//
// Stage execution flags:
//   PIPELINE_STEP_PREPROCESSING
//   PIPELINE_STEP_CLEANING
//   PIPELINE_STEP_BOUNDARY_EXTRACTION
//   PIPELINE_STEP_CONTOUR_EXTRACTION
//   PIPELINE_STEP_CONTOUR_SMOOTHING
//   PIPELINE_STEP_ENCLOSING_CIRCLE
//   PIPELINE_STEP_RADIAL_SIGNAL
//   PIPELINE_STEP_PEAK_DETECTION
//
// Backend selection flags for executed stages:
//   PIPELINE_USE_CUDA_PREPROCESSING
//   PIPELINE_USE_CUDA_CLEANING
//   PIPELINE_USE_CUDA_BOUNDARY_EXTRACTION
//   PIPELINE_USE_CUDA_CONTOUR_EXTRACTION
//   PIPELINE_USE_CUDA_CONTOUR_SMOOTHING
//   PIPELINE_USE_CUDA_ENCLOSING_CIRCLE
//   PIPELINE_USE_CUDA_RADIAL_SIGNAL
//   PIPELINE_USE_CUDA_PEAK_DETECTION
//
// A CUDA stage must provide the matching *_cuda function declared in
// include/cuda/ops_cuda.hpp and be linked into the final target.
#if !defined(PIPELINE_STEP_PREPROCESSING) && \
    !defined(PIPELINE_STEP_CLEANING) && \
    !defined(PIPELINE_STEP_BOUNDARY_EXTRACTION) && \
    !defined(PIPELINE_STEP_CONTOUR_EXTRACTION) && \
    !defined(PIPELINE_STEP_CONTOUR_SMOOTHING) && \
    !defined(PIPELINE_STEP_ENCLOSING_CIRCLE) && \
    !defined(PIPELINE_STEP_RADIAL_SIGNAL) && \
    !defined(PIPELINE_STEP_PEAK_DETECTION)
#define PIPELINE_STEP_PREPROCESSING
#define PIPELINE_STEP_CLEANING
#define PIPELINE_STEP_BOUNDARY_EXTRACTION
#define PIPELINE_STEP_CONTOUR_EXTRACTION
#define PIPELINE_STEP_CONTOUR_SMOOTHING
#define PIPELINE_STEP_ENCLOSING_CIRCLE
#define PIPELINE_STEP_RADIAL_SIGNAL
#define PIPELINE_STEP_PEAK_DETECTION
#endif

namespace pipeline {

struct StageTiming {
    double seconds = 0.0;
    const char* backend = "disabled";
};

struct PipelineTimings {
    StageTiming preprocessing;
    StageTiming cleaning;
    StageTiming boundary_extraction;
    StageTiming contour_extraction;
    StageTiming contour_smoothing;
    StageTiming enclosing_circle;
    StageTiming radial_signal;
    StageTiming peak_detection;
    double total_seconds = 0.0;
};

struct PipelineOptions {
    std::string input_image_path = "data/single.JPG";
    std::string test_data_dir = "data/test_data";
    std::string output_dir = "data/pipeline_output";

    bool write_debug_outputs = true;

    int gaussian_kernel_size = 5;
    float gaussian_sigma = 1.0f;

    int morphology_kernel_width = 4;
    int morphology_kernel_height = 4;
    int morphology_iterations = 2;

    int contour_smoothing_window = 5;

    int peak_min_distance = 5;
    float peak_min_prominence = 5.0f;
    int peak_smoothing_window = 2;
    float peak_min_sharpness = 20.0f;
};

struct PipelineResult {
    ImageU8 binary;
    ImageU8 cleaned;
    ImageU8 boundary;

    std::vector<Coordinate> contour;
    std::vector<Coordinate> smoothed_contour;

    CoordinateFloat circle_center{0.0f, 0.0f};
    float circle_radius = 0.0f;

    std::vector<float> radial_signal;
    std::vector<int> triangular_peak_indices;

    PipelineTimings timings;
};

PipelineResult run(const PipelineOptions& options = PipelineOptions{});

void print_summary(const PipelineResult& result);

} // namespace pipeline

int pipeline_main(int argc, char** argv);
