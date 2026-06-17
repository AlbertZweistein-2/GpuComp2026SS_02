#pragma once

// Aggregate header for CUDA-backed pipeline stages. Prefer including the
// stage-specific headers directly when a translation unit only needs one stage.

#include "parallel/boundary_extraction.cuh"
#include "parallel/cleaning.cuh"
#include "parallel/component_labeling.cuh"
#include "parallel/contour_to_signal.hpp"
#include "parallel/pipeline.hpp"
#include "parallel/preprocessing.hpp"
#include "parallel/signal_analysis.cuh"
#include "parallel/visualization.hpp"
