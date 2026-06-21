#pragma once

#include "types.hpp"

// Runs the CUDA-backed puzzle-piece pipeline for one input image and returns
// piece metadata plus optional timing information, depending on compile flags.
PipelineResult run_cuda(const PipelineOptions& options = PipelineOptions{});
