#pragma once

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

#include "types.hpp"

PipelineResult run(const PipelineOptions& options = PipelineOptions{});

#if DEBUG_LEVEL >= 1
void print_summary(const PipelineOptions& options, const PipelineResult& results);
#endif
