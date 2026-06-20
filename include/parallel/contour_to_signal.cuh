#pragma once

#include <cstdint>
#include <vector>

#include <thrust/device_vector.h>

#include "types.hpp"

struct ContourCudaScratch
{
    // Active contour on the device. After smoothing, this holds the smoothed
    // contour so enclosing-circle and radial-signal stages can reuse it.
    thrust::device_vector<Coordinate<int>> points;
    // Temporary output buffer for contour smoothing.
    thrust::device_vector<Coordinate<int>> smoothed;
    // Device radial signal produced from points. Signal smoothing reuses this
    // instead of uploading the host signal again.
    thrust::device_vector<float> signal;
};

// Moore neighbor tracing is inherently sequential, but it gives the connected
// contour order needed by the current visualization and radial signal logic.
void trace_contour_cuda(
    const uint8_t *boundary,
    int width,
    int height,
    CoordinateVector<int> &result);

void trace_contour_cuda(
    const std::vector<uint8_t> &boundary,
    int width,
    int height,
    CoordinateVector<int> &result);

void simplify_chain_approx_cuda(CoordinateVector<int> &contour);

void find_contour_chain_approx_simple_cuda(
    const uint8_t *boundary,
    int width,
    int height,
    CoordinateVector<int> &result);

void find_contour_chain_approx_simple_cuda(
    const std::vector<uint8_t> &boundary,
    int width,
    int height,
    CoordinateVector<int> &result);

// parallelized via custom CUDA kernel with shared memory
void smooth_contour_cuda(CoordinateVector<int> &points, int window);

// Reuses device buffers and leaves the smoothed contour in scratch.points.
void smooth_contour_cuda(
    CoordinateVector<int> &points,
    int window,
    ContourCudaScratch &scratch);

// parallelized via thrust minmax + transform_reduce
void enclosing_circle_approx_cuda(
    const CoordinateVector<int> &points,
    Coordinate<float> &center,
    float &radius);

// Reuses the contour already stored in scratch.points.
void enclosing_circle_approx_cuda(
    const ContourCudaScratch &scratch,
    Coordinate<float> &center,
    float &radius);

// parallelized via custom CUDA kernel, one thread per contour point
void radial_signal_cuda(
    const CoordinateVector<int> &points,
    Coordinate<float> center,
    Signal &signal);

// Reuses the contour already stored in scratch.points and scratch.signal.
void radial_signal_cuda(
    ContourCudaScratch &scratch,
    Coordinate<float> center,
    Signal &signal);
