#pragma once

#include <array>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

#include "types.hpp"

struct SignalCudaScratch
{
    // Device copy of the smoothed batched signal. Peak detection consumes this
    // directly, while the host copy is kept for the CPU-only peak scoring pass.
    thrust::device_vector<float> smoothed;
    // Flat local-maxima mask matching the radial-signal offsets/lengths layout:
    // index offsets[piece] + local_i belongs to that piece's local sample.
    thrust::device_vector<int> peak_mask;
};

// Smooths all radial signal slices in one 2D launch using contour offsets/lengths.
// The returned host copy is intentionally kept because CPU peak scoring still
// needs random access to the smoothed signal.
void smooth_signals_batched_cuda(
    const thrust::device_vector<float> &d_signal,
    const thrust::device_vector<int> &offsets,
    const thrust::device_vector<int> &lengths,
    int max_length,
    int k,
    Signal &smoothed,
    SignalCudaScratch &scratch);

// Partially parallelized: local maxima are found on GPU, while prominence,
// sharpness, and final four-corner selection stay on CPU.
void find_triangular_peaks_batched_cuda(
    const Signal &smooth,
    const thrust::device_vector<float> &d_smooth,
    const std::vector<int> &offsets,
    const std::vector<int> &lengths,
    const thrust::device_vector<int> &d_offsets,
    const thrust::device_vector<int> &d_lengths,
    int max_length,
    SignalCudaScratch &scratch,
    float min_prominence,
    float min_sharpness,
    int min_distance,
    std::vector<Corners> &corners);

// Batched CPU classification over flat radial signal slices. A GPU version was
// tested but its launch and tiny copy overhead outweighed the removed raw-signal
// copy for the current piece counts.
void classify_edges_batched_cuda(
    const Signal &raw,
    const std::vector<int> &offsets,
    const std::vector<int> &lengths,
    const std::vector<Corners> &corners,
    float tol_factor,
    std::vector<EdgeLabels> &labels);

// Inherently sequential, identical to serial implementation.
std::string edges_to_string_cuda(const EdgeLabels &labels);

// Maps the four-edge string to a rotation-invariant puzzle-piece class label.
class PuzzleLookupTable
{
public:
    PuzzleLookupTable();

    std::string getClassLabel(const std::string &edges) const;

private:
    // 3^4 possible strings because each of four edges has L/V/C state.
    std::array<std::string, 81> table;

    int charToDigit(char c) const;
    int getBase3Index(const std::string &s) const;
    std::string rotate(const std::string &s) const;
    std::string getCategory(const std::string &s) const;
};
