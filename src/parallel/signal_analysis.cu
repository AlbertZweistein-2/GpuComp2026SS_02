#include "parallel/signal_analysis.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "helpers.hpp"

namespace
{
    __global__ void batched_local_maxima_mask_kernel(
        const float *src,
        int *mask,
        const int *offsets,
        const int *lengths,
        int max_length)
    {
        // One 2D launch covers all smoothed radial signals. Each piece keeps
        // circular indexing local to its own offset/length slice.
        const int piece_idx = blockIdx.y;
        const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
        if (local_i >= max_length)
        {
            return;
        }

        const int n = lengths[piece_idx];
        if (local_i >= n)
        {
            return;
        }

        const int offset = offsets[piece_idx];
        const int global_i = offset + local_i;
        const int local_l = (local_i == 0) ? n - 1 : local_i - 1;
        const int local_r = (local_i == n - 1) ? 0 : local_i + 1;
        const float val = src[global_i];
        mask[global_i] = (val >= src[offset + local_l] && val > src[offset + local_r]) ? 1 : 0;
    }

    __global__ void batched_signal_smoothing_kernel(
        const float *src,
        float *dst,
        const int *offsets,
        const int *lengths,
        int max_length,
        int k,
        int pad)
    {
        // Same flat layout as contour batching: y selects the piece, x selects
        // the local signal sample inside that piece.
        const int piece_idx = blockIdx.y;
        const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
        if (local_i >= max_length)
        {
            return;
        }

        const int n = lengths[piece_idx];
        if (local_i >= n)
        {
            return;
        }

        const int offset = offsets[piece_idx];
        const int global_i = offset + local_i;
        if (k <= 1 || k >= n)
        {
            dst[global_i] = src[global_i];
            return;
        }

        float sum = 0.0f;
        for (int j = -pad; j <= pad; ++j)
        {
            int local_j = (local_i + j) % n;
            if (local_j < 0)
            {
                local_j += n;
            }
            sum += src[offset + local_j];
        }
        dst[global_i] = sum / static_cast<float>(k);
    }

    struct Candidate
    {
        int idx;
        float val;
    };

    Corners select_triangular_peaks_from_candidates(
        const Signal &smooth,
        std::size_t signal_offset,
        int n,
        const std::vector<int> &candidates,
        float min_prominence,
        float min_sharpness,
        int min_distance)
    {
        // The GPU only finds local maxima. The original prominence, sharpness,
        // spacing, and "pick four far-apart peaks" logic stays here unchanged.
        Corners final_corners = {0, 0, 0, 0};
        if (n < 3)
        {
            return final_corners;
        }

        auto value_at = [&](int local_i)
        {
            return smooth[signal_offset + static_cast<std::size_t>(local_i)];
        };
        auto wrap = [n](int i)
        { return ((i % n) + n) % n; };

        std::vector<Candidate> kept;
        for (int idx : candidates)
        {
            const float val = value_at(idx);

            float lmin = val;
            float rmin = val;
            for (int step = 1; step < n; ++step)
            {
                const int j = wrap(idx - step);
                if (value_at(j) > val)
                {
                    break;
                }
                lmin = std::min(lmin, value_at(j));
            }
            for (int step = 1; step < n; ++step)
            {
                const int j = wrap(idx + step);
                if (value_at(j) > val)
                {
                    break;
                }
                rmin = std::min(rmin, value_at(j));
            }

            const float prominence = val - std::max(lmin, rmin);
            if (prominence < min_prominence)
            {
                continue;
            }

            const int li = wrap(idx - 20);
            const int ri = wrap(idx + 20);
            const float sharpness = val - 0.5f * (value_at(li) + value_at(ri));
            if (sharpness < min_sharpness)
            {
                continue;
            }

            if (!kept.empty())
            {
                const int linear_dist = std::abs(idx - kept.back().idx);
                const int circ_dist = std::min(linear_dist, n - linear_dist);

                if (circ_dist < min_distance)
                {
                    if (val > kept.back().val)
                    {
                        kept.back() = {idx, val};
                    }
                    continue;
                }
            }

            kept.push_back({idx, val});
        }

        if (kept.size() > 1)
        {
            const int linear_dist = std::abs(kept.front().idx - kept.back().idx);
            const int circ_dist = std::min(linear_dist, n - linear_dist);
            if (circ_dist < min_distance)
            {
                if (kept.back().val > kept.front().val)
                {
                    kept.front() = kept.back();
                }
                kept.pop_back();
            }
        }

        std::vector<Candidate> remaining = kept;
        std::sort(remaining.begin(), remaining.end(),
                  [](const Candidate &a, const Candidate &b)
                  {
                      return a.val > b.val;
                  });

        std::vector<int> corners;
        if (!remaining.empty())
        {
            corners.push_back(remaining.front().idx);
        }

        while (corners.size() < 4 && !remaining.empty())
        {
            int best_idx = -1;
            int best_dist = -1;

            for (const auto &peak : remaining)
            {
                int min_d = n;
                for (int corner : corners)
                {
                    const int d = std::abs(peak.idx - corner);
                    min_d = std::min(min_d, std::min(d, n - d));
                }
                if (min_d > best_dist)
                {
                    best_dist = min_d;
                    best_idx = peak.idx;
                }
            }

            if (best_idx != -1)
            {
                corners.push_back(best_idx);
                remaining.erase(
                    std::remove_if(
                        remaining.begin(),
                        remaining.end(),
                        [best_idx](const Candidate &peak)
                        {
                            return peak.idx == best_idx;
                        }),
                    remaining.end());
            }
        }

        std::sort(corners.begin(), corners.end());
        for (int i = 0; i < std::min(4, static_cast<int>(corners.size())); ++i)
        {
            final_corners[static_cast<std::size_t>(i)] = corners[static_cast<std::size_t>(i)];
        }

        return final_corners;
    }

} // namespace

void smooth_signals_batched_cuda(
    const thrust::device_vector<float> &d_signal,
    const thrust::device_vector<int> &offsets,
    const thrust::device_vector<int> &lengths,
    int max_length,
    int k,
    Signal &smoothed,
    SignalCudaScratch &scratch)
{
    const int total_count = static_cast<int>(d_signal.size());
    const int piece_count = static_cast<int>(offsets.size());
    smoothed.resize(static_cast<std::size_t>(total_count));
    if (total_count == 0 || piece_count == 0)
    {
        scratch.smoothed.clear();
        return;
    }

    scratch.smoothed.resize(static_cast<std::size_t>(total_count));

    // Smooth each piece inside its own circular signal slice. This replaces
    // one launch and one DtoH copy per piece with one batched launch/copy.
    const int block = 256;
    const dim3 grid(
        static_cast<unsigned int>((max_length + block - 1) / block),
        static_cast<unsigned int>(piece_count));
    batched_signal_smoothing_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(d_signal.data()),
        thrust::raw_pointer_cast(scratch.smoothed.data()),
        thrust::raw_pointer_cast(offsets.data()),
        thrust::raw_pointer_cast(lengths.data()),
        max_length,
        k,
        k / 2);
    CUDA_CHECK(cudaGetLastError());

    thrust::copy(scratch.smoothed.begin(), scratch.smoothed.end(), smoothed.begin());
}

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
    std::vector<Corners> &corners)
{
    const int piece_count = static_cast<int>(offsets.size());
    corners.assign(static_cast<std::size_t>(piece_count), Corners{0, 0, 0, 0});
    if (piece_count == 0 || smooth.empty())
    {
        scratch.peak_mask.clear();
        return;
    }

    scratch.peak_mask.resize(smooth.size());

    // Generate local-maxima masks for every piece in one launch. The expensive
    // prominence/sharpness walk remains on CPU, but it now consumes one flat
    // mask copy instead of one GPU compaction/copy per piece.
    const int block = 256;
    const dim3 grid(
        static_cast<unsigned int>((max_length + block - 1) / block),
        static_cast<unsigned int>(piece_count));
    batched_local_maxima_mask_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(d_smooth.data()),
        thrust::raw_pointer_cast(scratch.peak_mask.data()),
        thrust::raw_pointer_cast(d_offsets.data()),
        thrust::raw_pointer_cast(d_lengths.data()),
        max_length);
    CUDA_CHECK(cudaGetLastError());

    std::vector<int> host_mask(smooth.size());
    thrust::copy(scratch.peak_mask.begin(), scratch.peak_mask.end(), host_mask.begin());

    std::vector<int> candidates;
    for (int piece_idx = 0; piece_idx < piece_count; ++piece_idx)
    {
        const int offset = offsets[static_cast<std::size_t>(piece_idx)];
        const int n = lengths[static_cast<std::size_t>(piece_idx)];
        if (n < 3)
        {
            continue;
        }

        candidates.clear();
        candidates.reserve(static_cast<std::size_t>(n / 4));
        for (int local_i = 0; local_i < n; ++local_i)
        {
            if (host_mask[static_cast<std::size_t>(offset + local_i)] != 0)
            {
                candidates.push_back(local_i);
            }
        }

        corners[static_cast<std::size_t>(piece_idx)] =
            select_triangular_peaks_from_candidates(
                smooth,
                static_cast<std::size_t>(offset),
                n,
                candidates,
                min_prominence,
                min_sharpness,
                min_distance);
    }
}

void classify_edges_batched_cuda(
    const Signal &raw,
    const std::vector<int> &offsets,
    const std::vector<int> &lengths,
    const std::vector<Corners> &corners,
    float tol_factor,
    std::vector<EdgeLabels> &labels)
{
    // Classification is tiny and branch-heavy, so batching it on CPU avoids
    // per-piece pipeline overhead without paying an extra GPU launch/copy.
    const std::size_t piece_count = offsets.size();
    labels.assign(
        piece_count,
        EdgeLabels{EdgeType::Straight, EdgeType::Straight, EdgeType::Straight, EdgeType::Straight});

    for (std::size_t piece_idx = 0; piece_idx < piece_count; ++piece_idx)
    {
        const int offset = offsets[piece_idx];
        const int n = lengths[piece_idx];
        if (n < 4)
        {
            continue;
        }

        auto value_at = [&](int local_i)
        {
            return raw[static_cast<std::size_t>(offset + local_i)];
        };

        for (int edge_idx = 0; edge_idx < 4; ++edge_idx)
        {
            const Corners &piece_corners = corners[piece_idx];
            const int ca = piece_corners[static_cast<std::size_t>(edge_idx)];
            const int cb = piece_corners[static_cast<std::size_t>((edge_idx + 1) % 4)];

            const int edge_len = (cb - ca + n) % n;
            const int margin = edge_len / 5;

            const int start_idx = (ca + margin) % n;
            const int end_idx = (cb - margin + n) % n;
            const float v_start = value_at(start_idx);
            const float v_end = value_at(end_idx);

            const float shoulder_max = std::max(v_start, v_end);
            const float shoulder_min = std::min(v_start, v_end);

            float mx = -1e30f;
            float mn = 1e30f;
            for (int step = margin; step <= edge_len - margin; ++step)
            {
                const int local_i = (ca + step) % n;
                const float val = value_at(local_i);
                mx = std::max(mx, val);
                mn = std::min(mn, val);
            }

            const float local_tol = tol_factor * shoulder_max;
            if (mx > shoulder_max + local_tol)
            {
                labels[piece_idx][static_cast<std::size_t>(edge_idx)] = EdgeType::Knob;
            }
            else if (mn < shoulder_min - local_tol)
            {
                labels[piece_idx][static_cast<std::size_t>(edge_idx)] = EdgeType::Hole;
            }
            else
            {
                labels[piece_idx][static_cast<std::size_t>(edge_idx)] = EdgeType::Straight;
            }
        }
    }
}

static char edge_char_cuda(EdgeType e)
{
    switch (e)
    {
    case EdgeType::Straight:
        return 'L';
    case EdgeType::Knob:
        return 'V';
    case EdgeType::Hole:
        return 'C';
    }
    return '?';
}

std::string edges_to_string_cuda(const EdgeLabels &labels)
{
    std::string s;
    for (auto e : labels)
        s += edge_char_cuda(e);
    return s;
}

int PuzzleLookupTable::charToDigit(char c) const
{
    if (c == 'L')
        return 0;
    if (c == 'V')
        return 1;
    if (c == 'C')
        return 2;
    return 0;
}

int PuzzleLookupTable::getBase3Index(const std::string &s) const
{
    return charToDigit(s[0]) * 27 +
           charToDigit(s[1]) * 9 +
           charToDigit(s[2]) * 3 +
           charToDigit(s[3]);
}

std::string PuzzleLookupTable::rotate(const std::string &s) const
{
    return s.substr(1) + s[0];
}

std::string PuzzleLookupTable::getCategory(const std::string &s) const
{
    int l_count = std::count(s.begin(), s.end(), 'L');
    if (l_count == 0)
        return "I";
    if (l_count == 1)
        return "E";
    if (l_count == 2)
        return "C";
    if (l_count == 3)
        return "T";
    if (l_count == 4)
        return "S";
    return "U";
}

PuzzleLookupTable::PuzzleLookupTable()
{
    table.fill("");

    int count_I = 0, count_E = 0, count_C = 0, count_T = 0, count_S = 0;
    const char edges[] = {'L', 'V', 'C'};

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                for (int l = 0; l < 3; ++l)
                {
                    std::string s = {edges[i], edges[j], edges[k], edges[l]};
                    int idx = getBase3Index(s);

                    if (table[idx].empty())
                    {
                        std::string r1 = rotate(s);
                        std::string r2 = rotate(r1);
                        std::string r3 = rotate(r2);

                        std::string canonical = s;
                        if (r1 < canonical)
                            canonical = r1;
                        if (r2 < canonical)
                            canonical = r2;
                        if (r3 < canonical)
                            canonical = r3;

                        std::string cat = getCategory(canonical);
                        int class_id = 0;

                        if (cat == "I")
                            class_id = count_I++;
                        else if (cat == "E")
                            class_id = count_E++;
                        else if (cat == "C")
                            class_id = count_C++;
                        else if (cat == "T")
                            class_id = count_T++;
                        else if (cat == "S")
                            class_id = count_S++;

                        std::string final_label = cat + "_" + std::to_string(class_id) + ": " + canonical;

                        table[getBase3Index(s)] = final_label;
                        table[getBase3Index(r1)] = final_label;
                        table[getBase3Index(r2)] = final_label;
                        table[getBase3Index(r3)] = final_label;
                    }
                }
            }
        }
    }
}

std::string PuzzleLookupTable::getClassLabel(const std::string &edges) const
{
    if (edges.length() != 4)
        return "Invalid";
    return table[getBase3Index(edges)];
}
