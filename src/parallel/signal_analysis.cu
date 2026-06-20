#include "parallel/signal_analysis.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace
{

    struct MovingAverageOp
    {
        const float *src;
        int n;
        int k;
        int pad;

        __host__ __device__ float operator()(int i) const
        {
            float sum = 0.0f;
            for (int j = -pad; j <= pad; ++j)
            {
                int idx = (i + j) % n;
                if (idx < 0)
                {
                    idx += n;
                }
                sum += src[idx];
            }
            return sum / k;
        }
    };

    struct IsLocalMaxOp
    {
        const float *src;
        int n;

        __host__ __device__ int operator()(int i) const
        {
            const int l = (i == 0) ? n - 1 : i - 1;
            const int r = (i == n - 1) ? 0 : i + 1;
            return (src[i] >= src[l] && src[i] > src[r]) ? 1 : 0;
        }
    };

    struct MaskIsSet
    {
        const int *mask;

        __host__ __device__ bool operator()(int i) const
        {
            return mask[i] != 0;
        }
    };

    struct Candidate
    {
        int idx;
        float val;
    };

    void copy_signal_to_device(
        const Signal &signal,
        thrust::device_vector<float> &d_signal)
    {
        // Legacy callers still pass host signals. Reusing d_signal keeps
        // capacity around instead of allocating a new device vector each call.
        d_signal.resize(signal.size());
        if (!signal.empty())
        {
            thrust::copy(signal.begin(), signal.end(), d_signal.begin());
        }
    }

    void smooth_signal_device_cuda(
        const thrust::device_vector<float> &d_signal,
        int k,
        Signal &smoothed,
        SignalCudaScratch &scratch)
    {
        const int n = static_cast<int>(d_signal.size());
        smoothed.resize(static_cast<std::size_t>(n));
        if (n == 0)
        {
            scratch.smoothed.clear();
            return;
        }

        scratch.smoothed.resize(static_cast<std::size_t>(n));
        const int pad = k / 2;

        // One thread computes one circular moving-average sample. The result
        // stays in scratch.smoothed so peak detection can reuse it on device.
        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(n),
            scratch.smoothed.begin(),
            MovingAverageOp{thrust::raw_pointer_cast(d_signal.data()), n, k, pad});

        thrust::copy(scratch.smoothed.begin(), scratch.smoothed.end(), smoothed.begin());
    }

    std::vector<int> find_local_maxima_cuda(
        const thrust::device_vector<float> &d_smooth,
        SignalCudaScratch &scratch)
    {
        const int n = static_cast<int>(d_smooth.size());
        if (n < 3)
        {
            return {};
        }

        scratch.peak_mask.resize(static_cast<std::size_t>(n));
        scratch.peak_candidates.resize(static_cast<std::size_t>(n));

        // First build a 0/1 maxima mask, then compact the selected indices.
        // This is still a GPU prefilter; prominence/sharpness filtering remains
        // on CPU because it walks variable-length neighborhoods.
        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(n),
            scratch.peak_mask.begin(),
            IsLocalMaxOp{thrust::raw_pointer_cast(d_smooth.data()), n});

        auto end = thrust::copy_if(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(n),
            scratch.peak_candidates.begin(),
            MaskIsSet{thrust::raw_pointer_cast(scratch.peak_mask.data())});
        const std::size_t candidate_count = static_cast<std::size_t>(end - scratch.peak_candidates.begin());

        std::vector<int> candidates(candidate_count);
        if (candidate_count > 0)
        {
            thrust::copy(
                scratch.peak_candidates.begin(),
                scratch.peak_candidates.begin() + static_cast<std::ptrdiff_t>(candidate_count),
                candidates.begin());
        }
        return candidates;
    }

} // namespace

Signal smooth_signal_cuda(const Signal &signal, int k)
{
    Signal smoothed;
    SignalCudaScratch scratch;
    smooth_signal_cuda(signal, k, smoothed, scratch);
    return smoothed;
}

void smooth_signal_cuda(
    const Signal &signal,
    int k,
    Signal &smoothed,
    SignalCudaScratch &scratch)
{
    copy_signal_to_device(signal, scratch.input);
    smooth_signal_device_cuda(scratch.input, k, smoothed, scratch);
}

void smooth_signal_cuda(
    const thrust::device_vector<float> &d_signal,
    int k,
    Signal &smoothed,
    SignalCudaScratch &scratch)
{
    smooth_signal_device_cuda(d_signal, k, smoothed, scratch);
}

Corners find_triangular_peaks_cuda(
    const Signal &smooth,
    float min_prominence,
    float min_sharpness,
    int min_distance)
{
    SignalCudaScratch scratch;
    copy_signal_to_device(smooth, scratch.smoothed);
    return find_triangular_peaks_cuda(
        smooth,
        scratch,
        min_prominence,
        min_sharpness,
        min_distance);
}

Corners find_triangular_peaks_cuda(
    const Signal &smooth,
    SignalCudaScratch &scratch,
    float min_prominence,
    float min_sharpness,
    int min_distance)
{
    const int n = static_cast<int>(smooth.size());
    Corners final_corners = {0, 0, 0, 0};
    if (n < 3)
    {
        return final_corners;
    }

    if (scratch.smoothed.size() != smooth.size())
    {
        copy_signal_to_device(smooth, scratch.smoothed);
    }

    const std::vector<int> candidates = find_local_maxima_cuda(scratch.smoothed, scratch);
    auto wrap = [n](int i)
    { return ((i % n) + n) % n; };

    std::vector<Candidate> kept;
    for (int idx : candidates)
    {
        const float val = smooth[static_cast<std::size_t>(idx)];

        float lmin = val;
        float rmin = val;
        for (int step = 1; step < n; ++step)
        {
            const int j = wrap(idx - step);
            if (smooth[static_cast<std::size_t>(j)] > val)
            {
                break;
            }
            lmin = std::min(lmin, smooth[static_cast<std::size_t>(j)]);
        }
        for (int step = 1; step < n; ++step)
        {
            const int j = wrap(idx + step);
            if (smooth[static_cast<std::size_t>(j)] > val)
            {
                break;
            }
            rmin = std::min(rmin, smooth[static_cast<std::size_t>(j)]);
        }

        const float prominence = val - std::max(lmin, rmin);
        if (prominence < min_prominence)
        {
            continue;
        }

        const int li = wrap(idx - 20);
        const int ri = wrap(idx + 20);
        const float sharpness = val - 0.5f * (smooth[static_cast<std::size_t>(li)] +
                                              smooth[static_cast<std::size_t>(ri)]);
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

EdgeLabels classify_edges_cuda(
    const Signal &raw,
    const Corners &corners,
    float tol_factor)
{
    const int n = static_cast<int>(raw.size());
    EdgeLabels labels{
        EdgeType::Straight,
        EdgeType::Straight,
        EdgeType::Straight,
        EdgeType::Straight};
    if (n < 4)
    {
        return labels;
    }

    for (int e = 0; e < 4; ++e)
    {
        const int ca = corners[static_cast<std::size_t>(e)];
        const int cb = corners[static_cast<std::size_t>((e + 1) % 4)];

        const int edge_len = (cb - ca + n) % n;
        const int margin = edge_len / 5;

        const int start_idx = (ca + margin) % n;
        const int end_idx = (cb - margin + n) % n;
        const float v_start = raw[static_cast<std::size_t>(start_idx)];
        const float v_end = raw[static_cast<std::size_t>(end_idx)];

        const float shoulder_max = std::max(v_start, v_end);
        const float shoulder_min = std::min(v_start, v_end);

        float mx = -1e30f;
        float mn = 1e30f;
        for (int step = margin; step <= edge_len - margin; ++step)
        {
            const int i = (ca + step) % n;
            const float val = raw[static_cast<std::size_t>(i)];
            mx = std::max(mx, val);
            mn = std::min(mn, val);
        }

        const float local_tol = tol_factor * shoulder_max;

        if (mx > shoulder_max + local_tol)
        {
            labels[static_cast<std::size_t>(e)] = EdgeType::Knob;
        }
        else if (mn < shoulder_min - local_tol)
        {
            labels[static_cast<std::size_t>(e)] = EdgeType::Hole;
        }
        else
        {
            labels[static_cast<std::size_t>(e)] = EdgeType::Straight;
        }
    }

    return labels;
}

char edge_char_cuda(EdgeType e)
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
