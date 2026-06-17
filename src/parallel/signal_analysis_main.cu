#include "signal_analysis_funcs.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace {

struct MovingAverageOp {
    const float* src;
    int n;
    int k;
    int pad;

    __host__ __device__
    float operator()(int i) const
    {
        float sum = 0.0f;
        for (int j = -pad; j <= pad; ++j) {
            int idx = (i + j) % n;
            if (idx < 0) {
                idx += n;
            }
            sum += src[idx];
        }
        return sum / k;
    }
};

struct IsLocalMaxOp {
    const float* src;
    int n;

    __host__ __device__
    int operator()(int i) const
    {
        const int l = (i == 0) ? n - 1 : i - 1;
        const int r = (i == n - 1) ? 0 : i + 1;
        return (src[i] >= src[l] && src[i] > src[r]) ? 1 : 0;
    }
};

struct MaskIsSet {
    const int* mask;

    __host__ __device__
    bool operator()(int i) const
    {
        return mask[i] != 0;
    }
};

struct Candidate {
    int idx;
    float val;
};

std::vector<int> find_local_maxima_cuda(const Signal& smooth)
{
    const int n = static_cast<int>(smooth.size());
    if (n < 3) {
        return {};
    }

    thrust::device_vector<float> d_smooth(smooth.begin(), smooth.end());
    thrust::device_vector<int> d_mask(static_cast<std::size_t>(n));

    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(n),
        d_mask.begin(),
        IsLocalMaxOp{thrust::raw_pointer_cast(d_smooth.data()), n});

    thrust::device_vector<int> d_candidates(static_cast<std::size_t>(n));
    auto end = thrust::copy_if(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(n),
        d_candidates.begin(),
        MaskIsSet{thrust::raw_pointer_cast(d_mask.data())});
    d_candidates.resize(static_cast<std::size_t>(end - d_candidates.begin()));

    std::vector<int> candidates(d_candidates.size());
    thrust::copy(d_candidates.begin(), d_candidates.end(), candidates.begin());
    return candidates;
}

} // namespace

namespace parallel_signal {

Signal smooth_signal(const Signal& signal, int k)
{
    const int n = static_cast<int>(signal.size());
    Signal smoothed(static_cast<std::size_t>(n));
    if (n == 0) {
        return smoothed;
    }

    thrust::device_vector<float> d_signal(signal.begin(), signal.end());
    thrust::device_vector<float> d_smoothed(static_cast<std::size_t>(n));
    const int pad = k / 2;

    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(n),
        d_smoothed.begin(),
        MovingAverageOp{thrust::raw_pointer_cast(d_signal.data()), n, k, pad});

    thrust::copy(d_smoothed.begin(), d_smoothed.end(), smoothed.begin());
    return smoothed;
}

Corners find_triangular_peaks(
    const Signal& smooth,
    float min_prominence,
    float min_sharpness,
    int min_distance)
{
    const int n = static_cast<int>(smooth.size());
    Corners final_corners = {0, 0, 0, 0};
    if (n < 3) {
        return final_corners;
    }

    const std::vector<int> candidates = find_local_maxima_cuda(smooth);
    auto wrap = [n](int i) { return ((i % n) + n) % n; };

    std::vector<Candidate> kept;
    for (int idx : candidates) {
        const float val = smooth[static_cast<std::size_t>(idx)];

        float lmin = val;
        float rmin = val;
        for (int step = 1; step < n; ++step) {
            const int j = wrap(idx - step);
            if (smooth[static_cast<std::size_t>(j)] > val) {
                break;
            }
            lmin = std::min(lmin, smooth[static_cast<std::size_t>(j)]);
        }
        for (int step = 1; step < n; ++step) {
            const int j = wrap(idx + step);
            if (smooth[static_cast<std::size_t>(j)] > val) {
                break;
            }
            rmin = std::min(rmin, smooth[static_cast<std::size_t>(j)]);
        }

        const float prominence = val - std::max(lmin, rmin);
        if (prominence < min_prominence) {
            continue;
        }

        const int li = wrap(idx - 20);
        const int ri = wrap(idx + 20);
        const float sharpness = val - 0.5f * (
            smooth[static_cast<std::size_t>(li)] +
            smooth[static_cast<std::size_t>(ri)]);
        if (sharpness < min_sharpness) {
            continue;
        }

        if (!kept.empty()) {
            const int linear_dist = std::abs(idx - kept.back().idx);
            const int circ_dist = std::min(linear_dist, n - linear_dist);

            if (circ_dist < min_distance) {
                if (val > kept.back().val) {
                    kept.back() = {idx, val};
                }
                continue;
            }
        }

        kept.push_back({idx, val});
    }

    if (kept.size() > 1) {
        const int linear_dist = std::abs(kept.front().idx - kept.back().idx);
        const int circ_dist = std::min(linear_dist, n - linear_dist);
        if (circ_dist < min_distance) {
            if (kept.back().val > kept.front().val) {
                kept.front() = kept.back();
            }
            kept.pop_back();
        }
    }

    std::vector<Candidate> remaining = kept;
    std::sort(remaining.begin(), remaining.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.val > b.val;
              });

    std::vector<int> corners;
    if (!remaining.empty()) {
        corners.push_back(remaining.front().idx);
    }

    while (corners.size() < 4 && !remaining.empty()) {
        int best_idx = -1;
        int best_dist = -1;

        for (const auto& peak : remaining) {
            int min_d = n;
            for (int corner : corners) {
                const int d = std::abs(peak.idx - corner);
                min_d = std::min(min_d, std::min(d, n - d));
            }
            if (min_d > best_dist) {
                best_dist = min_d;
                best_idx = peak.idx;
            }
        }

        if (best_idx != -1) {
            corners.push_back(best_idx);
            remaining.erase(
                std::remove_if(
                    remaining.begin(),
                    remaining.end(),
                    [best_idx](const Candidate& peak) {
                        return peak.idx == best_idx;
                    }),
                remaining.end());
        }
    }

    std::sort(corners.begin(), corners.end());
    for (int i = 0; i < std::min(4, static_cast<int>(corners.size())); ++i) {
        final_corners[static_cast<std::size_t>(i)] = corners[static_cast<std::size_t>(i)];
    }

    return final_corners;
}

EdgeLabels classify_edges(
    const Signal& raw,
    const Corners& corners,
    float tol_factor)
{
    const int n = static_cast<int>(raw.size());
    EdgeLabels labels{
        EdgeType::Straight,
        EdgeType::Straight,
        EdgeType::Straight,
        EdgeType::Straight};
    if (n < 4) {
        return labels;
    }

    for (int e = 0; e < 4; ++e) {
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
        for (int step = margin; step <= edge_len - margin; ++step) {
            const int i = (ca + step) % n;
            const float val = raw[static_cast<std::size_t>(i)];
            mx = std::max(mx, val);
            mn = std::min(mn, val);
        }

        const float local_tol = tol_factor * shoulder_max;

        if (mx > shoulder_max + local_tol) {
            labels[static_cast<std::size_t>(e)] = EdgeType::Knob;
        } else if (mn < shoulder_min - local_tol) {
            labels[static_cast<std::size_t>(e)] = EdgeType::Hole;
        } else {
            labels[static_cast<std::size_t>(e)] = EdgeType::Straight;
        }
    }

    return labels;
}

} // namespace parallel_signal
