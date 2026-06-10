#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>
#include <cstdint>
#include <set>
#include <array>


std::vector<float> smooth_signal(const std::vector<float>& signal, int k) {
    int n = signal.size();
    std::vector<float> smoothed(n);

    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < k; ++j) {
            
            sum += signal[(i + j) % n];
        }
        smoothed[i] = sum / k;
    }
    
    return smoothed;
}

std::vector<float> compute_1d_sharpness(const std::vector<float>& radial_signal, int k) {
    int n = static_cast<int>(radial_signal.size());
    std::vector<float> sharpness(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        float curr = radial_signal[i];
        float prev = radial_signal[(i - k + n) % n];
        float next = radial_signal[(i + k) % n];

        sharpness[i] = (curr - prev) + (curr - next);
    }
    
    return sharpness;
}



std::array<int, 4> find_triangular_peaks(
    const std::vector<float>& curvature,
    float min_prominence,
    int   min_distance) 
{
    int n = static_cast<int>(curvature.size());
    std::array<int, 4> final_corners = {0, 0, 0, 0};
    if (n < 3) return final_corners;

    struct Cand { int idx; float val; float prom; };
    std::vector<Cand> cands;

    for (int i = 0; i < n; ++i) {
        float prev = curvature[(i - 1 + n) % n];
        float curr = curvature[i];
        float next = curvature[(i + 1) % n];

        if (curr > prev && curr > next) {
            float left_min  = curr;
            float right_min = curr;

            for (int s = 1; s < n; ++s) {
                float v = curvature[(i - s + n) % n];
                if (v > curr) break;
                left_min = std::min(left_min, v);
            }
            for (int s = 1; s < n; ++s) {
                float v = curvature[(i + s) % n];
                if (v > curr) break;
                right_min = std::min(right_min, v);
            }

            float prominence = curr - std::max(left_min, right_min);
            if (prominence >= min_prominence) {
                cands.push_back({i, curr, prominence});
            }
        }
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.prom > b.prom; });

    std::vector<Cand> kept;
    for (const auto& c : cands) {
        bool ok = true;
        for (const auto& k : kept) {
            int d = std::abs(c.idx - k.idx);
            d = std::min(d, n - d);          // circular distance
            if (d < min_distance) { ok = false; break; }
        }
        if (ok) kept.push_back(c);
    }
    int num_found = std::min(4, static_cast<int>(kept.size()));
    kept.resize(num_found); 

    std::sort(kept.begin(), kept.end(),
              [](const Cand& a, const Cand& b) { return a.idx < b.idx; });

    for (int i = 0; i < num_found; ++i) {
        final_corners[i] = kept[i].idx;
    }
    
    return final_corners;
}



enum class EdgeType { Straight, Knob, Hole };

inline char edge_char(EdgeType e) {
    switch (e) {
        case EdgeType::Straight: return 'L';
        case EdgeType::Knob:     return 'V';
        case EdgeType::Hole:     return 'C';
    }
    return '?';
}

std::array<EdgeType,4> classify_edges(
    const std::vector<float>& signal,
    const std::array<int,4>& corner_idx,
    float knob_factor = 1.08f,
    float hole_factor = 0.92f)
{
    int n = (int)signal.size();
    std::array<EdgeType,4> labels{ EdgeType::Straight, EdgeType::Straight,
                                   EdgeType::Straight, EdgeType::Straight };
    if (n < 4) return labels;

    std::vector<float> sorted_sig = signal;
    std::sort(sorted_sig.begin(), sorted_sig.end());
    float base = sorted_sig[n / 2];
    if (base <= 0.0f) return labels;

    for (int e = 0; e < 4; ++e) {
        int start = corner_idx[e];
        int end   = corner_idx[(e + 1) % 4];

        std::vector<float> seg;
        for (int i = start; i != end; i = (i + 1) % n)
            seg.push_back(signal[i]);
        if (seg.size() < 4) { labels[e] = EdgeType::Straight; continue; }

        int mlo = (int)(seg.size() * 0.25f);
        int mhi = (int)(seg.size() * 0.75f);
        if (mhi <= mlo) mhi = mlo + 1;

        float sum = 0.0f;
        for (int i = mlo; i < mhi; ++i) sum += seg[i];
        float mid = sum / (mhi - mlo);

        float ratio = mid / base;
        if      (ratio > knob_factor) labels[e] = EdgeType::Knob;   // V
        else if (ratio < hole_factor) labels[e] = EdgeType::Hole;   // C
        else                          labels[e] = EdgeType::Straight; // L
    }
    return labels;
} 

std::string edges_to_string(const std::array<EdgeType,4>& labels) {
    std::string s;
    for (auto e : labels) s += edge_char(e);
    return s;  
}


