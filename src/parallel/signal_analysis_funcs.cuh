// signal_analysis_funcs.cuh
#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <array>


// ============================================================================
//  Global verbose flag -- set to false with "quiet" argument
// ============================================================================

extern bool VERBOSE;

// Simple log helper: only prints when VERBOSE is true
#define LOG(msg) do { if (VERBOSE) { std::cout << msg << "\n"; } } while(0)


// ============================================================================
//  Parameters
// ============================================================================

struct Params {
    int   window         = 5;     // smoothing window (k)
    int   min_distance   = 5;     // minimum index gap between two kept peaks
    float min_prominence = 5.0f;  // how much a peak must stand above its base
    float min_sharpness  = 20.0f; // how much a peak must stand above ±20 neighbours
    float tol_factor     = 0.1f;  // edge classification: tol = tol_factor * mean_r
};


// ============================================================================
//  I/O helpers
// ============================================================================

static std::vector<float> load_signal_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "[ERROR] cannot open signal file: " << path << "\n";
        std::exit(1);
    }
    std::vector<float> v;
    std::string line;
    std::getline(in, line); // skip header
    while (std::getline(in, line))
        if (!line.empty()) v.push_back(std::stof(line));

    if (v.empty()) {
        std::cerr << "[ERROR] signal file is empty: " << path << "\n";
        std::exit(1);
    }
    return v;
}

static void save_floats(const std::string& path,
                        const std::vector<float>& v,
                        const std::string& header = "") {
    std::ofstream out(path);
    if (!header.empty()) out << header << "\n";
    out << std::fixed << std::setprecision(6);
    for (float x : v) out << x << "\n";
    LOG("  saved " << v.size() << " floats -> " << path);
}

class PuzzleLookupTable {
private:
    // Flat array now holds formatted strings instead of ints
    std::array<std::string, 81> table;

    // Helper: Convert edge character to Base-3 digit
    int charToDigit(char c) const {
        if (c == 'L') return 0; // Straight
        if (c == 'V') return 1; // Knob
        if (c == 'C') return 2; // Hole
        return 0; 
    }

    // Helper: Convert a 4-character string to an integer index (0 - 80)
    int getBase3Index(const std::string& s) const {
        return charToDigit(s[0]) * 27 + 
               charToDigit(s[1]) * 9 + 
               charToDigit(s[2]) * 3 + 
               charToDigit(s[3]) * 1;
    }

    // Helper: Rotate string left by one position
    std::string rotate(const std::string& s) const {
        return s.substr(1) + s[0];
    }

    // Helper: Determine category based on the number of straight edges
    std::string getCategory(const std::string& s) const {
        int l_count = std::count(s.begin(), s.end(), 'L');
        if (l_count == 0) return "I"; // Interior
        if (l_count == 1) return "E"; // Edge
        if (l_count == 2) return "C"; // Corner
        if (l_count == 3) return "T"; // Terminal (Rare/Edge case)
        if (l_count == 4) return "S"; // Square/Standalone
        return "U"; // Unknown
    }

public:
    PuzzleLookupTable() {
        // Initialize with empty strings
        table.fill("");
        
        // Track unique IDs per category
        int count_I = 0, count_E = 0, count_C = 0, count_T = 0, count_S = 0;

        const char edges[] = {'L', 'V', 'C'};

        // Precompute the table for all 81 combinations
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        std::string s = {edges[i], edges[j], edges[k], edges[l]};
                        int idx = getBase3Index(s);

                        // If this combination hasn't been processed yet
                        if (table[idx].empty()) {
                            // 1. Generate all rotations
                            std::string r1 = rotate(s);
                            std::string r2 = rotate(r1);
                            std::string r3 = rotate(r2);

                            // 2. Find the canonical (alphabetically smallest) rotation for consistent display
                            std::string canonical = s;
                            if (r1 < canonical) canonical = r1;
                            if (r2 < canonical) canonical = r2;
                            if (r3 < canonical) canonical = r3;

                            // 3. Determine category and assign the next unique ID
                            std::string cat = getCategory(canonical);
                            int class_id = 0;
                            
                            if (cat == "I") class_id = count_I++;
                            else if (cat == "E") class_id = count_E++;
                            else if (cat == "C") class_id = count_C++;
                            else if (cat == "T") class_id = count_T++;
                            else if (cat == "S") class_id = count_S++;

                            // 4. Construct the final formatted string (e.g., "E_1: CVCL")
                            std::string final_label = cat + "_" + std::to_string(class_id) + ": " + canonical;

                            // 5. Assign this same exact label to ALL rotations of this piece
                            table[getBase3Index(s)]  = final_label;
                            table[getBase3Index(r1)] = final_label;
                            table[getBase3Index(r2)] = final_label;
                            table[getBase3Index(r3)] = final_label;
                        }
                    }
                }
            }
        }
    }

    // $O(1)$ Lookup function returning the formatted string
    std::string getClassLabel(const std::string& edges) const {
        if (edges.length() != 4) return "Invalid";
        return table[getBase3Index(edges)];
    }
};


// ============================================================================
//  Stage 1 -- moving-average smoothing  (Thrust, Device-to-Device)
// ============================================================================

struct moving_average_op {
    const float* src;
    int n;
    int k;
    int pad;

    __host__ __device__
    float operator()(int i) const {
        float sum = 0.0f;
        for (int j = -pad; j <= pad; ++j) {
            int idx = (i + j) % n;
            if (idx < 0) idx += n; // wrap negative indices
            sum += src[idx];
        }
        return sum / k;
    }
};

static thrust::device_vector<float>
smooth_signal(const thrust::device_vector<float>& d_in, int window) {
    LOG("\n[Stage 1] smoothing  window=" << window);

    int n = d_in.size();
    thrust::device_vector<float> d_out(n);
    int pad = window / 2;

    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(n),
        d_out.begin(),
        moving_average_op{ thrust::raw_pointer_cast(d_in.data()), n, window, pad }
    );

    return d_out;
}


// ============================================================================
//  Stage 2a & 2b -- local-maximum mask & stream compaction (Device-to-Device)
// ============================================================================

struct is_local_max_op {
    const float* src;
    int n;

    __host__ __device__
    int operator()(int i) const {
        int l = (i == 0)     ? n - 1 : i - 1;
        int r = (i == n - 1) ? 0     : i + 1;
        return (src[i] >= src[l] && src[i] > src[r]) ? 1 : 0;
    }
};

struct mask_is_set {
    const int* mask;
    __host__ __device__
    bool operator()(int i) const { return mask[i] != 0; }
};

static thrust::device_vector<int>
find_local_maxima(const thrust::device_vector<float>& d_smooth) {
    LOG("\n[Stage 2] local maxima  (mask + stream compaction)");

    int n = d_smooth.size();
    thrust::device_vector<int> d_mask(n);

    // 0/1 mask
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(n),
        d_mask.begin(),
        is_local_max_op{ thrust::raw_pointer_cast(d_smooth.data()), n }
    );

    // stream compaction
    thrust::device_vector<int> d_idx(n);
    auto end = thrust::copy_if(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(n),
        d_idx.begin(),
        mask_is_set{ thrust::raw_pointer_cast(d_mask.data()) }
    );
    d_idx.resize(end - d_idx.begin());

    LOG("  local maxima found (on device): " << d_idx.size());
    return d_idx;
}


// ============================================================================
//  Stage 2c -- prominence / sharpness / distance filtering (Host)
// ============================================================================

struct Peak { int index; float value; };

static std::vector<Peak>
find_triangular_peaks(const std::vector<float>& smooth,
                   const std::vector<int>&   candidates,
                   const Params&             p) {
    LOG("\n[Stage 2c] filter peaks" 
        "  min_dist=" << p.min_distance <<
        "  min_prom=" << p.min_prominence <<
        "  min_sharp=" << p.min_sharpness);

    const int n = smooth.size();
    std::vector<Peak> kept;
    auto wrap = [n](int i) { return ((i % n) + n) % n; };

    for (int idx : candidates) {
        float val = smooth[idx];

        // --- Circular prominence
        float lmin = val, rmin = val;
        for (int step = 1; step < n; ++step) {
            int j = wrap(idx - step);
            if (smooth[j] > val) break;
            lmin = std::min(lmin, smooth[j]);
        }
        for (int step = 1; step < n; ++step) {
            int j = wrap(idx + step);
            if (smooth[j] > val) break;
            rmin = std::min(rmin, smooth[j]);
        }
        float prom = val - std::max(lmin, rmin);

        if (prom < p.min_prominence) {
            LOG("  drop idx=" << idx << " val=" << val << "  prom=" << prom << " < " << p.min_prominence);
            continue;
        }

        // --- Circular sharpness
        int li = wrap(idx - 20);
        int ri = wrap(idx + 20);
        float sharp = val - 0.5f * (smooth[li] + smooth[ri]);

        if (sharp < p.min_sharpness) {
            LOG("  drop idx=" << idx << " val=" << val << "  sharp=" << sharp << " < " << p.min_sharpness);
            continue;
        }

        // --- Circular minimum distance
        if (!kept.empty()) {
            int linear_dist = std::abs(idx - kept.back().index);
            int circ_dist = std::min(linear_dist, n - linear_dist);
            
            if (circ_dist < p.min_distance) {
                if (val > kept.back().value) {
                    LOG("  replace idx=" << kept.back().index << " with idx=" << idx << " (closer but higher)");
                    kept.back() = {idx, val};
                } else {
                    LOG("  drop idx=" << idx << " too close to " << kept.back().index);
                }
                continue;
            }
        }

        LOG("  keep idx=" << idx << " val=" << val << "  prom=" << prom << "  sharp=" << sharp);
        kept.push_back({idx, val});
    }

    // End of loop wrap distance check (last to first)
    if (kept.size() > 1) {
        int linear_dist = std::abs(kept.front().index - kept.back().index);
        int circ_dist = std::min(linear_dist, n - linear_dist);
        if (circ_dist < p.min_distance) {
            if (kept.back().value > kept.front().value) {
                kept.front() = kept.back();
            }
            kept.pop_back();
        }
    }

    LOG("  peaks after filtering: " << kept.size());
    return kept;
}


// ============================================================================
//  Stage 3 -- edge classification
// ============================================================================

static std::string
classify_edges(const std::vector<float>& raw,
               const std::vector<Peak>&  peaks,
               const Params&             p) {
    LOG("\n[Stage 3] classification  tol_factor=" << p.tol_factor);

    if (peaks.size() < 4) {
        std::cerr << "[ERROR] need at least 4 peaks to classify, got " << peaks.size() << "\n";
        return "????";
    }

    const int n = raw.size();
    float mean_r = std::accumulate(raw.begin(), raw.end(), 0.0f) / n;
    float tol    = p.tol_factor * mean_r;
    LOG("  mean_radius=" << mean_r << "  tol=" << tol);

    // --- Greedy max-min distance corner selection
    std::vector<Peak> remaining = peaks;
    std::sort(remaining.begin(), remaining.end(), 
              [](const Peak& a, const Peak& b){ return a.value > b.value; });

    std::vector<int> corners;
    corners.push_back(remaining[0].index); // Start with global max

    while (corners.size() < 4 && !remaining.empty()) {
        int best_idx = -1;
        int best_dist = -1;
        
        for (const auto& pk : remaining) {
            int min_d = n;
            for (int c : corners) {
                int d = std::abs(pk.index - c);
                min_d = std::min(min_d, std::min(d, n - d)); // Circular distance
            }
            if (min_d > best_dist) {
                best_dist = min_d;
                best_idx  = pk.index;
            }
        }
        
        if (best_idx != -1) {
            corners.push_back(best_idx);
            remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
                [best_idx](const Peak& pk){ return pk.index == best_idx; }),
                remaining.end());
        }
    }

    std::sort(corners.begin(), corners.end());
    LOG("  corners (sorted by contour index): "
        << corners[0] << " " << corners[1] << " " << corners[2] << " " << corners[3]);

    std::string label(4, '?');

    for (int e = 0; e < 4; ++e) {
        int ca = corners[e];
        int cb = corners[(e + 1) % 4];

        int edge_len = (cb - ca + n) % n;
        int margin = edge_len / 5; 
        
        // 1. Find the "shoulders" (the signal values at the edges of our crop)
        int start_idx = (ca + margin) % n;
        int end_idx = (cb - margin + n) % n;
        float v_start = raw[start_idx];
        float v_end = raw[end_idx];
        
        float shoulder_max = std::max(v_start, v_end);
        float shoulder_min = std::min(v_start, v_end);

        // 2. Find the absolute max and min in the cropped region
        float mx = -1e30f, mn = 1e30f;
        int count = 0;
        for (int step = margin; step <= edge_len - margin; ++step) {
            int i = (ca + step) % n;
            float val = raw[i];
            mx = std::max(mx, val);
            mn = std::min(mn, val);
            ++count;
        }

        // 3. Dynamic tolerance based on the scale of this specific edge
        float local_tol = p.tol_factor * shoulder_max; 

        char c;
        // A Tab bulges OUTWARD past the highest shoulder
        if (mx > shoulder_max + local_tol) {
            c = 'V';
        } 
        // A Cutout caves INWARD deeply past the lowest shoulder
        else if (mn < shoulder_min - local_tol) {
            c = 'C';
        } 
        // If it roughly stays between the shoulders, it's flat
        else {
            c = 'L';
        }
        
        label[e] = c;

        LOG("  edge " << e << "  corners [" << ca << "->" << cb << "]"
            << "\n      shoulders=[" << v_start << ", " << v_end << "]"
            << "  max=" << mx << "  min=" << mn
            << "\n      -> '" << c << "'");
    }

    LOG("  label: " << label);
    return label;
}
