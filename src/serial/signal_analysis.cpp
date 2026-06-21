#include "serial/signal_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// --- Stage 1: Centered Moving Average
Signal smooth_signal(const Signal& signal, int k) {
    int n = signal.size();
    Signal smoothed(n);
    int pad = k / 2;

    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = -pad; j <= pad; ++j) {
            // Radial signals are circular, so the smoothing window wraps across
            // the first/last sample instead of shrinking at the contour seam.
            int idx = ((i + j) % n + n) % n; 
            sum += signal[idx];
        }
        smoothed[i] = sum / k;
    }
    
    return smoothed;
}

// --- Stage 2: Peak Detection & Filtering
Corners find_triangular_peaks(
    const Signal& smooth,
    float min_prominence,
    float min_sharpness,
    int   min_distance) 
{
    int n = static_cast<int>(smooth.size());
    Corners final_corners = {0, 0, 0, 0};
    if (n < 3) return final_corners;

    // Normalize any negative or overflowing index back into this closed signal.
    auto wrap = [n](int i) { return ((i % n) + n) % n; };

    // 1. Find local maxima (candidates)
    std::vector<int> candidates;
    for(int i = 0; i < n; ++i) {
        int l = (i == 0)     ? n - 1 : i - 1;
        int r = (i == n - 1) ? 0     : i + 1;
        if (smooth[i] >= smooth[l] && smooth[i] > smooth[r]) {
            candidates.push_back(i);
        }
    }

    // 2. Filter candidates by prominence, sharpness, and distance
    struct Cand { int idx; float val; };
    std::vector<Cand> kept;

    for (int idx : candidates) {
        float val = smooth[idx];

        // Circular prominence: walk left and right until another higher sample
        // bounds the candidate's basin. The weaker side limits the prominence.
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

        if (prom < min_prominence) continue;

        // Circular sharpness: a true puzzle corner should stand above shoulder
        // samples on both sides of the peak.
        int li = wrap(idx - 20);
        int ri = wrap(idx + 20);
        float sharp = val - 0.5f * (smooth[li] + smooth[ri]);

        if (sharp < min_sharpness) continue;

        // Circular minimum distance
        if (!kept.empty()) {
            int linear_dist = std::abs(idx - kept.back().idx);
            int circ_dist = std::min(linear_dist, n - linear_dist);
            
            // Suppress clusters of nearby local maxima, keeping the highest
            // sample in each minimum-distance neighborhood.
            if (circ_dist < min_distance) {
                if (val > kept.back().val) {
                    kept.back() = {idx, val};
                }
                continue;
            }
        }
        kept.push_back({idx, val});
    }

    // End of loop wrap distance check (last to first)
    if (kept.size() > 1) {
        // The first and last kept peaks may still be neighbors on the closed
        // contour even if they are far apart linearly.
        int linear_dist = std::abs(kept.front().idx - kept.back().idx);
        int circ_dist = std::min(linear_dist, n - linear_dist);
        if (circ_dist < min_distance) {
            if (kept.back().val > kept.front().val) {
                kept.front() = kept.back();
            }
            kept.pop_back();
        }
    }

    // 3. Greedy max-min distance corner selection (Guarantees exactly 4 corners)
    std::vector<Cand> remaining = kept;
    // Start from the strongest remaining peak.
    std::sort(remaining.begin(), remaining.end(), 
              [](const Cand& a, const Cand& b){ return a.val > b.val; });

    std::vector<int> corners;
    if (!remaining.empty()) corners.push_back(remaining[0].idx); 

    while (corners.size() < 4 && !remaining.empty()) {
        int best_idx = -1;
        int best_dist = -1;
        
        for (const auto& pk : remaining) {
            int min_d = n;
            for (int c : corners) {
                // Distance on the closed contour is the shorter of the forward
                // and wraparound arcs.
                int d = std::abs(pk.idx - c);
                min_d = std::min(min_d, std::min(d, n - d));
            }
            if (min_d > best_dist) {
                best_dist = min_d;
                best_idx  = pk.idx;
            }
        }
        
        if (best_idx != -1) {
            corners.push_back(best_idx);
            // Remove the selected peak so it cannot be chosen again.
            remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
                [best_idx](const Cand& pk){ return pk.idx == best_idx; }),
                remaining.end());
        }
    }

    // Edge traversal expects contour order, not peak-strength selection order.
    std::sort(corners.begin(), corners.end());
    for (int i = 0; i < std::min(4, (int)corners.size()); ++i) {
        final_corners[i] = corners[i];
    }
    
    return final_corners;
}

// --- Stage 3: Edge Classification
char edge_char(EdgeType e) {
    switch (e) {
        case EdgeType::Straight: return 'L';
        case EdgeType::Knob:     return 'V';
        case EdgeType::Hole:     return 'C';
    }
    return '?';
}

EdgeLabels classify_edges(
    const Signal& raw,
    const Corners& corners,
    float tol_factor)
{
    int n = (int)raw.size();
    EdgeLabels labels{ EdgeType::Straight, EdgeType::Straight,
                        EdgeType::Straight, EdgeType::Straight };
    if (n < 4) return labels;

    for (int e = 0; e < 4; ++e) {
        int ca = corners[e];
        int cb = corners[(e + 1) % 4];

        // The modulo handles the fourth edge, which wraps from corner 3 back
        // to corner 0.
        int edge_len = (cb - ca + n) % n;
        int margin = edge_len / 5; 
        
        // 1. Find the "shoulders". Ignore 20% near both corners because corner
        // shape dominates the radial signal there.
        int start_idx = (ca + margin) % n;
        int end_idx = (cb - margin + n) % n;
        float v_start = raw[start_idx];
        float v_end = raw[end_idx];
        
        float shoulder_max = std::max(v_start, v_end);
        float shoulder_min = std::min(v_start, v_end);

        // 2. Find the absolute max and min in the middle of the edge.
        float mx = -1e30f, mn = 1e30f;
        for (int step = margin; step <= edge_len - margin; ++step) {
            int i = (ca + step) % n;
            float val = raw[i];
            mx = std::max(mx, val);
            mn = std::min(mn, val);
        }

        // 3. Dynamic tolerance based on the scale of this specific edge
        float local_tol = tol_factor * shoulder_max; 

        // A bump above both shoulders is a knob; a dip below both shoulders is
        // a hole. Otherwise the edge is straight.
        if (mx > shoulder_max + local_tol) {
            labels[e] = EdgeType::Knob;     // V
        } 
        else if (mn < shoulder_min - local_tol) {
            labels[e] = EdgeType::Hole;     // C
        } 
        else {
            labels[e] = EdgeType::Straight; // L
        }
    }
    
    return labels;
} 

std::string edges_to_string(const EdgeLabels& labels) {
    std::string s;
    // Convert enum labels to the compact four-character string used by the
    // lookup table and debug output.
    for (auto e : labels) s += edge_char(e);
    return s;  
}

int PuzzleLookupTable::charToDigit(char c) const {
    if (c == 'L') return 0;
    if (c == 'V') return 1;
    if (c == 'C') return 2;
    return 0;
}

int PuzzleLookupTable::getBase3Index(const std::string& s) const {
    // Treat L/V/C as base-3 digits so every four-edge string maps to [0, 80].
    return charToDigit(s[0]) * 27 +
           charToDigit(s[1]) * 9 +
           charToDigit(s[2]) * 3 +
           charToDigit(s[3]);
}

std::string PuzzleLookupTable::rotate(const std::string& s) const {
    return s.substr(1) + s[0];
}

std::string PuzzleLookupTable::getCategory(const std::string& s) const {
    // The number of straight edges determines the broad puzzle-piece family.
    int l_count = std::count(s.begin(), s.end(), 'L');
    if (l_count == 0) return "I";
    if (l_count == 1) return "E";
    if (l_count == 2) return "C";
    if (l_count == 3) return "T";
    if (l_count == 4) return "S";
    return "U";
}

PuzzleLookupTable::PuzzleLookupTable() {
    table.fill("");

    int count_I = 0, count_E = 0, count_C = 0, count_T = 0, count_S = 0;
    const char edges[] = {'L', 'V', 'C'};

    // Precompute all 3^4 edge combinations. Rotations of the same edge string
    // get the same class label, so runtime lookup is a single base-3 index.
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    std::string s = {edges[i], edges[j], edges[k], edges[l]};
                    int idx = getBase3Index(s);

                    if (table[idx].empty()) {
                        std::string r1 = rotate(s);
                        std::string r2 = rotate(r1);
                        std::string r3 = rotate(r2);

                        // Lexicographically smallest rotation is the canonical
                        // representative shown in the class label.
                        std::string canonical = s;
                        if (r1 < canonical) canonical = r1;
                        if (r2 < canonical) canonical = r2;
                        if (r3 < canonical) canonical = r3;

                        std::string cat = getCategory(canonical);
                        int class_id = 0;

                        if (cat == "I") class_id = count_I++;
                        else if (cat == "E") class_id = count_E++;
                        else if (cat == "C") class_id = count_C++;
                        else if (cat == "T") class_id = count_T++;
                        else if (cat == "S") class_id = count_S++;

                        std::string final_label = cat + "_" + std::to_string(class_id) + ": " + canonical;

                        // Fill every rotation with the same label so callers do
                        // not need to normalize the edge string first.
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

std::string PuzzleLookupTable::getClassLabel(const std::string& edges) const {
    if (edges.length() != 4) return "Invalid";
    return table[getBase3Index(edges)];
}
