#pragma once

#include "serial/signal_analysis.hpp"

namespace parallel_signal {

Signal smooth_signal(
    const Signal& signal,
    int k
);

Corners find_triangular_peaks(
    const Signal& smooth,
    float min_prominence,
    float min_sharpness,
    int min_distance
);

EdgeLabels classify_edges(
    const Signal& raw,
    const Corners& corners,
    float tol_factor
);

} // namespace parallel_signal
