// signal_analysis_main.cu
//
// Entry point for the signal analysis pipeline.
// Functions, types, and CUDA kernels live in signal_analysis_funcs.cuh.
//
// Build:
//   nvcc -O2 -std=c++17 -arch=sm_70 signal_analysis_main.cu -o signal_analysis
//
// Run (verbose mode on by default):
//   ./signal_analysis smoothed_signal.csv
//   ./signal_analysis smoothed_signal.csv 5          # window=5
//   ./signal_analysis smoothed_signal.csv 5 quiet    # suppress per-step printout

#include "signal_analysis_funcs.cuh"

bool VERBOSE = true;


// ============================================================================
//  main
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <smoothed_signal.csv> [window=5] [quiet]\n";
        return 1;
    }

    const std::string in_path = argv[1];
    Params p;
    if (argc >= 3) p.window = std::atoi(argv[2]);
    if (argc >= 4 && std::string(argv[3]) == "quiet") VERBOSE = false;

    LOG("=== signal_analysis ===");
    
    // --- 1. Load on Host
    std::vector<float> h_signal = load_signal_csv(in_path);
    LOG("signal length: " << h_signal.size());

    // --- 2. Upload to Device (H2D - Once)
    thrust::device_vector<float> d_signal = h_signal;

    // --- 3. Execute Device pipeline (D2D)
    thrust::device_vector<float> d_smooth     = smooth_signal(d_signal, p.window);
    thrust::device_vector<int>   d_candidates = find_local_maxima(d_smooth);

    // --- 4. Download Results (D2H - Once)
    std::vector<float> h_smooth(d_smooth.size());
    thrust::copy(d_smooth.begin(), d_smooth.end(), h_smooth.begin());
    
    std::vector<int> h_candidates(d_candidates.size());
    thrust::copy(d_candidates.begin(), d_candidates.end(), h_candidates.begin());

    // --- 5. Host-side sequential filtering & classification
    save_floats("smoothed.csv", h_smooth, "r_smooth");

    auto peaks = find_triangular_peaks(h_smooth, h_candidates, p);

    {
        std::ofstream out("peaks.csv");
        out << "index,value\n";
        out << std::fixed << std::setprecision(6);
        for (const auto& pk : peaks)
            out << pk.index << "," << pk.value << "\n";
        LOG("\n  saved " << peaks.size() << " peaks -> peaks.csv");
    }

    auto label = classify_edges(h_signal, peaks, p);
    std::ofstream("label.txt") << label << "\n";
    
    // Final Summary
    std::cout << "\n=== result ===\n"
              << "signal length : " << h_signal.size() << "\n"
              << "peaks found   : " << peaks.size() << "\n"
              << "label         : " << label << "\n";

    return 0;
}
