#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "types.hpp"
#include "serial/signal_analysis.hpp"

// ______________________________________________________________________________________________

bool load_signal_csv(const std::string& filepath, Signal& signal)
{
    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    signal.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) signal.push_back(std::stof(line));
    }
    return true;
}

bool load_peaks_csv(const std::string& filepath, std::vector<int>& peaks)
{
    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    peaks.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) peaks.push_back(std::stoi(line));
    }
    return true;
}

// ______________________________________________________________________________________________

void test_smooth_signal()
{
    std::cout << "Running test_smooth_signal..." << std::endl;

    Signal raw_signal;
    assert(load_signal_csv("data/test_data/contour_to_signal/radial_signal.csv", raw_signal));

    Signal expected;
    assert(load_signal_csv("data/test_data/contour_to_signal/smoothed_signal.csv", expected));

    Signal actual = smooth_signal(raw_signal, 5);

    assert(actual.size() == expected.size());
    std::cout << "test_smooth_signal passed! (" << actual.size() << " floats)" << std::endl;
}

// ______________________________________________________________________________________________

void test_peak_detection()
{
    std::cout << "Running test_peak_detection..." << std::endl;

    Signal smoothed;
    assert(load_signal_csv("data/test_data/contour_to_signal/smoothed_signal.csv", smoothed));

    std::vector<int> expected_peaks;
    assert(load_peaks_csv("data/test_data/contour_to_signal/find_triangular_peaks.csv", expected_peaks));

    Corners actual_peaks = find_triangular_peaks(smoothed, 2.0f, 20.0f, 50);

    assert(expected_peaks.size() == 4);

    const int n = static_cast<int>(smoothed.size());
    const int tolerance = 3;

    for (size_t i = 0; i < 4; ++i) {
        int diff = std::abs(actual_peaks[i] - expected_peaks[i]);
        int circular_diff = std::min(diff, n - diff);
        assert(circular_diff <= tolerance && "Peak index mismatch!");
    }

    std::cout << "test_peak_detection passed! (All 4 corners within tolerance " << tolerance << ")" << std::endl;
}

// ______________________________________________________________________________________________

void test_classify_edges()
{
    std::cout << "Running test_classify_edges..." << std::endl;

    Signal smoothed;
    assert(load_signal_csv("data/test_data/contour_to_signal/smoothed_signal.csv", smoothed));

    Corners corners = find_triangular_peaks(smoothed, 2.0f, 20.0f, 50);
    std::array<EdgeType, 4> edge_labels = classify_edges(smoothed, corners, 0.1f);
    std::string shape = edges_to_string(edge_labels);

    for (int i = 1; i < 4; ++i) {
        assert(corners[i] > corners[i-1] && "Invalid corner ordering!");
    }
    assert(shape.length() == 4 && "Shape string wrong length!");
    assert(shape.find('?') == std::string::npos && "Unclassified edge detected!");

    std::cout << "test_classify_edges passed! (Shape: " << shape << ")" << std::endl;
}

// ______________________________________________________________________________________________

void test_puzzle_lookup_table()
{
    std::cout << "Running test_puzzle_lookup_table..." << std::endl;

    PuzzleLookupTable lookup;

    assert(lookup.getClassLabel("LVVC") == lookup.getClassLabel("VVCL"));
    assert(lookup.getClassLabel("LVVC").find(": ") != std::string::npos);
    assert(lookup.getClassLabel("LVC") == "Invalid");

    std::cout << "test_puzzle_lookup_table passed!" << std::endl;
}

// ______________________________________________________________________________________________

int main()
{
    test_smooth_signal();
    test_peak_detection();
    test_classify_edges();
    test_puzzle_lookup_table();

    std::cout << "All signal_analysis tests passed!" << std::endl;
    return 0;
}
