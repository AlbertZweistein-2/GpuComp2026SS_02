#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "types.hpp"
#include "serial/contour_to_signal.hpp"

// ______________________________________________________________________________________________

bool load_boundary_csv(const std::string& filepath, std::vector<uint8_t>& boundary, int& width, int& height)
{
    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    boundary.clear();
    std::string line, val;
    height = 0;
    width = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int current_width = 0;
        while (std::getline(ss, val, ',')) {
            boundary.push_back(static_cast<uint8_t>(std::stoi(val)));
            current_width++;
        }
        if (height == 0) width = current_width;
        height++;
    }
    return true;
}

bool load_contour_csv(const std::string& filepath, CoordinateVector<int>& contour)
{
    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    contour.clear();
    std::string line, val1, val2;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        if (std::getline(ss, val1, ',') && std::getline(ss, val2, ',')) {
            contour.push_back({std::stoi(val1), std::stoi(val2)});
        }
    }
    return true;
}

bool load_circle_txt(const std::string& filepath, Coordinate<float>& center, float& radius)
{
    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    std::string line, val1, val2;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::getline(ss, val1, ',');
        std::getline(ss, val2, ',');
        center.a = std::stof(val1);
        center.b = std::stof(val2);
    }
    if (std::getline(file, line)) {
        radius = std::stof(line);
    }
    return true;
}

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

// ______________________________________________________________________________________________

void test_trace_contour()
{
    std::cout << "Running test_trace_contour..." << std::endl;

    std::vector<uint8_t> boundary;
    int width, height;
    assert(load_boundary_csv("data/test_data/contour_to_signal/boundary_input.csv", boundary, width, height));

    CoordinateVector<int> expected;
    assert(load_contour_csv("data/test_data/contour_to_signal/trace_contour.csv", expected));

    CoordinateVector<int> actual = trace_contour(boundary, width, height);

    assert(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(actual[i] == expected[i] && "trace_contour mismatch!");
    }
    std::cout << "test_trace_contour passed! (" << actual.size() << " points match)" << std::endl;
}

// ______________________________________________________________________________________________

void test_simplify_chain_approx()
{
    std::cout << "Running test_simplify_chain_approx..." << std::endl;

    CoordinateVector<int> input;
    assert(load_contour_csv("data/test_data/contour_to_signal/trace_contour.csv", input));

    CoordinateVector<int> expected;
    assert(load_contour_csv("data/test_data/contour_to_signal/simplify_chain_approx.csv", expected));

    CoordinateVector<int> actual = simplify_chain_approx(input);

    assert(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(actual[i] == expected[i] && "simplify_chain_approx mismatch!");
    }
    std::cout << "test_simplify_chain_approx passed! (" << actual.size() << " points match)" << std::endl;
}

// ______________________________________________________________________________________________

void test_find_contour_chain_approx_simple()
{
    std::cout << "Running test_find_contour_chain_approx_simple..." << std::endl;

    std::vector<uint8_t> boundary;
    int width, height;
    assert(load_boundary_csv("data/test_data/contour_to_signal/boundary_input.csv", boundary, width, height));

    CoordinateVector<int> expected;
    assert(load_contour_csv("data/test_data/contour_to_signal/find_contour_chain_approx_simple.csv", expected));

    CoordinateVector<int> actual = find_contour_chain_approx_simple(boundary, width, height);

    assert(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(actual[i] == expected[i] && "find_contour_chain_approx_simple mismatch!");
    }
    std::cout << "test_find_contour_chain_approx_simple passed! (" << actual.size() << " points match)" << std::endl;
}

// ______________________________________________________________________________________________

void test_enclosing_circle_approx()
{
    std::cout << "Running test_enclosing_circle_approx..." << std::endl;

    CoordinateVector<int> smooth_points;
    assert(load_contour_csv("data/test_data/contour_to_signal/smooth_contour.csv", smooth_points));

    Coordinate<float> expected_center;
    float expected_radius;
    assert(load_circle_txt("data/test_data/contour_to_signal/enclosing_circle_approx.txt", expected_center, expected_radius));

    auto [center, radius] = enclosing_circle_approx(smooth_points);

    assert(std::abs(center.a - expected_center.a) < 1e-3f && "Center X mismatch!");
    assert(std::abs(center.b - expected_center.b) < 1e-3f && "Center Y mismatch!");
    assert(std::abs(radius - expected_radius) < 1e-3f && "Radius mismatch!");

    std::cout << "test_enclosing_circle_approx passed!" << std::endl;
}

// ______________________________________________________________________________________________

void test_radial_signal()
{
    std::cout << "Running test_radial_signal..." << std::endl;

    CoordinateVector<int> smooth_points;
    assert(load_contour_csv("data/test_data/contour_to_signal/smooth_contour.csv", smooth_points));

    Coordinate<float> center;
    float radius;
    assert(load_circle_txt("data/test_data/contour_to_signal/enclosing_circle_approx.txt", center, radius));

    Signal expected;
    assert(load_signal_csv("data/test_data/contour_to_signal/radial_signal.csv", expected));

    Signal actual = radial_signal(smooth_points, center);

    assert(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(std::abs(actual[i] - expected[i]) < 1e-3f && "Radial signal value mismatch!");
    }
    std::cout << "test_radial_signal passed! (" << actual.size() << " floats match)" << std::endl;
}

// ______________________________________________________________________________________________

int main()
{
    test_trace_contour();
    test_simplify_chain_approx();
    test_find_contour_chain_approx_simple();
    test_enclosing_circle_approx();
    test_radial_signal();

    std::cout << "All contour_to_signal tests passed!" << std::endl;
    return 0;
}
