#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <cstdint>
#include <cmath>
#include "cleaning_and_boundary_extraction.hpp"

struct Coordinate {
    int a;
    int b;
    bool operator==(const Coordinate& c2) const {
        return a == c2.a && b == c2.b;
    }
};

struct CoordinateFloat {
    float a;
    float b;
};

// Forward declarations step 1
std::vector<Coordinate> trace_contour(const std::vector<uint8_t>& boundary, int width, int height);
std::vector<Coordinate> simplify_chain_approx(const std::vector<Coordinate>& contour);
std::vector<Coordinate> find_contour_chain_approx_simple(const std::vector<uint8_t>& boundary, int width, int height);
std::pair<CoordinateFloat, float> enclosing_circle_approx(const std::vector<Coordinate>& points);
std::vector<float> radial_signal(const std::vector<Coordinate>& points, CoordinateFloat center);

bool load_boundary_csv(const std::string& filepath, std::vector<uint8_t>& boundary, int& width, int& height) {
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

bool load_contour_csv(const std::string& filepath, std::vector<Coordinate>& expected_contour) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    expected_contour.clear();
    std::string line, val1, val2;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        if (std::getline(ss, val1, ',') && std::getline(ss, val2, ',')) {
            expected_contour.push_back({std::stoi(val1), std::stoi(val2)});
        }
    }
    return true;
}

bool load_enclosing_circle_txt(const std::string& filepath, CoordinateFloat& center, float& radius) {
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

bool load_signal_csv(const std::string& filepath, std::vector<float>& signal) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    signal.clear();
    std::string line;
    while (std::getline(file, line)) {
        signal.push_back(std::stof(line));
    }
    return true;
}

void test_trace_contour() {
    std::cout << "Running test_trace_contour..." << std::endl;
    std::vector<uint8_t> boundary;
    int width, height;
    assert(load_boundary_csv("../../data/test_data/contour_to_signal/boundary_input.csv", boundary, width, height));

    std::vector<Coordinate> expected;
    assert(load_contour_csv("../../data/test_data/contour_to_signal/trace_contour.csv", expected));

    std::vector<Coordinate> actual = trace_contour(boundary, width, height);

    assert(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(actual[i] == expected[i] && "trace_contour mismatch!");
    }
    std::cout << "test_trace_contour passed! (" << actual.size() << " points match)" << std::endl;
}

void test_simplify_chain_approx() {
    std::cout << "Running test_simplify_chain_approx..." << std::endl;
    std::vector<Coordinate> input_contour;
    assert(load_contour_csv("../../data/test_data/contour_to_signal/trace_contour.csv", input_contour));

    std::vector<Coordinate> expected;
    assert(load_contour_csv("../../data/test_data/contour_to_signal/simplify_chain_approx.csv", expected));

    std::vector<Coordinate> actual = simplify_chain_approx(input_contour);

    assert(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(actual[i] == expected[i] && "simplify_chain_approx mismatch!");
    }
    std::cout << "test_simplify_chain_approx passed! (" << actual.size() << " points match)" << std::endl;
}

void test_find_contour_chain_approx_simple() {
    std::cout << "Running test_find_contour_chain_approx_simple..." << std::endl;
    std::vector<uint8_t> boundary;
    int width, height;
    assert(load_boundary_csv("../../data/test_data/contour_to_signal/boundary_input.csv", boundary, width, height));

    std::vector<Coordinate> expected;
    assert(load_contour_csv("../../data/test_data/contour_to_signal/find_contour_chain_approx_simple.csv", expected));

    std::vector<Coordinate> actual = find_contour_chain_approx_simple(boundary, width, height);

    assert(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(actual[i] == expected[i] && "find_contour_chain_approx_simple mismatch!");
    }
    std::cout << "test_find_contour_chain_approx_simple passed! (" << actual.size() << " points match)" << std::endl;
}

void test_enclosing_circle_approx() {
    std::cout << "Running test_enclosing_circle_approx..." << std::endl;
    std::vector<Coordinate> smooth_points;
    assert(load_contour_csv("../../data/test_data/contour_to_signal/smooth_contour.csv", smooth_points));

    CoordinateFloat expected_center;
    float expected_radius;
    assert(load_enclosing_circle_txt("../../data/test_data/contour_to_signal/enclosing_circle_approx.txt", expected_center, expected_radius));

    std::pair<CoordinateFloat, float> actual = enclosing_circle_approx(smooth_points);

    assert(std::abs(actual.first.a - expected_center.a) < 1e-3 && "Center X mismatch!");
    assert(std::abs(actual.first.b - expected_center.b) < 1e-3 && "Center Y mismatch!");
    assert(std::abs(actual.second - expected_radius) < 1e-3 && "Radius mismatch!");
    
    std::cout << "test_enclosing_circle_approx passed!" << std::endl;
}

void test_radial_signal() {
    std::cout << "Running test_radial_signal..." << std::endl;
    std::vector<Coordinate> smooth_points;
    assert(load_contour_csv("../../data/test_data/contour_to_signal/smooth_contour.csv", smooth_points));

    CoordinateFloat center;
    float radius;
    assert(load_enclosing_circle_txt("../../data/test_data/contour_to_signal/enclosing_circle_approx.txt", center, radius));

    std::vector<float> expected_signal;
    assert(load_signal_csv("../../data/test_data/contour_to_signal/radial_signal.csv", expected_signal));

    std::vector<float> actual_signal = radial_signal(smooth_points, center);

    assert(actual_signal.size() == expected_signal.size());
    for (size_t i = 0; i < actual_signal.size(); ++i) {
        assert(std::abs(actual_signal[i] - expected_signal[i]) < 1e-3 && "Radial signal value mismatch!");
    }

    std::cout << "test_radial_signal passed! (" << actual_signal.size() << " floats match)" << std::endl;
}

bool same_image_u8(const ImageU8& a, const ImageU8& b) {
    return a == b;
}

void test_erode() {
    std::cout << "Running test_erode..." << std::endl;

    ImageU8 image = {
        {0,   0,   0,   0,   0,   0},
        {0, 255, 255, 255, 255, 0},
        {0, 255, 255, 255, 255, 0},
        {0, 255, 255, 255, 255, 0},
        {0, 255, 255, 255, 255, 0},
        {0,   0,   0,   0,   0,   0}
    };

    ImageU8 kernel = {
        {1,1,1},
        {1,1,1},
        {1,1,1}
    };

    ImageU8 expected = {
        {0, 0,   0,   0,   0, 0},
        {0, 0,   0,   0,   0, 0},
        {0, 0, 255, 255,   0, 0},
        {0, 0, 255, 255,   0, 0},
        {0, 0,   0,   0,   0, 0},
        {0, 0,   0,   0,   0, 0}
    };

    assert(same_image_u8(erode(image, kernel, 1), expected));
    std::cout << "test_erode passed!" << std::endl;
}

void test_dilate() {
    std::cout << "Running test_dilate..." << std::endl;

    ImageU8 image = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 255, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };

    ImageU8 kernel = {
        {1,1,1},
        {1,1,1},
        {1,1,1}
    };

    ImageU8 expected = {
        {0,   0,   0,   0, 0},
        {0, 255, 255, 255, 0},
        {0, 255, 255, 255, 0},
        {0, 255, 255, 255, 0},
        {0,   0,   0,   0, 0}
    };

    assert(same_image_u8(dilate(image, kernel, 1), expected));
    std::cout << "test_dilate passed!" << std::endl;
}

void test_morphological_open() {
    std::cout << "Running test_morphological_open..." << std::endl;

    ImageU8 image = {
        {0,   0,   0,   0,   0, 0},
        {0, 255, 255, 255, 255, 0},
        {0, 255, 255, 255, 255, 0},
        {0, 255, 255, 255, 255, 0},
        {0, 255, 255, 255, 255, 0},
        {0,   0,   0,   0,   0, 0}
    };

    ImageU8 kernel = {
        {1,1,1},
        {1,1,1},
        {1,1,1}
    };

    assert(same_image_u8(morphological_open(image, kernel, 1), image));
    std::cout << "test_morphological_open passed!" << std::endl;
}

void test_get_external_boundary_mask() {
    std::cout << "Running test_get_external_boundary_mask..." << std::endl;

    ImageU8 image = {
        {0,   0,   0,   0,   0, 0},
        {0, 255, 255, 255, 255, 0},
        {0, 255, 255, 255, 255, 0},
        {0, 255, 255, 255, 255, 0},
        {0, 255, 255, 255, 255, 0},
        {0,   0,   0,   0,   0, 0}
    };

    ImageU8 expected = {
        {0,   0,   0,   0,   0, 0},
        {0, 255, 255, 255, 255, 0},
        {0, 255,   0,   0, 255, 0},
        {0, 255,   0,   0, 255, 0},
        {0, 255, 255, 255, 255, 0},
        {0,   0,   0,   0,   0, 0}
    };

    assert(same_image_u8(get_external_boundary_mask(image), expected));
    std::cout << "test_get_external_boundary_mask passed!" << std::endl;
}

void test_connected_components() {
    std::cout << "Running test_connected_components..." << std::endl;

    ImageU8 image = {
        {0,   0,   0,   0,   0, 0},
        {0, 255, 255,   0,   0, 0},
        {0, 255, 255,   0, 255, 0},
        {0,   0,   0,   0, 255, 0},
        {0, 255, 255,   0,   0, 0},
        {0, 255, 255,   0,   0, 0}
    };

    auto [labels, regions] = connected_components(image, 3);

    ImageI32 expected_labels = {
        {0, 0, 0, 0, 0, 0},
        {0, 1, 1, 0, 0, 0},
        {0, 1, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0},
        {0, 2, 2, 0, 0, 0},
        {0, 2, 2, 0, 0, 0}
    };

    assert(labels == expected_labels);
    assert(regions.size() == 2);

    assert(regions[0].label == 1);
    assert(regions[0].area == 4);
    assert(regions[0].x == 1);
    assert(regions[0].y == 1);
    assert(regions[0].width == 2);
    assert(regions[0].height == 2);

    assert(regions[1].label == 2);
    assert(regions[1].area == 4);
    assert(regions[1].x == 1);
    assert(regions[1].y == 4);
    assert(regions[1].width == 2);
    assert(regions[1].height == 2);

    std::cout << "test_connected_components passed!" << std::endl;
}

void test_extract_piece_mask() {
    std::cout << "Running test_extract_piece_mask..." << std::endl;

    ImageI32 labels = {
        {0, 0, 0, 0, 0},
        {0, 1, 1, 0, 2},
        {0, 1, 1, 0, 2},
        {0, 0, 0, 3, 3}
    };

    ImageU8 expected = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 255},
        {0, 0, 0, 0, 255},
        {0, 0, 0, 0, 0}
    };

    assert(same_image_u8(extract_piece_mask(labels, 2), expected));
    std::cout << "test_extract_piece_mask passed!" << std::endl;
}



int main() {
    test_trace_contour();
    test_simplify_chain_approx();
    test_find_contour_chain_approx_simple();
    test_enclosing_circle_approx();
    test_radial_signal();

    test_erode();
    test_dilate();
    test_morphological_open();
    test_get_external_boundary_mask();
    test_connected_components();
    test_extract_piece_mask();
    std::cout << "All tests passed successfully!" << std::endl;
    return 0;
}
