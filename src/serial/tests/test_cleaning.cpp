#include <cassert>
#include <iostream>
#include <vector>

#include "types.hpp"
#include "serial/cleaning.hpp"
#include "serial/boundary_extraction.hpp"
#include "serial/component_labeling.hpp"

// ______________________________________________________________________________________________

void test_erode()
{
    std::cout << "Running test_erode..." << std::endl;

    ImageU8 image;
    image.width = 6;
    image.height = 6;
    image.data = {
        0,   0,   0,   0,   0,   0,
        0, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255,   0,
        0,   0,   0,   0,   0,   0
    };

    ImageU8 kernel;
    kernel.width = 3;
    kernel.height = 3;
    kernel.data = { 1,1,1, 1,1,1, 1,1,1 };

    ImageU8 expected;
    expected.width = 6;
    expected.height = 6;
    expected.data = {
        0, 0,   0,   0,   0, 0,
        0, 0,   0,   0,   0, 0,
        0, 0, 255, 255,   0, 0,
        0, 0, 255, 255,   0, 0,
        0, 0,   0,   0,   0, 0,
        0, 0,   0,   0,   0, 0
    };

    ImageU8 result;
    morphological_open(image, kernel, result, 0); // erode only via open with 0 iterations? 
    // TODO: adjust if erode() is exposed directly in cleaning.hpp
    assert(result.data == expected.data && "test_erode failed!");
    std::cout << "test_erode passed!" << std::endl;
}

// ______________________________________________________________________________________________

void test_morphological_open()
{
    std::cout << "Running test_morphological_open..." << std::endl;

    ImageU8 image;
    image.width = 6;
    image.height = 6;
    image.data = {
        0,   0,   0,   0,   0,   0,
        0, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255,   0,
        0,   0,   0,   0,   0,   0
    };

    ImageU8 kernel;
    kernel.width = 3;
    kernel.height = 3;
    kernel.data = { 1,1,1, 1,1,1, 1,1,1 };

    ImageU8 result;
    morphological_open(image, kernel, result, 1);

    assert(result.data == image.data && "test_morphological_open failed!");
    std::cout << "test_morphological_open passed!" << std::endl;
}

// ______________________________________________________________________________________________

void test_get_external_boundary_mask()
{
    std::cout << "Running test_get_external_boundary_mask..." << std::endl;

    ImageU8 image;
    image.width = 6;
    image.height = 6;
    image.data = {
        0,   0,   0,   0,   0,   0,
        0, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255,   0,
        0,   0,   0,   0,   0,   0
    };

    ImageU8 expected;
    expected.width = 6;
    expected.height = 6;
    expected.data = {
        0,   0,   0,   0,   0,   0,
        0, 255, 255, 255, 255,   0,
        0, 255,   0,   0, 255,   0,
        0, 255,   0,   0, 255,   0,
        0, 255, 255, 255, 255,   0,
        0,   0,   0,   0,   0,   0
    };

    ImageU8 result;
    get_external_boundary_mask(image, result);

    assert(result.data == expected.data && "test_get_external_boundary_mask failed!");
    std::cout << "test_get_external_boundary_mask passed!" << std::endl;
}

// ______________________________________________________________________________________________

void test_connected_components()
{
    std::cout << "Running test_connected_components..." << std::endl;

    ImageU8 image;
    image.width = 6;
    image.height = 6;
    image.data = {
        0,   0,   0,   0,   0,   0,
        0, 255, 255,   0,   0,   0,
        0, 255, 255,   0, 255,   0,
        0,   0,   0,   0, 255,   0,
        0, 255, 255,   0,   0,   0,
        0, 255, 255,   0,   0,   0
    };

    auto [labels, regions] = connected_components(image, 3);

    ImageI32 expected_labels;
    expected_labels.width = 6;
    expected_labels.height = 6;
    expected_labels.data = {
        0, 0, 0, 0, 0, 0,
        0, 1, 1, 0, 0, 0,
        0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 2, 2, 0, 0, 0,
        0, 2, 2, 0, 0, 0
    };

    assert(labels.data == expected_labels.data && "connected_components labels mismatch!");
    assert(regions.size() == 2 && "connected_components region count mismatch!");

    assert(regions[0].label == 1);
    assert(regions[0].area  == 4);
    assert(regions[0].x     == 1);
    assert(regions[0].y     == 1);
    assert(regions[0].width == 2);
    assert(regions[0].height == 2);

    assert(regions[1].label == 2);
    assert(regions[1].area  == 4);
    assert(regions[1].x     == 1);
    assert(regions[1].y     == 4);
    assert(regions[1].width == 2);
    assert(regions[1].height == 2);

    std::cout << "test_connected_components passed!" << std::endl;
}

// ______________________________________________________________________________________________

void test_extract_piece_mask()
{
    std::cout << "Running test_extract_piece_mask..." << std::endl;

    ImageI32 labels;
    labels.width = 5;
    labels.height = 4;
    labels.data = {
        0, 0, 0, 0, 0,
        0, 1, 1, 0, 2,
        0, 1, 1, 0, 2,
        0, 0, 0, 3, 3
    };

    ImageU8 expected;
    expected.width = 5;
    expected.height = 4;
    expected.data = {
          0,   0,   0,   0,   0,
          0,   0,   0,   0, 255,
          0,   0,   0,   0, 255,
          0,   0,   0,   0,   0
    };

    ImageU8 result = extract_piece_mask(labels, 2);

    assert(result.data == expected.data && "test_extract_piece_mask failed!");
    std::cout << "test_extract_piece_mask passed!" << std::endl;
}

// ______________________________________________________________________________________________

int main()
{
    test_morphological_open();
    test_get_external_boundary_mask();
    test_connected_components();
    test_extract_piece_mask();

    std::cout << "All cleaning tests passed!" << std::endl;
    return 0;
}
