#pragma once

#include <cstdint>
// #include <utility>
#include <vector>
#include "types.hpp"

void morphological_open(
    const ImageU8& image,
    const ImageU8& kernel,
    ImageU8 &result,
    int iterations = 1
);
