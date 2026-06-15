#pragma once

#include <cstdint>
// #include <utility>
#include <vector>
#include "types.hpp"

void morphological_open(
    const ImageU8& image,
    const ImageU8& kernel,
    ImageU8 &result,
    int iterations
);

inline ImageU8 make_kernel(int width, int height)
{
    ImageU8 kernel;
    kernel.width = width;
    kernel.height = height;
    kernel.data.assign(static_cast<size_t>(width) * height, 1);
    return kernel;
}