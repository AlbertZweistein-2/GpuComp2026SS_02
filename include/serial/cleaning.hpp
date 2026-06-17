#pragma once

#include <cstddef>

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
    kernel.data.assign(static_cast<std::size_t>(width) * height, 1);
    return kernel;
}
