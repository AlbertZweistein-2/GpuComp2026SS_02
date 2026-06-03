#pragma once

// #include <cstdint>
// #include <utility>
// #include <vector>
#include "types.hpp"

void get_external_boundary_mask(
    const ImageU8& input,
    ImageU8 &boundary
);