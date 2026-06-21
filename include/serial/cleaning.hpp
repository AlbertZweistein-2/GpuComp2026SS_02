#pragma once

#include "types.hpp"

// Serial morphological opening: erosion followed by dilation.
// Used after thresholding to remove small foreground noise from the binary mask.
void morphological_open(
    const ImageU8& image,
    ImageU8 &result
);
