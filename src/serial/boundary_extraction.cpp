#include <algorithm>
#include <vector>

#include "serial/boundary_extraction.hpp"

// Naive serial boundary extraction for one piece. The CUDA path batches the
// same per-region operation, but this baseline keeps the direct nested loops.
void get_piece_boundary_mask_from_labels(
    const ImageI32& labels,
    const Region& region,
    ImageU8& boundary,
    Coordinate<int>& offset
) {
    const int image_width = labels.width;
    const int image_height = labels.height;

    // Expand the crop by one pixel where possible so pixels on the component
    // edge can see neighboring background or another component.
    const int x0 = std::max(0, region.x - 1);
    const int y0 = std::max(0, region.y - 1);
    const int x1 = std::min(image_width, region.x + region.width + 1);
    const int y1 = std::min(image_height, region.y + region.height + 1);

    const int width = x1 - x0;
    const int height = y1 - y0;

    // Later contour tracing works in local mask coordinates. The caller uses
    // this offset to shift traced contour points back to full-image space.
    offset = Coordinate<int>{x0, y0};

    boundary.width = width;
    boundary.height = height;
    boundary.data.assign(static_cast<size_t>(width) * height, 0);

    // Only scan the original component bounding box. The padded border remains
    // available in the output mask but is never itself marked as boundary.
    const int scan_x0 = std::max(0, region.x);
    const int scan_y0 = std::max(0, region.y);
    const int scan_x1 = std::min(image_width, region.x + region.width);
    const int scan_y1 = std::min(image_height, region.y + region.height);

    for (int y = scan_y0; y < scan_y1; ++y) {
        for (int x = scan_x0; x < scan_x1; ++x) {
            if (labels.data[y * image_width + x] != region.label) {
                continue;
            }

            bool is_boundary = false;
            // A component pixel is boundary if any of its 8 neighbors is
            // outside the image or does not have the same component label.
            for (int dy = -1; dy <= 1 && !is_boundary; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dy == 0 && dx == 0) {
                        continue;
                    }

                    const int ny = y + dy;
                    const int nx = x + dx;
                    if (ny < 0 || ny >= image_height || nx < 0 || nx >= image_width ||
                        labels.data[ny * image_width + nx] != region.label) {
                        is_boundary = true;
                        break;
                    }
                }
            }

            if (is_boundary) {
                // Store in the cropped mask using local coordinates.
                boundary.data[(y - y0) * width + (x - x0)] = 255;
            }
        }
    }
}
