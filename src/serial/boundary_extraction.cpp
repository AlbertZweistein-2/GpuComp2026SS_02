#include <algorithm>
#include <vector>

#include "serial/boundary_extraction.hpp"

void get_piece_boundary_mask_from_labels(
    const ImageI32& labels,
    const Region& region,
    ImageU8& boundary,
    Coordinate<int>& offset
) {
    const int image_width = labels.width;
    const int image_height = labels.height;

    const int x0 = std::max(0, region.x - 1);
    const int y0 = std::max(0, region.y - 1);
    const int x1 = std::min(image_width, region.x + region.width + 1);
    const int y1 = std::min(image_height, region.y + region.height + 1);

    const int width = x1 - x0;
    const int height = y1 - y0;

    offset = Coordinate<int>{x0, y0};

    boundary.width = width;
    boundary.height = height;
    boundary.data.assign(static_cast<size_t>(width) * height, 0);

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
                boundary.data[(y - y0) * width + (x - x0)] = 255;
            }
        }
    }
}
