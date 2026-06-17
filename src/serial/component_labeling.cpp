#include "serial/component_labeling.hpp"

#include <algorithm>
#include <queue>

void connected_components(
    const ImageU8& binary,
    int min_area,
    ImageI32& labels,
    std::vector<Region>& regions)
{
    const int height = binary.height;
    const int width = binary.width;

    labels.width = width;
    labels.height = height;
    labels.data.resize(static_cast<size_t>(width) * height, 0);

    regions.clear();

    const std::pair<int, int> neighbors8[] = {
        {-1, -1}, {-1, 0}, {-1, 1},
        { 0, -1},          { 0, 1},
        { 1, -1}, { 1, 0}, { 1, 1}
    };

    int current_label = 1;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (binary.data[y * width + x] == 0 || labels.data[y * width + x] != 0) {
                continue;
            }

            std::queue<std::pair<int, int>> q;
            q.push({y, x});
            labels.data[y * width + x] = current_label;

            int area = 0;
            int min_x = x;
            int max_x = x;
            int min_y = y;
            int max_y = y;

            while (!q.empty()) {
                auto [cy, cx] = q.front();
                q.pop();

                ++area;
                min_x = std::min(min_x, cx);
                max_x = std::max(max_x, cx);
                min_y = std::min(min_y, cy);
                max_y = std::max(max_y, cy);

                for (const auto& [dy, dx] : neighbors8) {
                    int ny = cy + dy;
                    int nx = cx + dx;

                    if (ny < 0 || ny >= height || nx < 0 || nx >= width) {
                        continue;
                    }

                    if (binary.data[ny * width + nx] == 0 || labels.data[ny * width + nx] != 0) {
                        continue;
                    }

                    labels.data[ny * width + nx] = current_label;
                    q.push({ny, nx});
                }
            }

            if (area >= min_area) {
                regions.push_back({
                    current_label,
                    area,
                    min_x,
                    min_y,
                    max_x - min_x + 1,
                    max_y - min_y + 1
                });
                ++current_label;
            } else {
                for (int yy = min_y; yy <= max_y; ++yy) {
                    for (int xx = min_x; xx <= max_x; ++xx) {
                        if (labels.data[yy * width + xx] == current_label) {
                            labels.data[yy * width + xx] = 0;
                        }
                    }
                }
            }
        }
    }
}

void extract_piece_mask(
    const ImageI32& labels,
    int target_label,
    ImageU8& mask)
{
    // QUESTION:
    // Maybe possible to return only an image of size 
    // of the bounding box of the region, but then we would need to store the offset of the bounding box as well
    mask.width = labels.width;
    mask.height = labels.height;
    mask.data.resize(static_cast<size_t>(labels.width) * labels.height);

    for (size_t i = 0; i < labels.data.size(); ++i) {
        mask.data[i] = (labels.data[i] == target_label) ? 255 : 0;
    }
}