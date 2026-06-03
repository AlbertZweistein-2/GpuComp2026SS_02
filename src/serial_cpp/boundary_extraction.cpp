#include <queue>
#include "serial/boundary_extraction.hpp"

void get_external_boundary_mask(
    const ImageU8& input,
    ImageU8 &boundary
) {
    const int height = input.height;
    const int width = input.width;

    std::vector<std::vector<bool>> visited(
        height,
        std::vector<bool>(width, false)
    );

    std::vector<std::vector<bool>> external_bg(
        height,
        std::vector<bool>(width, false)
    );

    std::queue<std::pair<int, int>> q;

    for (int x = 0; x < width; ++x) {
        if (input.data[x] == 0) q.push({0, x});
        if (input.data[(height - 1) * width + x] == 0) q.push({height - 1, x});
    }

    for (int y = 0; y < height; ++y) {
        if (input.data[y * width] == 0) q.push({y, 0});
        if (input.data[y * width + width - 1] == 0) q.push({y, width - 1});
    }

    const std::vector<std::pair<int, int>> neighbors4 = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1}
    };

    while (!q.empty()) {
        auto [y, x] = q.front();
        q.pop();

        if (y < 0 || y >= height || x < 0 || x >= width) {
            continue;
        }

        if (visited[y][x]) {
            continue;
        }

        visited[y][x] = true;

        if (input.data[y * width + x] != 0) {
            continue;
        }

        external_bg[y][x] = true;

        for (const auto& [dy, dx] : neighbors4) {
            q.push({y + dy, x + dx});
        }
    }

    boundary.width = width;
    boundary.height = height;
    boundary.data.assign(static_cast<size_t>(width) * height, 0);

    const std::vector<std::pair<int, int>> neighbors8 = {
        {-1, -1}, {-1, 0}, {-1, 1},
        { 0, -1},          { 0, 1},
        { 1, -1}, { 1, 0}, { 1, 1}
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (input.data[y * width + x] == 0) {
                continue;
            }

            for (const auto& [dy, dx] : neighbors8) {
                int ny = y + dy;
                int nx = x + dx;

                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                    if (external_bg[ny][nx]) {
                        boundary.data[y * width + x] = 255;
                        break;
                    }
                }
            }
        }
    }
}