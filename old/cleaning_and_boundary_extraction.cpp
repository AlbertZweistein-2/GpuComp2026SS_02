
#include "cleaning_and_boundary_extraction.hpp"
#include <cstdint>
#include <vector>
#include <algorithm>
#include <queue>
#include <utility>
#include <iostream>


ImageU8 erode(const ImageU8& image, const ImageU8& kernel, int iterations) {
    ImageU8 result = image;

    const int height = image.size();
    const int width = image[0].size();

    const int kh = kernel.size();
    const int kw = kernel[0].size();

    const int pad_h = kh / 2;
    const int pad_w = kw / 2;

    for (int iter = 0; iter < iterations; ++iter) {
        ImageU8 new_image(height, std::vector<uint8_t>(width, 0));

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {

                bool all_foreground = true;

                for (int ky = 0; ky < kh; ++ky) {
                    for (int kx = 0; kx < kw; ++kx) {

                        if (kernel[ky][kx] != 1) {
                            continue;
                        }

                        int iy = y + ky - pad_h;
                        int ix = x + kx - pad_w;

                        uint8_t pixel = 0;

                        if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                            pixel = result[iy][ix];
                        }

                        if (pixel != 255) {
                            all_foreground = false;
                            break;
                        }
                    }

                    if (!all_foreground) {
                        break;
                    }
                }

                if (all_foreground) {
                    new_image[y][x] = 255;
                }
            }
        }

        result = new_image;
    }

    return result;
}

ImageU8 dilate(const ImageU8& image, const ImageU8& kernel, int iterations) {
    ImageU8 result = image;

    const int height = image.size();
    const int width = image[0].size();

    const int kh = kernel.size();
    const int kw = kernel[0].size();

    const int pad_h = kh / 2;
    const int pad_w = kw / 2;

    for (int iter = 0; iter < iterations; ++iter) {
        ImageU8 new_image(height, std::vector<uint8_t>(width, 0));

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {

                bool any_foreground = false;

                for (int ky = 0; ky < kh; ++ky) {
                    for (int kx = 0; kx < kw; ++kx) {

                        if (kernel[ky][kx] != 1) {
                            continue;
                        }

                        int iy = y + ky - pad_h;
                        int ix = x + kx - pad_w;

                        if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                            if (result[iy][ix] == 255) {
                                any_foreground = true;
                                break;
                            }
                        }
                    }

                    if (any_foreground) {
                        break;
                    }
                }

                if (any_foreground) {
                    new_image[y][x] = 255;
                }
            }
        }

        result = new_image;
    }

    return result;
}

ImageU8 morphological_open(const ImageU8& image, const ImageU8& kernel, int iterations) {
    ImageU8 eroded = erode(image, kernel, iterations);
    ImageU8 opened = dilate(eroded, kernel, iterations);
    return opened;
}


ImageU8 get_external_boundary_mask(const ImageU8& input) {
    const int height = input.size();
    const int width = input[0].size();

    ImageU8 binary(height, std::vector<uint8_t>(width, 0));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            binary[y][x] = input[y][x] > 0 ? 1 : 0;
        }
    }

    std::vector<std::vector<bool>> external_bg(
        height,
        std::vector<bool>(width, false)
    );

    std::queue<std::pair<int, int>> q;

    for (int x = 0; x < width; ++x) {
        if (binary[0][x] == 0) q.push({0, x});
        if (binary[height - 1][x] == 0) q.push({height - 1, x});
    }

    for (int y = 0; y < height; ++y) {
        if (binary[y][0] == 0) q.push({y, 0});
        if (binary[y][width - 1] == 0) q.push({y, width - 1});
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

        if (external_bg[y][x]) {
            continue;
        }

        if (binary[y][x] != 0) {
            continue;
        }

        external_bg[y][x] = true;

        for (const auto& [dy, dx] : neighbors4) {
            q.push({y + dy, x + dx});
        }
    }

    ImageU8 boundary(height, std::vector<uint8_t>(width, 0));

    const std::vector<std::pair<int, int>> neighbors8 = {
        {-1, -1}, {-1, 0}, {-1, 1},
        { 0, -1},          { 0, 1},
        { 1, -1}, { 1, 0}, { 1, 1}
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (binary[y][x] == 0) {
                continue;
            }

            for (const auto& [dy, dx] : neighbors8) {
                int ny = y + dy;
                int nx = x + dx;

                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                    if (external_bg[ny][nx]) {
                        boundary[y][x] = 255;
                        break;
                    }
                }
            }
        }
    }

    return boundary;
}

std::pair<ImageI32, std::vector<Region>>
connected_components(const ImageU8& binary, int min_area) {
    const int height = binary.size();
    const int width = binary[0].size();

    ImageI32 labels(height, std::vector<int>(width, 0));
    std::vector<Region> regions;

    const std::vector<std::pair<int, int>> neighbors8 = {
        {-1, -1}, {-1, 0}, {-1, 1},
        { 0, -1},          { 0, 1},
        { 1, -1}, { 1, 0}, { 1, 1}
    };

    int current_label = 1;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            if (binary[y][x] == 0 || labels[y][x] != 0) {
                continue;
            }

            std::queue<std::pair<int, int>> q;
            q.push({y, x});
            labels[y][x] = current_label;

            int area = 0;
            int min_x = x;
            int max_x = x;
            int min_y = y;
            int max_y = y;

            while (!q.empty()) {
                auto [cy, cx] = q.front();
                q.pop();

                area++;

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

                    if (binary[ny][nx] == 0 || labels[ny][nx] != 0) {
                        continue;
                    }

                    labels[ny][nx] = current_label;
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

                current_label++;
            } else {
                for (int yy = min_y; yy <= max_y; ++yy) {
                    for (int xx = min_x; xx <= max_x; ++xx) {
                        if (labels[yy][xx] == current_label) {
                            labels[yy][xx] = 0;
                        }
                    }
                }
            }
        }
    }

    return {labels, regions};
}

ImageU8 extract_piece_mask(const ImageI32& labels, int target_label) {
    const int height = labels.size();
    const int width = labels[0].size();

    ImageU8 mask(height, std::vector<uint8_t>(width, 0));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (labels[y][x] == target_label) {
                mask[y][x] = 255;
            }
        }
    }

    return mask;
}
