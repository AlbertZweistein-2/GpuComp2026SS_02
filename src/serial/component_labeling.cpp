#include "serial/component_labeling.hpp"

#include <algorithm>
#include <queue>

// Connected Component Labeling (CCL)
//
// Scans the binary image and identifies all connected foreground regions
// using an 8-connected Breadth First Search (BFS).
//
// For each detected component:
//
// 1. Assign a unique label.
// 2. Compute its area (number of pixels).
// 3. Compute its bounding box.
// 4. Remove the component if its area is below min_area.
// 5. Store valid regions for later contour extraction.
//
// Output:
// - labels: image where each puzzle piece has a unique integer label.
// - regions: metadata for all valid puzzle pieces.

void connected_components(
    const ImageU8& binary,
    int min_area,
    ImageI32& labels,
    std::vector<Region>& regions)
{
    // store image dimensions
    const int height = binary.height;
    const int width = binary.width;

    // Create output labels, initially assigned to 0
    labels.width = width;
    labels.height = height;
    labels.data.resize(static_cast<size_t>(width) * height, 0);

    regions.clear(); // clears previous results

    // 8 connected neighbourhood around a pixel
    const std::pair<int, int> neighbors8[] = {
        {-1, -1}, {-1, 0}, {-1, 1},
        { 0, -1},          { 0, 1},
        { 1, -1}, { 1, 0}, { 1, 1}
    };

    int current_label = 1; // label counter

    // iterate per image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // skip if component already labelled or background
            if (binary.data[y * width + x] == 0 || labels.data[y * width + x] != 0) {
                continue;
            }

            // pushed new component to queue
            std::queue<std::pair<int, int>> q;
            q.push({y, x});
            labels.data[y * width + x] = current_label;

            // keeping track of region information
            int area = 0;
            int min_x = x;
            int max_x = x;
            int min_y = y;
            int max_y = y;

            // continues until all all connected pixels are visited
            while (!q.empty()) {
                // Take one pixel
                auto [cy, cx] = q.front();
                q.pop();

                // increase area for every visited pixel
                ++area;
                // leftmost pixel
                min_x = std::min(min_x, cx);
                // rightmost pixel
                max_x = std::max(max_x, cx);
                min_y = std::min(min_y, cy);
                max_y = std::max(max_y, cy);

                // Check all 8 surrounding pixels
                for (const auto& [dy, dx] : neighbors8) {
                    // compute neighbour coordinates
                    int ny = cy + dy;
                    int nx = cx + dx;
                    
                    // check if they are out of bounds or not, so ignore pixels outside image
                    if (ny < 0 || ny >= height || nx < 0 || nx >= width) {
                        continue;
                    }
                    // skip invalid neighbours
                    if (binary.data[ny * width + nx] == 0 || labels.data[ny * width + nx] != 0) {
                        continue;
                    }
                    // Add neighbour to component
                    labels.data[ny * width + nx] = current_label;
                    q.push({ny, nx});
                }
            }
            // keep only area that is large enough
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
                        // reset images back to background so small images are set to 0
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
    // get binary mask containing only one puzzle piece
    mask.width = labels.width;
    mask.height = labels.height;
    mask.data.resize(static_cast<size_t>(labels.width) * labels.height);

    for (size_t i = 0; i < labels.data.size(); ++i) {
        // output is a binary mask containing only one puzzle piece
        mask.data[i] = (labels.data[i] == target_label) ? 255 : 0;
    }
}