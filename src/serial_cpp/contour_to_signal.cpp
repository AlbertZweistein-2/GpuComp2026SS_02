#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>
#include <cstdint>
#include <set>

struct Coordinate {
    int a;
    int b;

    bool operator<(const Coordinate& c2) const {
        if (b != c2.b) return b < c2.b;
        return a < c2.a;
    }

    bool operator==(const Coordinate& c2) const {
        return a == c2.a && b == c2.b;
    }
};

struct CoordinateFloat {
    float a;
    float b;
};

std::vector<Coordinate> trace_contour(const std::vector<uint8_t>& boundary, int width, int height) {

    Coordinate start{-1, -1};
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (boundary[i * width + j] > 0) { 
                start = {i, j};
                break;
            }
        }
        if (start.a != -1) break;
    }

    if (start.a == -1){
        std::cerr << "No contour points found in the boundary image." << std::endl;
        return std::vector<Coordinate>();
    }

    std::vector<Coordinate> contour;
    contour.push_back(start);
    Coordinate current = start;
    int prev_dir = 0;
    std::set<Coordinate> visited;
    visited.insert(start);

    const int dy[8] = { 0,  1,  1,  1,  0, -1, -1, -1};
    const int dx[8] = { 1,  1,  0, -1, -1, -1,  0,  1};

    while (true) {
        bool found = false;
        for (int i = 0; i < 8; ++i) {
            int dir_idx = (prev_dir + i) % 8;

            int ny = current.a + dy[dir_idx];
            int nx = current.b + dx[dir_idx];

            if (0 <= ny && ny < height && 0 <= nx && nx < width && boundary[ny * width + nx] > 0) {
                
                Coordinate next_point = Coordinate{ny, nx};
                if (next_point == start && contour.size() > 10) {
                    return contour; 
                }

                if (visited.insert(next_point).second) { 
                    contour.push_back(next_point);
                    current = next_point;
                    prev_dir = (dir_idx + 5) % 8; 
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            break;
        }
    }

    return contour;
}   

std::vector<Coordinate> simplify_chain_approx(const std::vector<Coordinate>& contour) {
    if (contour.size() < 3) {
        return contour; 
    }

    std::vector<Coordinate> simplified;
    simplified.push_back(contour.front());

    for (size_t i = 1; i < contour.size() - 1; ++i) {
        int y0 = contour[i - 1].a;
        int x0 = contour[i - 1].b;
        int y1 = contour[i].a;
        int x1 = contour[i].b;
        int y2 = contour[i + 1].a;
        int x2 = contour[i + 1].b;

        int dx1 = std::clamp(x1 - x0, -1, 1);
        int dy1 = std::clamp(y1 - y0, -1, 1);
        int dx2 = std::clamp(x2 - x1, -1, 1);
        int dy2 = std::clamp(y2 - y1, -1, 1);

        if (dx1 != dx2 || dy1 != dy2) {
            simplified.push_back(contour[i]);
        }
    }
    simplified.push_back(contour.back());
    return simplified;
}

std::vector<Coordinate> find_contour_chain_approx_simple(const std::vector<uint8_t>& boundary, int width, int height) {
    std::vector<Coordinate> contour = trace_contour(boundary, width, height);
    contour = simplify_chain_approx(contour);

    for (size_t i = 0; i < contour.size(); ++i) {
        std::swap(contour[i].a, contour[i].b);
    }

    return contour;
}


std::pair<CoordinateFloat, float> enclosing_circle_approx(const std::vector<Coordinate>& points) {
    if (points.empty()) {
        return std::make_pair(CoordinateFloat{0.0f, 0.0f}, 0.0f);
    }

    int min_x = points[0].a;
    int max_x = points[0].a;
    int min_y = points[0].b;
    int max_y = points[0].b;

    for (size_t i = 1; i < points.size(); ++i) {
        min_x = std::min(min_x, points[i].a);
        max_x = std::max(max_x, points[i].a);
        min_y = std::min(min_y, points[i].b);
        max_y = std::max(max_y, points[i].b);
    }

    float cx = (min_x + max_x) / 2.0f;
    float cy = (min_y + max_y) / 2.0f;

    float max_sq_dist = 0.0f;
    for (size_t i = 0; i < points.size(); ++i) {
        float dx = points[i].a - cx;
        float dy = points[i].b - cy;
        max_sq_dist = std::max(max_sq_dist, dx * dx + dy * dy);
    }

    float radius = std::sqrt(max_sq_dist);
    return std::make_pair(CoordinateFloat{cx, cy}, radius);
}

std::vector<float> radial_signal(const std::vector<Coordinate>& points, CoordinateFloat center) {
    std::vector<float> signal(points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        float dx = points[i].a - center.a;
        float dy = points[i].b - center.b;
        signal[i] = std::sqrt(dx * dx + dy * dy);
    }

    return signal;
}