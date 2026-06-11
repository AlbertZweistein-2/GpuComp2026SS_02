#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>
#include <cstdint>
#include <set>

#include "types.hpp"
#include "serial/contour_to_signal.hpp"

// Now the actual functions
// ______________________________________________________________________________________________

// Moore neighbor tracing algorithm to get the vector of contour coordinates
// Inherently sequential
std::vector<Coordinate<int>> trace_contour(const std::vector<uint8_t> &boundary, int width, int height)
{

    // Goes from top-left to bottom-right to find the first nonzero pixel
    // this then becomes the starting point
    Coordinate<int> start{-1, -1};
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            // boundary is the image as a 1D vector
            if (boundary[i * width + j] > 0)
            {
                start = {i, j};
                break;
            }
        }
        if (start.a != -1)
            break;
    }

    // If not a single nonzero pixel was found
    if (start.a == -1)
    {
        std::cerr << "No contour points found in the boundary image." << std::endl;
        return std::vector<Coordinate<int>>();
    }

    // the actual contour vector
    std::vector<Coordinate<int>> contour;
    // adding the start pixel
    contour.push_back(start);

    Coordinate<int> current = start;
    int prev_dir = 0;
    // and keeping a set of unique coordinates that have been visited
    std::set<Coordinate<int>> visited;
    visited.insert(start);

    // an array of the indices corresponding to the eight neighboring pixels
    const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};

    while (true)
    {
        bool found = false;
        // checking all eight neighbors clockwise
        for (int i = 0; i < 8; ++i)
        {
            int dir_idx = (prev_dir + i) % 8;

            // nx and ny are the coordinates of the neighboring pixel we are currently checking
            int ny = current.a + dy[dir_idx];
            int nx = current.b + dx[dir_idx];

            // a lot of conditions
            // if nx and ny are within the image bounds
            // and if the pixal at (ny, nx) is nonzero
            if (0 <= ny && ny < height && 0 <= nx && nx < width && boundary[ny * width + nx] > 0)
            {

                Coordinate<int> next_point = Coordinate<int>{ny, nx};
                // if this is the starting point again
                // and we have found more than ten points
                // we converge
                if (next_point == start && contour.size() > 10)
                {
                    return contour;
                }

                // if it has not been visited before
                if (visited.insert(next_point).second)
                {
                    // we add it to the contour vector
                    contour.push_back(next_point);
                    // and start again, but now from the new point
                    current = next_point;
                    prev_dir = (dir_idx + 5) % 8;
                    // we log that we have found a new point
                    found = true;
                    break;
                }
            }
        }

        // if we have checked all eight neighbors and found no new point
        // the contour has ended
        if (!found)
        {
            break;
        }
    }

    return contour;
} 

// ______________________________________________________________________________________________

// Removes unnecessary points along straight lines, we only need the corners
// Also inherently sequential
std::vector<Coordinate<int>> simplify_chain_approx(const std::vector<Coordinate<int>> &contour)
{
    if (contour.size() < 3)
    {
        return contour;
    }

    std::vector<Coordinate<int>> simplified;
    simplified.push_back(contour.front());

    // we iterate through the contour points, starting from the second point
    for (size_t i = 1; i < contour.size() - 1; ++i)
    {
        // we use a sliding window of three points
        // the previous one
        int y0 = contour[i - 1].a;
        int x0 = contour[i - 1].b;
        // the current one
        int y1 = contour[i].a;
        int x1 = contour[i].b;
        // and the next one
        int y2 = contour[i + 1].a;
        int x2 = contour[i + 1].b;

        // we then check the direction between the
        // previous and current point
        int dx1 = std::clamp(x1 - x0, -1, 1);
        int dy1 = std::clamp(y1 - y0, -1, 1);
        // and current and next point
        int dx2 = std::clamp(x2 - x1, -1, 1);
        int dy2 = std::clamp(y2 - y1, -1, 1);

        // if the two directions are different, the contour is turning
        // so the current point is a necessary corner
        if (dx1 != dx2 || dy1 != dy2)
        {
            // and we add it to the simplified contour
            simplified.push_back(contour[i]);
        }
    }
    simplified.push_back(contour.back());
    return simplified;
}

// ______________________________________________________________________________________________

// Just a function to first call the two previous functions
// for contour tracing and simplification
// and then swaps the coordinates from (y, x) back to (x, y) just like in the original python code
std::vector<Coordinate<int>> find_contour_chain_approx_simple(const std::vector<uint8_t>& boundary, int width, int height) {

    // just calling the two previous functions
    std::vector<Coordinate<int>> contour = trace_contour(boundary, width, height);
    contour = simplify_chain_approx(contour);

    // swapping coordinates from (y, x) to (x, y)
    // just like in the original python code
    for (size_t i = 0; i < contour.size(); ++i) {
        std::swap(contour[i].a, contour[i].b);
    }

    return contour;
}

// ______________________________________________________________________________________________

// Calculates the center and radius of a circle that encloses the contour points
std::pair<Coordinate<float>, float> enclosing_circle_approx(const std::vector<Coordinate<int>>& points) {
    if (points.empty()) {
        return std::make_pair(Coordinate<float>{0.0f, 0.0f}, 0.0f);
    }

    int min_x = points[0].a;
    int max_x = points[0].a;
    int min_y = points[0].b;
    int max_y = points[0].b;

    // iterating over all contour points
    // and finding the min and max x and y coordinates
    for (size_t i = 1; i < points.size(); ++i) {
        min_x = std::min(min_x, points[i].a);
        max_x = std::max(max_x, points[i].a);
        min_y = std::min(min_y, points[i].b);
        max_y = std::max(max_y, points[i].b);
    }

    // computing the center based on the min and max x and y
    float cx = (min_x + max_x) / 2.0f;
    float cy = (min_y + max_y) / 2.0f;

    // again iterating over all contour points
    // and finding which one has the max distance to the center
    float max_sq_dist = 0.0f;
    for (size_t i = 0; i < points.size(); ++i) {
        float dx = points[i].a - cx;
        float dy = points[i].b - cy;
        max_sq_dist = std::max(max_sq_dist, dx * dx + dy * dy);
    }

    // taking the sqrt to get the actual radius from the max square distance
    float radius = std::sqrt(max_sq_dist);
    return std::make_pair(Coordinate<float>{cx, cy}, radius);
}

// ______________________________________________________________________________________________

// Calculates a vector of distances from the center, one for each contour point
Signal radial_signal(const std::vector<Coordinate<int>>& points, Coordinate<float> center) {
    Signal signal(points.size());

    // again iterating over all contour points
    // and calculating the distance from the center for each contour point
    // and storing it in the signal vector
    for (size_t i = 0; i < points.size(); ++i) {
        float dx = points[i].a - center.a;
        float dy = points[i].b - center.b;
        signal[i] = std::sqrt(dx * dx + dy * dy);
    }

    return signal;
}