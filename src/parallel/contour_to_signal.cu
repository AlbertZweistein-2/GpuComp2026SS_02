// Coordinates -> struct of Arrays statt vector of structs

#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/copy.h>

#include "types.hpp"
#include "parallel/contour_to_signal.cuh"

// Moore neighbor tracing algorithm to get the vector of contour coordinates
// Inherently sequential, therefore not parallelizeable
// Gemini recommends not to parallelize the contour tracing, very complex for little speedup
CoordinateVector<int> trace_contour(const std::vector<uint8_t> &boundary, int width, int height)
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
        return CoordinateVector<int>();
    }

    // the actual contour vector
    CoordinateVector<int> contour;
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
// Also inherently sequential, therefore not parallelizeable
// Gemini also recommends not to parallelize the contour simplification, very complex for little speedup
CoordinateVector<int> simplify_chain_approx(const CoordinateVector<int> &contour)
{
    if (contour.size() < 3)
    {
        return contour;
    }

    CoordinateVector<int> simplified;
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

struct swap_and_b
{
    __host__ __device__ Coordinate<int> operator()(const Coordinate<int> &c) const
    {
        return Coordinate<int>{c.b, c.a};
    }
};

// ______________________________________________________________________________________________

// Just a function to first call the two previous functions
// for contour tracing and simplification
// and then swaps the coordinates from (y, x) back to (x, y) just like in the original python code
CoordinateVector<int> find_contour_chain_approx_simple(const std::vector<uint8_t> &boundary, int width, int height)
{
    // just calling the two previous functions
    CoordinateVector<int> contour = trace_contour(boundary, width, height);
    contour = simplify_chain_approx(contour);

    // swapping coordinates from (y, x) to (x, y)
    // just like in the original python code

    // finally something to parallelize
    // we need to check if the speedup from parallelizing exceeds the overhead
    thrust::device_vector<Coordinate<int>> d_contour = contour;
    thrust::transform(d_contour.begin(), d_contour.end(), d_contour.begin(), swap_and_b());
    thrust::copy(d_contour.begin(), d_contour.end(), contour.begin());

    return contour;
}

// ______________________________________________________________________________________________

// custom CUDA kernel for moving average smoothing
// one thread per contour point, each thread computes the average of its window independently
__global__ void moving_average_kernel(const Coordinate<int> *points, Coordinate<int> *smoothed, int half_window, int n)
{
    // getting the global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // we only need as many threads as there are contour points
    if (i < n)
    {
        // keeping separate sums for the x and y coordinates
        float sum_a = 0.0f;
        float sum_b = 0.0f;
        // iterating over the window within the vector
        // window_j is the index within the window
        // the center of the window is at offset 0

        // for large window sizes, it would make sense
        // to also parallelize this loop
        // so one thread per window element
        // but would require reduction using shared memory
        // does not pay off for our small windows between 5 and 15
        for (int window_j = -half_window; window_j <= half_window; ++window_j)
        {
            // window_i is the index within the vector mapped back to the whole vector
            int window_i = (i + window_j + n) % n;
            sum_a += static_cast<float>(points[window_i].a);
            sum_b += static_cast<float>(points[window_i].b);
        }

        // calculating the average
        const float divisor = static_cast<float>(2 * half_window + 1);
        smoothed[i] = Coordinate<int>{
            static_cast<int>(roundf(sum_a / divisor)),
            static_cast<int>(roundf(sum_b / divisor))};
    }
}

// moving average smoothing of the contour points vector
CoordinateVector<int> smooth_contour(const CoordinateVector<int> &points, int window)
{
    // in case the contour is empty
    // or the window size is too small or too large
    const int n = static_cast<int>(points.size());
    if (points.empty() || window <= 1 || window >= n)
    {
        return points;
    }

    // the smoothed contour vector will be the same size as the original
    int half_window = window / 2;

    // using thrust to create device vectors
    // so we do not need to manually free the memory afterwards, thrust takes care of that
    thrust::device_vector<Coordinate<int>> d_points(points.begin(), points.end());
    thrust::device_vector<Coordinate<int>> d_smoothed(n);

    // the many cores lectures taught us that this is a good choice
    // of threads per block and number of blocks for most GPUs
    int threads_per_block = 256;
    int num_blocks = 256;
    // calling the kernel
    moving_average_kernel<<<num_blocks, threads_per_block>>>(
        thrust::raw_pointer_cast(d_points.data()),
        thrust::raw_pointer_cast(d_smoothed.data()),
        half_window, n);

    // copying the result back from device to host
    CoordinateVector<int> smoothed(n);
    thrust::copy(d_smoothed.begin(), d_smoothed.end(), smoothed.begin());

    return smoothed;
}

// ______________________________________________________________________________________________

// Some thrust functors
// as far as I understand it, these are like the custom algorithmic parts of kernels
// which get translated into true kernels by thrust
// So every thread executes the same functor, but on different data

// Thrust functor for comparing the x and y coordinates
// needed for finding the min and max x and y of the contour points
struct compare_x
{
    __host__ __device__ bool operator()(const Coordinate<int> &lhs, const Coordinate<int> &rhs) const
    {
        return lhs.a < rhs.a;
    }
};
struct compare_y
{
    __host__ __device__ bool operator()(const Coordinate<int> &lhs, const Coordinate<int> &rhs) const
    {
        return lhs.b < rhs.b;
    }
};

// Needed for finding the contour point with the farthest distance to the center
// essentially just calculates the square distance from the center for each point
struct square_distance
{
    float cx;
    float cy;
    __host__ __device__ float operator()(const Coordinate<int> &p) const
    {
        float dx = p.a - cx;
        float dy = p.b - cy;
        return dx * dx + dy * dy;
    }
};

// Calculates the center and radius of a circle that encloses the contour points
std::pair<Coordinate<float>, float> enclosing_circle_approx(const CoordinateVector<int> &points)
{
    if (points.empty())
    {
        return std::make_pair(Coordinate<float>{0.0f, 0.0f}, 0.0f);
    }

    // copying the contour points to the device
    thrust::device_vector<Coordinate<int>> d_points(points.begin(), points.end());

    // finding the min and max x and y coordinates of the contour points in parallel
    auto minmax_x = thrust::minmax_element(d_points.begin(), d_points.end(), compare_x());
    auto minmax_y = thrust::minmax_element(d_points.begin(), d_points.end(), compare_y());

    // copying back from device to host
    Coordinate<int> cx_min = *(minmax_x.first);
    Coordinate<int> cx_max = *(minmax_x.second);
    Coordinate<int> cy_min = *(minmax_y.first);
    Coordinate<int> cy_max = *(minmax_y.second);

    // computing the center based on the min and max x and y
    float cx = (cx_min.a + cx_max.a) / 2.0f;
    float cy = (cy_min.b + cy_max.b) / 2.0f;

    // goes through all the boundary points
    // and finds which one has the max distance to the center
    float max_sq_dist = thrust::transform_reduce(
        d_points.begin(),
        d_points.end(),
        square_distance{cx, cy}, // Using the functor here
        0.0f,                    // Starting value for finding the max
        thrust::maximum<float>() // Reduction operation, we want the max
    );

    // taking the sqrt to get the actual radius from the max square distance
    // using cuda sqrtf for consistency
    float radius = sqrtf(max_sq_dist);
    return std::make_pair(Coordinate<float>{cx, cy}, radius);
}

// ______________________________________________________________________________________________

// Finally a real custom cuda kernel
__global__ void radial_signal_kernel(const Coordinate<int> *points, float *signal, int num_points, Coordinate<float> center)
{
    // getting the global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // we only need as many threads as there are contour points
    if (i < num_points)
    {
        // calculating the distance from the center for each contour point in parallel
        float dx = points[i].a - center.a;
        float dy = points[i].b - center.b;
        // using cuda sqrtf
        signal[i] = sqrtf(dx * dx + dy * dy);
    }
}

// Calculates a vector of distances from the center, one for each contour point
Signal radial_signal(const CoordinateVector<int> &points, Coordinate<float> center)
{
    Signal signal(points.size());
    // using thrust to create device vectors
    // so we do not need to manually free the memory afterwards, thrust takes care of that
    thrust::device_vector<float> d_signal(points.size());
    thrust::device_vector<Coordinate<int>> d_points(points.begin(), points.end());

    // the many cores lectures taught us that this is a good choice
    // of threads per block and number of blocks for most GPUs
    // he said ts often even better than scaling the number of blocks with the number of points
    int threads_per_block = 256;
    int num_blocks = 256;
    // calling the kernel
    radial_signal_kernel<<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(d_points.data()), thrust::raw_pointer_cast(d_signal.data()), points.size(), center);

    // copying the result back from device to host
    thrust::copy(d_signal.begin(), d_signal.end(), signal.begin());
    return signal;
}
