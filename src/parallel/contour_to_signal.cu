#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <utility>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/copy.h>

#include "types.hpp"
#include "parallel/contour_to_signal.cuh"
#include "helpers.hpp"

// Moore neighbor tracing algorithm to get the vector of contour coordinates.
// This part is intentionally sequential: every next point depends on the
// current point and the previous search direction. The CUDA pipeline still uses
// it for now because visualization and radial-signal classification need a
// connected contour order, not only a set of boundary pixels.
void trace_contour_cuda(const uint8_t *boundary, int width, int height, CoordinateVector<int> &result)
{
    // Goes from top-left to bottom-right to find the first nonzero pixel.
    // This then becomes the starting point.
    Coordinate<int> start{-1, -1};
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            if (boundary[i * width + j] > 0)
            {
                start = {i, j};
                break;
            }
        }
        if (start.a != -1)
            break;
    }

    if (start.a == -1)
    {
        std::cerr << "No contour points found in the boundary image." << std::endl;
        result.clear();
        return;
    }

    result.clear();
    result.push_back(start);

    Coordinate<int> current = start;
    int prev_dir = 0;
    std::set<Coordinate<int>> visited;
    visited.insert(start);

    const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};

    while (true)
    {
        bool found = false;
        for (int i = 0; i < 8; ++i)
        {
            const int dir_idx = (prev_dir + i) % 8;
            const int ny = current.a + dy[dir_idx];
            const int nx = current.b + dx[dir_idx];

            if (0 <= ny && ny < height && 0 <= nx && nx < width && boundary[ny * width + nx] > 0)
            {
                Coordinate<int> next_point{ny, nx};
                if (next_point == start && result.size() > 10)
                {
                    return;
                }

                if (visited.insert(next_point).second)
                {
                    result.push_back(next_point);
                    current = next_point;
                    prev_dir = (dir_idx + 5) % 8;
                    found = true;
                    break;
                }
            }
        }

        if (!found)
        {
            break;
        }
    }
}

void trace_contour_cuda(const std::vector<uint8_t> &boundary, int width, int height, CoordinateVector<int> &result)
{
    trace_contour_cuda(boundary.data(), width, height, result);
}

// Removes unnecessary points along straight lines.
// This is also sequential because it depends on adjacent contour order.
void simplify_chain_approx_cuda(CoordinateVector<int> &contour)
{
    if (contour.size() < 3)
    {
        return;
    }

    CoordinateVector<int> simplified;
    simplified.push_back(contour.front());

    for (size_t i = 1; i < contour.size() - 1; ++i)
    {
        const int y0 = contour[i - 1].a;
        const int x0 = contour[i - 1].b;
        const int y1 = contour[i].a;
        const int x1 = contour[i].b;
        const int y2 = contour[i + 1].a;
        const int x2 = contour[i + 1].b;

        const int dx1 = std::clamp(x1 - x0, -1, 1);
        const int dy1 = std::clamp(y1 - y0, -1, 1);
        const int dx2 = std::clamp(x2 - x1, -1, 1);
        const int dy2 = std::clamp(y2 - y1, -1, 1);

        if (dx1 != dx2 || dy1 != dy2)
        {
            simplified.push_back(contour[i]);
        }
    }
    simplified.push_back(contour.back());
    contour = std::move(simplified);
}

struct swap_ab
{
    __host__ __device__ Coordinate<int> operator()(const Coordinate<int> &c) const
    {
        return Coordinate<int>{c.b, c.a};
    }
};

void find_contour_chain_approx_simple_cuda(const uint8_t *boundary, int width, int height, CoordinateVector<int> &result)
{
    trace_contour_cuda(boundary, width, height, result);
    simplify_chain_approx_cuda(result);

    // The tracing code stores points as (row, column). Swap them back to
    // (x, y) like the serial implementation and the original Python code.
    thrust::device_vector<Coordinate<int>> d_contour = result;
    thrust::transform(d_contour.begin(), d_contour.end(), d_contour.begin(), swap_ab());
    thrust::copy(d_contour.begin(), d_contour.end(), result.begin());
}

void find_contour_chain_approx_simple_cuda(const std::vector<uint8_t> &boundary, int width, int height, CoordinateVector<int> &result)
{
    find_contour_chain_approx_simple_cuda(boundary.data(), width, height, result);
}

// ______________________________________________________________________________________________

// custom CUDA kernel for moving average smoothing
// one thread per contour point, each thread computes the average of its window independently
__global__ void moving_average_kernel(const Coordinate<int> *points, Coordinate<int> *smoothed, int half_window, int n)
{
    // we will load the contour points into shared memory to speed up the reads
    extern __shared__ Coordinate<int> s_points[];

    // getting the global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // getting the index within the block
    int local_i = threadIdx.x;

    // each thread loads one point into shared memory
    // wrap with % n so threads beyond the contour end do not leave shared mem uninitialized
    s_points[local_i + half_window] = points[i % n];

    // loading the left and right border points
    // this is for windows that extend beyond the border of the points assigned the the block
    if (local_i < half_window)
    {
        s_points[local_i + blockDim.x + half_window] = points[(i + blockDim.x) % n];
        s_points[local_i] = points[(i - half_window + n) % n];
    }

    // barrier to make sure all elements are loaded
    __syncthreads();

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
        // so one thread per window element and reduction using shared memory
        // But for our small windows of 5 or 7 the overhead woudl probably dominate
        for (int window_j = -half_window; window_j <= half_window; ++window_j)
        {
            // window_i is the index within the vector mapped back to the whole vector
            int window_i = local_i + half_window + window_j;
            sum_a += static_cast<float>(s_points[window_i].a);
            sum_b += static_cast<float>(s_points[window_i].b);
        }

        // calculating the average
        const float divisor = static_cast<float>(2 * half_window + 1);
        smoothed[i] = Coordinate<int>{
            static_cast<int>(roundf(sum_a / divisor)),
            static_cast<int>(roundf(sum_b / divisor))};
    }
}

// moving average smoothing of the contour points vector
void smooth_contour_cuda(CoordinateVector<int> &points, int window)
{
    // in case the contour is empty
    // or the window size is too small or too large
    const int n = static_cast<int>(points.size());
    if (points.empty() || window <= 1 || window >= n)
    {
        return;
    }

    // the smoothed contour vector will be the same size as the original
    int half_window = window / 2;

    // using thrust to create device vectors
    // so we do not need to manually free the memory afterwards, thrust takes care of that
    thrust::device_vector<Coordinate<int>> d_points(points.begin(), points.end());
    thrust::device_vector<Coordinate<int>> d_smoothed(n);

    // dynamically determining the number of blocks
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    // shared memory must also include the border points on each side
    int shared_mem_size = (threads_per_block + 2 * half_window) * sizeof(Coordinate<int>);
    // calling the kernel
    moving_average_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        thrust::raw_pointer_cast(d_points.data()),
        thrust::raw_pointer_cast(d_smoothed.data()),
        half_window, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copying the result back from device to host
    thrust::copy(d_smoothed.begin(), d_smoothed.end(), points.begin());
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
void enclosing_circle_approx_cuda(const CoordinateVector<int> &points, Coordinate<float> &center, float &radius)
{
    if (points.empty())
    {
        center = {0.0f, 0.0f};
        radius = 0.0f;
        return;
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
    center = {cx, cy};
    radius = sqrtf(max_sq_dist);
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
void radial_signal_cuda(const CoordinateVector<int> &points, Coordinate<float> center, Signal &signal)
{
    signal.resize(points.size());
    // using thrust to create device vectors
    // so we do not need to manually free the memory afterwards, thrust takes care of that
    thrust::device_vector<float> d_signal(points.size());
    thrust::device_vector<Coordinate<int>> d_points(points.begin(), points.end());

    // dynamically determining the number of blocks
    int threads_per_block = 256;
    int num_blocks = (static_cast<int>(points.size()) + threads_per_block - 1) / threads_per_block;
    // calling the kernel
    radial_signal_kernel<<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(d_points.data()), thrust::raw_pointer_cast(d_signal.data()), points.size(), center);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copying the result back from device to host
    thrust::copy(d_signal.begin(), d_signal.end(), signal.begin());
}
