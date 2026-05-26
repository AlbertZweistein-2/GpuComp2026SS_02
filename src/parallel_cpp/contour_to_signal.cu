

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <utility>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/remove.h>

struct Coordinate
{
    int a;
    int b;

    __host__ __device__ bool operator<(const Coordinate &c2) const
    {
        if (b != c2.b)
            return b < c2.b;
        return a < c2.a;
    }

    __host__ __device__ bool operator==(const Coordinate &c2) const
    {
        return a == c2.a && b == c2.b;
    }
};

struct CoordinateFloat
{
    float a;
    float b;
};

#include <set>
#include <algorithm>

std::vector<Coordinate> trace_contour(const std::vector<uint8_t> &boundary, int width, int height)
{

    Coordinate start{-1, -1};
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
        return std::vector<Coordinate>();
    }

    std::vector<Coordinate> contour;
    contour.push_back(start);
    Coordinate current = start;
    int prev_dir = 0;
    std::set<Coordinate> visited;
    visited.insert(start);

    const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};

    while (true)
    {
        bool found = false;
        for (int i = 0; i < 8; ++i)
        {
            int dir_idx = (prev_dir + i) % 8;

            int ny = current.a + dy[dir_idx];
            int nx = current.b + dx[dir_idx];

            if (0 <= ny && ny < height && 0 <= nx && nx < width && boundary[ny * width + nx] > 0)
            {

                Coordinate next_point = Coordinate{ny, nx};
                if (next_point == start && contour.size() > 10)
                {
                    return contour;
                }

                if (visited.insert(next_point).second)
                {
                    contour.push_back(next_point);
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

    return contour;
}

std::vector<Coordinate> simplify_chain_approx(const std::vector<Coordinate> &contour)
{
    if (contour.size() < 3)
    {
        return contour;
    }

    std::vector<Coordinate> simplified;
    simplified.push_back(contour.front());

    for (size_t i = 1; i < contour.size() - 1; ++i)
    {
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

        if (dx1 != dx2 || dy1 != dy2)
        {
            simplified.push_back(contour[i]);
        }
    }
    simplified.push_back(contour.back());
    return simplified;
}

std::vector<Coordinate> find_contour_chain_approx_simple(const std::vector<uint8_t> &boundary, int width, int height)
{
    std::vector<Coordinate> contour = trace_contour(boundary, width, height);
    contour = simplify_chain_approx(contour);

    for (size_t i = 0; i < contour.size(); ++i)
    {
        std::swap(contour[i].a, contour[i].b);
    }

    return contour;
}

// Thrust functors for computing min and max
struct compare_x
{
    __host__ __device__ bool operator()(const Coordinate &lhs, const Coordinate &rhs) const
    {
        return lhs.a < rhs.a;
    }
};
struct compare_y
{
    __host__ __device__ bool operator()(const Coordinate &lhs, const Coordinate &rhs) const
    {
        return lhs.b < rhs.b;
    }
};
// Thrust functor for the square distance transformation
struct square_distance
{
    float cx, cy;

    square_distance(float _cx, float _cy) : cx(_cx), cy(_cy) {}

    __host__ __device__ float operator()(const Coordinate &p) const
    {
        float dx = p.a - cx;
        float dy = p.b - cy;
        return dx * dx + dy * dy;
    }
};

std::pair<CoordinateFloat, float> enclosing_circle_approx(const std::vector<Coordinate> &points)
{
    if (points.empty())
    {
        return std::make_pair(CoordinateFloat{0.0f, 0.0f}, 0.0f);
    }

    int min_x = points[0].a;
    int max_x = points[0].a;
    int min_y = points[0].b;
    int max_y = points[0].b;

    thrust::device_vector<Coordinate> d_points(points.begin(), points.end());

    auto minmax_x = thrust::minmax_element(d_points.begin(), d_points.end(), compare_x());
    auto minmax_y = thrust::minmax_element(d_points.begin(), d_points.end(), compare_y());

    Coordinate cx_min = *(minmax_x.first);
    Coordinate cx_max = *(minmax_x.second);
    Coordinate cy_min = *(minmax_y.first);
    Coordinate cy_max = *(minmax_y.second);

    min_x = cx_min.a;
    max_x = cx_max.a;
    min_y = cy_min.b;
    max_y = cy_max.b;

    float cx = (min_x + max_x) / 2.0f;
    float cy = (min_y + max_y) / 2.0f;

    float max_sq_dist = thrust::transform_reduce(
        d_points.begin(),
        d_points.end(),
        square_distance(cx, cy), // Transform functor
        0.0f,                    // Starting value
        thrust::maximum<float>() // Reduction operation
    );

    float radius = std::sqrt(max_sq_dist);
    return std::make_pair(CoordinateFloat{cx, cy}, radius);
}

__global__ void radial_signal_kernel(const Coordinate *points, float *signal, int num_points, CoordinateFloat center)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_points)
    {
        float dx = points[i].a - center.a;
        float dy = points[i].b - center.b;
        signal[i] = sqrtf(dx * dx + dy * dy);
    }
}

std::vector<float> radial_signal(const std::vector<Coordinate> &points, CoordinateFloat center)
{
    std::vector<float> signal(points.size());
    thrust::device_vector<float> d_signal(points.size());
    thrust::device_vector<Coordinate> d_points(points.begin(), points.end());

    int threads_per_block = 256;
    int num_blocks = 256;
    radial_signal_kernel<<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(d_points.data()), thrust::raw_pointer_cast(d_signal.data()), points.size(), center);

    thrust::copy(d_signal.begin(), d_signal.end(), signal.begin());
    return signal;
}
