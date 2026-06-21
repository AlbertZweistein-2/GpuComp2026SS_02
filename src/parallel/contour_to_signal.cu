#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <set>
#include <utility>
#include <vector>
#include <algorithm>
#include <climits>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "types.hpp"
#include "parallel/contour_to_signal.cuh"
#include "helpers.hpp"

namespace
{

    // Moore neighbor tracing algorithm to get the vector of contour coordinates.
    // This part is intentionally sequential: every next point depends on the
    // current point and the previous search direction. The CUDA pipeline still
    // uses it for now because radial-signal classification needs connected
    // contour order, not only a set of boundary pixels.
    void trace_contour(const uint8_t *boundary, int width, int height, CoordinateVector<int> &result)
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
                // Resume the Moore-neighborhood scan relative to the previous
                // direction so the trace follows the boundary instead of
                // repeatedly restarting from the same neighbor.
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

    // Removes unnecessary points along straight lines.
    // This is also sequential because it depends on adjacent contour order.
    void simplify_chain_approx(CoordinateVector<int> &contour)
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

            // Keep only points where the chain direction changes. This reduces
            // downstream GPU work without changing the contour order.
            if (dx1 != dx2 || dy1 != dy2)
            {
                simplified.push_back(contour[i]);
            }
        }
        simplified.push_back(contour.back());
        contour = std::move(simplified);
    }

} // namespace

void find_contour_chain_approx_simple_cuda(const uint8_t *boundary, int width, int height, CoordinateVector<int> &result)
{
    // Contour extraction stays on CPU because the next contour point depends on
    // the previous point. Later stages batch all extracted contours for CUDA.
    trace_contour(boundary, width, height, result);
    simplify_chain_approx(result);

    // The tracing code stores points as (row, column). Swap them back to
    // (x, y) like the serial implementation and the original Python code.
    for (Coordinate<int> &point : result)
    {
        std::swap(point.a, point.b);
    }
}

void build_batched_contours(
    const std::vector<PuzzlePiece> &pieces,
    BatchedContours &batch)
{
    // Pack all CPU-traced contours into one structure-of-arrays-like layout:
    // points holds every contour back-to-back, while offsets/lengths describe
    // each piece slice. CUDA kernels can then process all pieces in one launch
    // and avoid per-piece allocations or vector-of-vectors device structures.
    batch.points.clear();
    batch.offsets.clear();
    batch.lengths.clear();
    batch.max_length = 0;

    std::size_t total_points = 0;
    for (const PuzzlePiece &piece : pieces)
    {
        const int length = static_cast<int>(piece.contour.size());
        total_points += piece.contour.size();
        batch.max_length = std::max(batch.max_length, length);
    }

    batch.points.reserve(total_points);
    batch.offsets.reserve(pieces.size());
    batch.lengths.reserve(pieces.size());

    int offset = 0;
    for (const PuzzlePiece &piece : pieces)
    {
        const int length = static_cast<int>(piece.contour.size());
        batch.offsets.push_back(offset);
        batch.lengths.push_back(length);
        batch.points.insert(batch.points.end(), piece.contour.begin(), piece.contour.end());
        offset += length;
    }
}

// Batched moving average smoothing.
// blockIdx.y selects the piece and blockIdx.x/threadIdx.x selects the local
// point index within that piece's slice of the flat contour buffer.
__global__ void batched_moving_average_kernel(
    const Coordinate<int> *points,
    Coordinate<int> *smoothed,
    const int *offsets,
    const int *lengths,
    int half_window,
    int window,
    int max_length)
{
    const int piece_idx = blockIdx.y;
    const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_i >= max_length)
    {
        return;
    }

    const int n = lengths[piece_idx];
    if (local_i >= n)
    {
        return;
    }

    const int offset = offsets[piece_idx];
    const int global_i = offset + local_i;
    if (window <= 1 || window >= n)
    {
        // Degenerate smoothing windows keep the original contour unchanged.
        smoothed[global_i] = points[global_i];
        return;
    }

    float sum_a = 0.0f;
    float sum_b = 0.0f;
    for (int window_j = -half_window; window_j <= half_window; ++window_j)
    {
        // The contour is closed, so smoothing wraps around the slice boundary.
        int local_j = (local_i + window_j) % n;
        if (local_j < 0)
        {
            local_j += n;
        }

        const Coordinate<int> point = points[offset + local_j];
        sum_a += static_cast<float>(point.a);
        sum_b += static_cast<float>(point.b);
    }

    const float divisor = static_cast<float>(window);
    smoothed[global_i] = Coordinate<int>{
        static_cast<int>(roundf(sum_a / divisor)),
        static_cast<int>(roundf(sum_b / divisor))};
}

void smooth_contours_batched_cuda(
    const BatchedContours &batch,
    int window,
    BatchedContourCudaScratch &scratch)
{
    // Moore tracing creates host contours. Upload all of them once, then run a
    // single batched kernel instead of launching one smoothing kernel per piece.
    // The smoothed contours stay on the device for the circle and radial stages.
    scratch.points.resize(batch.points.size());
    scratch.smoothed.resize(batch.points.size());
    scratch.offsets.resize(batch.offsets.size());
    scratch.lengths.resize(batch.lengths.size());

    if (batch.points.empty() || batch.offsets.empty())
    {
        return;
    }

    thrust::copy(batch.points.begin(), batch.points.end(), scratch.points.begin());
    thrust::copy(batch.offsets.begin(), batch.offsets.end(), scratch.offsets.begin());
    thrust::copy(batch.lengths.begin(), batch.lengths.end(), scratch.lengths.begin());

    const int block = 256;
    const dim3 grid(
        static_cast<unsigned int>((batch.max_length + block - 1) / block),
        static_cast<unsigned int>(batch.offsets.size()));
    batched_moving_average_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(scratch.points.data()),
        thrust::raw_pointer_cast(scratch.smoothed.data()),
        thrust::raw_pointer_cast(scratch.offsets.data()),
        thrust::raw_pointer_cast(scratch.lengths.data()),
        window / 2,
        window,
        batch.max_length);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void batched_enclosing_circle_kernel(
    const Coordinate<int> *points,
    const int *offsets,
    const int *lengths,
    Coordinate<float> *centers)
{
    __shared__ int s_min_x[256];
    __shared__ int s_max_x[256];
    __shared__ int s_min_y[256];
    __shared__ int s_max_y[256];

    // One CUDA block owns one piece and reduces that piece's bounding box.
    // The center of this box is the same approximation used by the old path.
    const int piece_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int n = lengths[piece_idx];
    const int offset = offsets[piece_idx];

    int min_x = INT_MAX;
    int max_x = INT_MIN;
    int min_y = INT_MAX;
    int max_y = INT_MIN;
    for (int i = tid; i < n; i += blockDim.x)
    {
        // Threads scan strided points from one contour, then reduce local
        // min/max values in shared memory. This avoids global atomics.
        const Coordinate<int> point = points[offset + i];
        if (point.a < min_x)
        {
            min_x = point.a;
        }
        if (point.a > max_x)
        {
            max_x = point.a;
        }
        if (point.b < min_y)
        {
            min_y = point.b;
        }
        if (point.b > max_y)
        {
            max_y = point.b;
        }
    }

    s_min_x[tid] = min_x;
    s_max_x[tid] = max_x;
    s_min_y[tid] = min_y;
    s_max_y[tid] = max_y;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            if (s_min_x[tid + stride] < s_min_x[tid])
            {
                s_min_x[tid] = s_min_x[tid + stride];
            }
            if (s_max_x[tid + stride] > s_max_x[tid])
            {
                s_max_x[tid] = s_max_x[tid + stride];
            }
            if (s_min_y[tid + stride] < s_min_y[tid])
            {
                s_min_y[tid] = s_min_y[tid + stride];
            }
            if (s_max_y[tid + stride] > s_max_y[tid])
            {
                s_max_y[tid] = s_max_y[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        if (n <= 0)
        {
            centers[piece_idx] = Coordinate<float>{0.0f, 0.0f};
        }
        else
        {
            // The serial approximation uses the center of the contour bounding
            // box, not a true minimum enclosing circle.
            const float center_x = (static_cast<float>(s_min_x[0]) + static_cast<float>(s_max_x[0])) * 0.5f;
            const float center_y = (static_cast<float>(s_min_y[0]) + static_cast<float>(s_max_y[0])) * 0.5f;
            centers[piece_idx] = Coordinate<float>{center_x, center_y};
        }
    }
}

void enclosing_circle_centers_batched_cuda(
    const BatchedContours &batch,
    BatchedContourCudaScratch &scratch)
{
    const std::size_t piece_count = batch.offsets.size();
    scratch.centers.resize(piece_count);
    if (piece_count == 0)
    {
        return;
    }
    if (scratch.smoothed.empty())
    {
        return;
    }

    // One block reduces one piece contour. The pipeline only needs the center,
    // so there is no radius reduction or host-side center copy in this path.
    const int block = 256;
    batched_enclosing_circle_kernel<<<static_cast<unsigned int>(piece_count), block>>>(
        thrust::raw_pointer_cast(scratch.smoothed.data()),
        thrust::raw_pointer_cast(scratch.offsets.data()),
        thrust::raw_pointer_cast(scratch.lengths.data()),
        thrust::raw_pointer_cast(scratch.centers.data()));
    CUDA_CHECK(cudaGetLastError());
}

__global__ void batched_radial_signal_kernel(
    const Coordinate<int> *points,
    const int *offsets,
    const int *lengths,
    const Coordinate<float> *centers,
    float *signals,
    int max_length)
{
    // blockIdx.y selects the piece; x-dimension threads cover local contour
    // indices. The output uses the same flat offsets as the contour buffer.
    const int piece_idx = blockIdx.y;
    const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_i >= max_length)
    {
        return;
    }

    const int n = lengths[piece_idx];
    if (local_i >= n)
    {
        return;
    }

    const int global_i = offsets[piece_idx] + local_i;
    const Coordinate<int> point = points[global_i];
    const Coordinate<float> center = centers[piece_idx];
    // Radial distance is the only value needed by smoothing, peak detection,
    // and edge classification.
    const float dx = static_cast<float>(point.a) - center.a;
    const float dy = static_cast<float>(point.b) - center.b;
    signals[global_i] = sqrtf(dx * dx + dy * dy);
}

void radial_signals_batched_cuda(
    const BatchedContours &batch,
    BatchedContourCudaScratch &scratch,
    Signal &signals)
{
    signals.resize(batch.points.size());
    scratch.signals.resize(batch.points.size());
    if (batch.points.empty() || batch.offsets.empty())
    {
        return;
    }

    // One batched launch produces the flat radial signal for all pieces.
    const int block = 256;
    const dim3 grid(
        static_cast<unsigned int>((batch.max_length + block - 1) / block),
        static_cast<unsigned int>(batch.offsets.size()));
    batched_radial_signal_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(scratch.smoothed.data()),
        thrust::raw_pointer_cast(scratch.offsets.data()),
        thrust::raw_pointer_cast(scratch.lengths.data()),
        thrust::raw_pointer_cast(scratch.centers.data()),
        thrust::raw_pointer_cast(scratch.signals.data()),
        batch.max_length);
    CUDA_CHECK(cudaGetLastError());

    // Edge classification currently runs on CPU. Keep this copy explicit in
    // the radial stage so its cost stays visible in profiling.
    thrust::copy(scratch.signals.begin(), scratch.signals.end(), signals.begin());
}
