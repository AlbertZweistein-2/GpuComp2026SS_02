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
    // Moore neighbor tracing algorithm to get the vector of contour coordinates
    // Ihnerently sequential: every next point depends on the
    // current point and the previous search direction

    // All possible parallel algorithms return a set of unordered contour points
    // and sorting and merging them is messy and slow
    // And since the serial function is fast enough, we just keep it this way
    void trace_contour(const uint8_t *boundary, int width, int height, CoordinateVector<int> &result)
    {
        // Goes from top-left to bottom-right to find the first nonzero pixel.
        // This then becomes the starting point.
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
            result.clear();
            return;
        }

        // the actual contour vector
        result.clear();
        // adding the start pixel
        result.push_back(start);

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
                const int dir_idx = (prev_dir + i) % 8;

                // nx and ny are the coordinates of the neighboring pixel we are currently checking
                const int ny = current.a + dy[dir_idx];
                const int nx = current.b + dx[dir_idx];

                // a lot of conditions
                // if nx and ny are within the image bounds
                // and if the pixal at (ny, nx) is nonzero
                if (0 <= ny && ny < height && 0 <= nx && nx < width && boundary[ny * width + nx] > 0)
                {
                    Coordinate<int> next_point{ny, nx};
                    // if this is the starting point again
                    // and we have found more than ten points
                    // we converge
                    if (next_point == start && result.size() > 10)
                    {
                        return;
                    }

                    // if it has not been visited before
                    if (visited.insert(next_point).second)
                    {
                        // we add it to the contour vector
                        result.push_back(next_point);
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
    }

    // Removes unnecessary points along straight lines.
    // Also iherently sequential due to the same reasons as stated above
    // Again, the serial function is fast enough, so we leave it this way
    void simplify_chain_approx(CoordinateVector<int> &contour)
    {
        if (contour.size() < 3)
        {
            return;
        }

        CoordinateVector<int> simplified;
        simplified.push_back(contour.front());

        // we iterate through the contour points, starting from the second point
        for (size_t i = 1; i < contour.size() - 1; ++i)
        {
            // we use a sliding window of three points
            // the previous one
            const int y0 = contour[i - 1].a;
            const int x0 = contour[i - 1].b;
            // the current one
            const int y1 = contour[i].a;
            const int x1 = contour[i].b;
            // and the next one
            const int y2 = contour[i + 1].a;
            const int x2 = contour[i + 1].b;

            // we then check the direction between the
            // previous and current point
            const int dx1 = std::clamp(x1 - x0, -1, 1);
            const int dy1 = std::clamp(y1 - y0, -1, 1);
            // and current and next point
            const int dx2 = std::clamp(x2 - x1, -1, 1);
            const int dy2 = std::clamp(y2 - y1, -1, 1);

            // if the two directions are different, the contour is turning
            // so the current point is a necessary corner
            if (dx1 != dx2 || dy1 != dy2)
            {
                // and we add it to the simplified contour
                simplified.push_back(contour[i]);
            }
        }
        simplified.push_back(contour.back());
        contour = std::move(simplified);
    }

} // namespace

// Just a function to first call the two previous functions
// for contour tracing and simplification
// and then swaps the coordinates from (y, x) back to (x, y) just like in the original python code
void find_contour_chain_approx_simple_cuda(const uint8_t *boundary, int width, int height, CoordinateVector<int> &result)
{
    // Contour extraction stays serial
    trace_contour(boundary, width, height, result);
    // Contour simplification too
    simplify_chain_approx(result);

    // swapping coordinates from (y, x) to (x, y)
    // just like in the original python code
    // We experimented with using Thrust for this
    // but the overhead was larger than the speedup
    // probably due to the simplicity of the operation and low number of points
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

// These serial functions above are run on the CPU
// Using OpenMP to parallelize them across pieces

// But the results are then put back into device memory
// in such a way that all pieces can be processed in parallel in a single batch

// Batched moving average smoothing
// Offsets[i] indicates where the i-th piece starts
// and lengths[i] indicates how many points it has
__global__ void batched_moving_average_kernel(
    const Coordinate<int> *points,
    Coordinate<int> *smoothed,
    const int *offsets,
    const int *lengths,
    int half_window,
    int window,
    int max_length)
{
    // The piece number is given by blockIdx.y
    const int piece_idx = blockIdx.y;
    // The index of the point within the piece
    const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_i >= max_length)
    {
        return;
    }

    const int n = lengths[piece_idx];
    // We only need as many threads as there are points in the piece
    if (local_i >= n)
    {
        return;
    }

    const int offset = offsets[piece_idx];
    const int global_i = offset + local_i;
    // If the window is too small or too large, just return the original point without smoothing
    if (window <= 1 || window >= n)
    {
        smoothed[global_i] = points[global_i];
        return;
    }

    // One sum for each coordinate component
    float sum_a = 0.0f;
    float sum_b = 0.0f;
    // Each thread iterates over a window of points centered on its local index
    for (int window_j = -half_window; window_j <= half_window; ++window_j)
    {
        // The contour is closed, so border points wrap around to
        // the other side of the contour, done using modulo
        int local_j = (local_i + window_j) % n;

        // If the modulo returns a negative index, we correct it
        if (local_j < 0)
        {
            local_j += n;
        }

        // offset + local_j gives the global index of the point in the flat points array
        const Coordinate<int> point = points[offset + local_j];
        sum_a += static_cast<float>(point.a);
        sum_b += static_cast<float>(point.b);
    }

    const float divisor = static_cast<float>(window);
    // dividing the sums by the window size and rounding to the nearest integer
    // Since the points are integer coordinates
    smoothed[global_i] = Coordinate<int>{
        static_cast<int>(roundf(sum_a / divisor)),
        static_cast<int>(roundf(sum_b / divisor))};
}

void smooth_contours_batched_cuda(
    const BatchedContours &batch,
    int window,
    BatchedContourCudaScratch &scratch)
{
    // Contour tracing returns contours on the host
    // Therefore we dynamically resize the thrust device vectors
    // within the scratch structure
    scratch.points.resize(batch.points.size());
    scratch.smoothed.resize(batch.points.size());
    scratch.offsets.resize(batch.offsets.size());
    scratch.lengths.resize(batch.lengths.size());

    // No need to launch any kernels if there are no points
    if (batch.points.empty() || batch.offsets.empty())
    {
        return;
    }

    // Loading the data from host to device
    // Into the newly resized thrust device vectors
    thrust::copy(batch.points.begin(), batch.points.end(), scratch.points.begin());
    thrust::copy(batch.offsets.begin(), batch.offsets.end(), scratch.offsets.begin());
    thrust::copy(batch.lengths.begin(), batch.lengths.end(), scratch.lengths.begin());

    // Using a 2D grid where gridDim.y covers pieces and gridDim.x covers points within pieces
    const int block = 256;
    // Launchin one block per piece, with 256 threads per block
    // One batched launch produces the flat smoothed contour for all pieces
    const dim3 grid(
        static_cast<unsigned int>((batch.max_length + block - 1) / block),
        static_cast<unsigned int>(batch.offsets.size()));
    batched_moving_average_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(scratch.points.data()),
        thrust::raw_pointer_cast(scratch.smoothed.data()),
        thrust::raw_pointer_cast(scratch.offsets.data()),
        thrust::raw_pointer_cast(scratch.lengths.data()),
        window / 2, // Integer division
        window,
        batch.max_length);
    // checking the kernel launch for success
    // smoothed stays on device after the kernel
    CUDA_CHECK(cudaGetLastError());
}

// Calculates the center and radius of a circle that encloses the contour points

// In a previous version we solved this with trust reductions
// But a custom kernel can do all x and y, min and max reductions in one pass
// and can also process all pieces in one launch
// But we could have done the reductions using warp shuffles which would have been even faster
// But this stage is fast enough
__global__ void batched_enclosing_circle_kernel(
    const Coordinate<int> *points,
    const int *offsets,
    const int *lengths,
    Coordinate<float> *centers)
{
    // one shared memory slot per thread for each of the four values
    // so 256 threads per block
    __shared__ int s_min_x[256];
    __shared__ int s_max_x[256];
    __shared__ int s_min_y[256];
    __shared__ int s_max_y[256];

    // One block per piece
    // With 256 threads per block
    const int piece_idx = blockIdx.x;
    const int tid = threadIdx.x;
    // length of the current piece
    const int n = lengths[piece_idx];
    // and its starting index in the flat points array
    const int offset = offsets[piece_idx];

    // Initial values
    int min_x = INT_MAX;
    int max_x = INT_MIN;
    int min_y = INT_MAX;
    int max_y = INT_MIN;

    // Every point performs a strided scan of his assigned points
    // to get his local mins and maxes
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

    // Filling shared memory with each threads local min and max values
    s_min_x[tid] = min_x;
    s_max_x[tid] = max_x;
    s_min_y[tid] = min_y;
    s_max_y[tid] = max_y;
    // Waiting for all threads to finish
    __syncthreads();

    // Doing tree reduction, every thread compares its value
    // with the value of the thread at stride distance away
    // And only keeps the min or max value
    // So the total number of potential min and max values is halved each iteration
    // Done after log2(blockDim.x) iterations
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
        // We need to wait for all threads to finish before the next iteration
        __syncthreads();
    }

    // Thread 0 now has the final min and max values in shared memory
    // in shared memory slots 0
    if (tid == 0)
    {
        if (n <= 0)
        {
            // One center per piece
            centers[piece_idx] = Coordinate<float>{0.0f, 0.0f};
        }
        else
        {
            // The whole thing is called approximate because
            // this is not the minimum enclosing circle but a somewhat accurate approximation of it
            const float center_x = (static_cast<float>(s_min_x[0]) + static_cast<float>(s_max_x[0])) * 0.5f;
            const float center_y = (static_cast<float>(s_min_y[0]) + static_cast<float>(s_max_y[0])) * 0.5f;
            centers[piece_idx] = Coordinate<float>{center_x, center_y};
        }
    }
}

// Just calling the kernel above
// As with the previous stage, we process all pieces at once in a single batch
void enclosing_circle_centers_batched_cuda(
    const BatchedContours &batch,
    BatchedContourCudaScratch &scratch)
{
    const std::size_t piece_count = batch.offsets.size();
    // Dynamically resizing the thrust device vector for the centers
    // to hold one center per piece
    scratch.centers.resize(piece_count);
    // No need to launch any kernels if there are no pieces
    if (piece_count == 0)
    {
        return;
    }
    // No need to launch any kernels if there are no points
    if (scratch.smoothed.empty())
    {
        return;
    }

    // One batched launch produces the centers for all pieces
    // One block processed the contour of one piece
    // With 256 threads per block
    const int block = 256;
    batched_enclosing_circle_kernel<<<static_cast<unsigned int>(piece_count), block>>>(
        thrust::raw_pointer_cast(scratch.smoothed.data()),
        thrust::raw_pointer_cast(scratch.offsets.data()),
        thrust::raw_pointer_cast(scratch.lengths.data()),
        thrust::raw_pointer_cast(scratch.centers.data()));
    // checking the kernel launch for success
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
    // Again, piece number is given by blockIdx.y
    // As there is one block per piece
    const int piece_idx = blockIdx.y;
    const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    // We only need as many threads as there are points in the longest piece
    if (local_i >= max_length)
    {
        return;
    }

    // Again, we only need as many threads as there are points in the current piece
    const int n = lengths[piece_idx];
    if (local_i >= n)
    {
        return;
    }

    const int global_i = offsets[piece_idx] + local_i;
    const Coordinate<int> point = points[global_i];
    const Coordinate<float> center = centers[piece_idx];
    // Calculating the L2 distance from the center to the point
    const float dx = static_cast<float>(point.a) - center.a;
    const float dy = static_cast<float>(point.b) - center.b;
    // And we only need to return this distance
    // No other results are needed
    signals[global_i] = sqrtf(dx * dx + dy * dy);
}

// As with the previous stages, we process all pieces at once in a single batch
void radial_signals_batched_cuda(
    const BatchedContours &batch,
    BatchedContourCudaScratch &scratch,
    Signal &signals)
{
    // Dynamically resizing the thrust device vector for the signals
    signals.resize(batch.points.size());
    scratch.signals.resize(batch.points.size());
    // If there are no points or pieces, we dont need to launch any kernels
    if (batch.points.empty() || batch.offsets.empty())
    {
        return;
    }

    // One batched launch produces the flat radial signal for all pieces
    const int block = 256;
    // Again, one block per piece, with 256 threads per block
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
    // checking the kernel launch for success
    CUDA_CHECK(cudaGetLastError());

    // The next stage runs on CPU, but we keep this copy in
    // the radial stage so its cost stays visible in profiling
    thrust::copy(scratch.signals.begin(), scratch.signals.end(), signals.begin());
}
