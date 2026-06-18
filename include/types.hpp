#pragma once

#include <array>
#include <cstdint>
#include <vector>

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

// Some custom structs
// ______________________________________________________________________________________________

struct ImageU8{
    int width;
    int height;
    std::vector<uint8_t> data; // row-major order
};

struct ImageI32{
    int width;
    int height;
    std::vector<int32_t> data; // row-major order
};

struct Region {
    int label;
    int area;
    int x;
    int y;
    int width;
    int height;
};

// Custom coordinate struct
template <typename T>
struct Coordinate {
    // Using a and b instead of x and y, since the orginal python code switches from (x, y) to (y, x) once
    T a;
    T b;

    // some custom operators
    // needed for the std::set to store unique coordinates during Moore neighbor tracing
    #ifdef __CUDACC__
        // making the operator compatile with use in cuda kernels when compiled with nvcc
        __host__ __device__
    #endif
    bool operator<(const Coordinate& c2) const {
        if (b != c2.b) return b < c2.b;
        return a < c2.a;
    }

    // needed for the convergence criterion in the Moore neighbor tracing
    // converges when the current point equals the starting point
    #ifdef __CUDACC__
        // making the operator compatile with use in cuda kernels when compiled with nvcc
        __host__ __device__ 
    #endif
    bool operator==(const Coordinate& c2) const {
        return a == c2.a && b == c2.b;
    }
};


template <typename T>
using CoordinateVector = std::vector<Coordinate<T>>;

enum class EdgeType { Straight, Knob, Hole };

using Signal = std::vector<float>;
using Corners = std::array<int, 4>;
using EdgeLabels = std::array<EdgeType, 4>;
