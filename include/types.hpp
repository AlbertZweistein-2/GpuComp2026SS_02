#pragma once

#include <array>
#include <cstdint>
#include <vector>

// Some custom structs
// ______________________________________________________________________________________________

struct ImageU8{
    int width;
    int height;
    std::vector<uint8_t> data; // row-major order
};

struct ImageI32{
    // QUESTION: Do we really need 32 bit?
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
    bool operator<(const Coordinate& c2) const {
        if (b != c2.b) return b < c2.b;
        return a < c2.a;
    }

    // needed for the convergence criterion in the Moore neighbor tracing
    // converges when the current point equals the starting point
    bool operator==(const Coordinate& c2) const {
        return a == c2.a && b == c2.b;
    }
};

// Again, a custom coordinate struct
// but this time with floats
// only needed for the center point of the encloding circle
// struct CoordinateFloat {
//     float a;
//     float b;
// };


template <typename T>
using CoordinateVector = std::vector<Coordinate<T>>;

using Signal = std::vector<float>;
using Corners = std::array<int, 4>;
