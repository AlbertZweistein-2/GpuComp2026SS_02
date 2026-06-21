#include "serial/cleaning.hpp"

#include <cstddef>

namespace
{
// The serial baseline uses one fixed rectangular structuring element. The CUDA
// path mirrors this with tiled erosion/dilation kernels.
constexpr int MORPH_KERNEL_WIDTH = 5;
constexpr int MORPH_KERNEL_HEIGHT = 5;
constexpr int MORPH_ITERATIONS = 1;

ImageU8 make_kernel(int width, int height)
{
    // A value of 1 means the kernel position participates in the morphology
    // test. This creates a dense rectangular kernel.
    ImageU8 kernel;
    kernel.width = width;
    kernel.height = height;
    kernel.data.assign(static_cast<std::size_t>(width) * height, 1);
    return kernel;
}
} // namespace

static void erode(const ImageU8 &image, const ImageU8 &kernel, ImageU8 &result, int iterations)
{
    // Start from the input image so multiple iterations can feed the previous
    // iteration result into the next one.
    result = image;

    const int height = image.height;
    const int width = image.width;

    const int kh = kernel.height;
    const int kw = kernel.width;

    const int pad_h = kh / 2;
    const int pad_w = kw / 2;

    for (int iter = 0; iter < iterations; ++iter) {
        // Write into a fresh image for this iteration; reading and writing the
        // same buffer would make neighborhood results order-dependent.
        ImageU8 new_image;
        new_image.width = width;
        new_image.height = height;
        new_image.data.resize(static_cast<std::size_t>(width) * height, 0);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {

                bool all_foreground = true;

                for (int ky = 0; ky < kh; ++ky) {
                    for (int kx = 0; kx < kw; ++kx) {

                        // Keep the generic kernel representation even though
                        // make_kernel currently creates an all-ones rectangle.
                        if (kernel.data[ky * kw + kx] != 1) {
                            continue;
                        }

                        int iy = y + ky - pad_h;
                        int ix = x + kx - pad_w;

                        uint8_t pixel = 0;

                        // Out-of-image samples are treated as background,
                        // shrinking foreground near image borders.
                        if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                            pixel = result.data[iy * width + ix];
                        }

                        if (pixel != 255) {
                            all_foreground = false;
                            break;
                        }
                    }

                    if (!all_foreground) {
                        break;
                    }
                }

                if (all_foreground) {
                    new_image.data[y * width + x] = 255;
                }
            }
        }
        result = new_image;
    }
}

static void dilate(const ImageU8 &image, const ImageU8 &kernel, ImageU8 &result, int iterations)
{
    // Same iteration structure as erosion, but dilation only needs one
    // foreground neighbor under the kernel.
    result = image;

    const int height = image.height;
    const int width = image.width;

    const int kh = kernel.height;
    const int kw = kernel.width;

    const int pad_h = kh / 2;
    const int pad_w = kw / 2;

    for (int iter = 0; iter < iterations; ++iter) {
        ImageU8 new_image;
        new_image.width = width;
        new_image.height = height;
        new_image.data.resize(static_cast<std::size_t>(width) * height, 0);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {

                bool any_foreground = false;

                for (int ky = 0; ky < kh; ++ky) {
                    for (int kx = 0; kx < kw; ++kx) {

                        // Supports sparse kernels if make_kernel is ever
                        // replaced, while preserving current rectangular use.
                        if (kernel.data[ky * kw + kx] != 1) {
                            continue;
                        }

                        int iy = y + ky - pad_h;
                        int ix = x + kx - pad_w;

                        if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                            if (result.data[iy * width + ix] == 255) {
                                any_foreground = true;
                                break;
                            }
                        }
                    }

                    if (any_foreground) {
                        break;
                    }
                }

                if (any_foreground) {
                    new_image.data[y * width + x] = 255;
                }
            }
        }

        result = new_image;
    }
}

void morphological_open(const ImageU8 &image, ImageU8 &result)
{
    // Opening removes small foreground specks: erode removes thin/noisy regions,
    // then dilation restores the remaining foreground shape.
    const ImageU8 kernel = make_kernel(MORPH_KERNEL_WIDTH, MORPH_KERNEL_HEIGHT);
    ImageU8 eroded;
    erode(image, kernel, eroded, MORPH_ITERATIONS);
    dilate(eroded, kernel, result, MORPH_ITERATIONS);
}
