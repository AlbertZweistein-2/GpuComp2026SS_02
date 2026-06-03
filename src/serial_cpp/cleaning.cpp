#include "serial/cleaning.hpp"

static void erode(const ImageU8 &image, const ImageU8 &kernel, ImageU8 &result, int iterations)
{
    result = image;

    const int height = image.height;
    const int width = image.width;

    const int kh = kernel.height;
    const int kw = kernel.width;

    const int pad_h = kh / 2;
    const int pad_w = kw / 2;

    for (int iter = 0; iter < iterations; ++iter) {
        //directly fill result
        ImageU8 new_image;
        new_image.width = width;
        new_image.height = height;
        new_image.data.resize(static_cast<size_t>(width) * height, 0);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {

                bool all_foreground = true;

                for (int ky = 0; ky < kh; ++ky) {
                    for (int kx = 0; kx < kw; ++kx) {

                        if (kernel.data[ky * kw + kx] != 1) {
                            continue;
                        }

                        int iy = y + ky - pad_h;
                        int ix = x + kx - pad_w;

                        uint8_t pixel = 0;

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
        new_image.data.resize(static_cast<size_t>(width) * height, 0);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {

                bool any_foreground = false;

                for (int ky = 0; ky < kh; ++ky) {
                    for (int kx = 0; kx < kw; ++kx) {

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

void morphological_open(const ImageU8 &image, const ImageU8 &kernel, ImageU8 &result, int iterations)
{
    ImageU8 eroded;
    erode(image, kernel, eroded, iterations);
    dilate(eroded, kernel, result, iterations);
}
