#include "image_io.h"

#include <png.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace dr {

bool writePng(const std::string& path, int width, int height,
              const std::vector<std::uint8_t>& rgbaTopDown, std::string& error) {
    if (width <= 0 || height <= 0 ||
        rgbaTopDown.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4U) {
        error = "invalid RGBA image dimensions";
        return false;
    }

    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    image.width = static_cast<png_uint_32>(width);
    image.height = static_cast<png_uint_32>(height);
    image.format = PNG_FORMAT_RGBA;

    if (png_image_write_to_file(&image, path.c_str(), 0, rgbaTopDown.data(), 0, nullptr) == 0) {
        error = image.message;
        return false;
    }
    return true;
}

bool readPng(const std::string& path, int& width, int& height,
             std::vector<std::uint8_t>& rgbaTopDown, std::string& error) {
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    if (png_image_begin_read_from_file(&image, path.c_str()) == 0) {
        error = image.message;
        return false;
    }

    image.format = PNG_FORMAT_RGBA;
    const png_size_t byteCount = PNG_IMAGE_SIZE(image);
    rgbaTopDown.resize(byteCount);
    if (png_image_finish_read(&image, nullptr, rgbaTopDown.data(), 0, nullptr) == 0) {
        error = image.message;
        png_image_free(&image);
        return false;
    }

    width = static_cast<int>(image.width);
    height = static_cast<int>(image.height);
    png_image_free(&image);
    return true;
}

bool compareImages(const std::vector<std::uint8_t>& actual, int actualWidth, int actualHeight,
                   const std::vector<std::uint8_t>& reference, int referenceWidth, int referenceHeight,
                   ImageDiff& diff, std::string& error) {
    if (actualWidth != referenceWidth || actualHeight != referenceHeight) {
        error = "image dimensions do not match";
        return false;
    }
    const std::size_t expected =
        static_cast<std::size_t>(actualWidth) * static_cast<std::size_t>(actualHeight) * 4U;
    if (actual.size() != expected || reference.size() != expected) {
        error = "image buffers have invalid sizes";
        return false;
    }

    diff = ImageDiff{};
    diff.width = actualWidth;
    diff.height = actualHeight;
    std::uint64_t absoluteDifference = 0;
    for (std::size_t pixel = 0; pixel < expected / 4U; ++pixel) {
        int pixelMaximum = 0;
        for (std::size_t channel = 0; channel < 4; ++channel) {
            const int a = actual[pixel * 4U + channel];
            const int b = reference[pixel * 4U + channel];
            const int channelDifference = std::abs(a - b);
            pixelMaximum = std::max(pixelMaximum, channelDifference);
            absoluteDifference += static_cast<std::uint64_t>(channelDifference);
        }
        diff.maxChannelDifference = std::max(diff.maxChannelDifference, pixelMaximum);
        if (pixelMaximum > 0) {
            ++diff.differentPixels;
        }
    }
    diff.meanAbsoluteDifference =
        static_cast<double>(absoluteDifference) / static_cast<double>(expected);
    return true;
}

} // namespace dr
