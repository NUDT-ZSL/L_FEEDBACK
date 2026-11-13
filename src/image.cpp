#include "image.hpp"

#include "sha256.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstring>
#include <stdexcept>

namespace lfb {

namespace {

void appendPngBytes(void* context, void* data, int size) {
    if (size <= 0) {
        return;
    }
    auto* output = static_cast<std::vector<std::uint8_t>*>(context);
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    output->insert(output->end(), bytes, bytes + size);
}

}  // namespace

double averageRgb(const Image& image) {
    const std::size_t pixelCount =
        static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height);
    if (pixelCount == 0) {
        return 0.0;
    }
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < pixelCount; ++i) {
        sum += image.rgba[i * 4];
        sum += image.rgba[i * 4 + 1];
        sum += image.rgba[i * 4 + 2];
    }
    return static_cast<double>(sum) / static_cast<double>(pixelCount * 3);
}

std::string pixelChecksum(const Image& image) {
    return Sha256::hash(image.rgba.data(), image.rgba.size());
}

std::vector<std::uint8_t> encodePng(const Image& image) {
    if (image.width <= 0 || image.height <= 0 ||
        image.rgba.size() != static_cast<std::size_t>(image.width) *
                                 static_cast<std::size_t>(image.height) * 4) {
        throw std::runtime_error("invalid image buffer for PNG encoding");
    }

    stbi_write_png_compression_level = 8;
    stbi_write_force_png_filter = 0;

    std::vector<std::uint8_t> png;
    const int ok = stbi_write_png_to_func(
        appendPngBytes, &png, image.width, image.height, 4,
        image.rgba.data(), image.width * 4);
    if (!ok || png.empty()) {
        throw std::runtime_error("PNG encoding failed");
    }
    return png;
}

}  // namespace lfb
