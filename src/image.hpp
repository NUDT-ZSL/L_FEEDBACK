#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lfb {

struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgba;

    Image() = default;
    Image(int w, int h) : width(w), height(h),
        rgba(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4u, 0u) {}
};

double averageRgb(const Image& image);
std::string pixelChecksum(const Image& image);
std::vector<std::uint8_t> encodePng(const Image& image);

}  // namespace lfb
