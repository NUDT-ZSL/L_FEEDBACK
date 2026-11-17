#pragma once

#include "math.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ct {

struct ColorBuffer {
    int width = 0;
    int height = 0;
    std::vector<Vec3> pixels;

    ColorBuffer() = default;
    ColorBuffer(int width_, int height_) : width(width_), height(height_), pixels(
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_)) {}

    Vec3& at(int x, int y) {
        return pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                      static_cast<std::size_t>(x)];
    }

    const Vec3& at(int x, int y) const {
        return pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                      static_cast<std::size_t>(x)];
    }
};

bool writeBmp(const std::string& path, const ColorBuffer& image, std::string& error);
bool writePpm(const std::string& path, const ColorBuffer& image, std::string& error);
bool writeJsonText(const std::string& path, const std::string& text, std::string& error);

}  // namespace ct
