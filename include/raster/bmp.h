#pragma once

#include "raster/math3d.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace raster {

struct Texture {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<Vec3> texels;

    Vec3 sampleNearestRepeat(const Vec2& uv) const {
        if (width == 0U || height == 0U ||
            texels.size() != static_cast<std::size_t>(width) * height) {
            return {};
        }

        const double wrappedU = uv.x - std::floor(uv.x);
        const double wrappedV = uv.y - std::floor(uv.y);
        std::uint32_t x = static_cast<std::uint32_t>(
            std::floor(wrappedU * static_cast<double>(width)));
        std::uint32_t yFromBottom = static_cast<std::uint32_t>(
            std::floor(wrappedV * static_cast<double>(height)));
        if (x >= width) {
            x = width - 1U;
        }
        if (yFromBottom >= height) {
            yFromBottom = height - 1U;
        }
        const std::uint32_t y = height - 1U - yFromBottom;
        return texels[static_cast<std::size_t>(y) * width + x];
    }
};

bool loadBmpTexture(const std::filesystem::path& path, Texture& texture,
                    std::string& error);

bool writeBmp24(const std::filesystem::path& path, int width, int height,
                const std::vector<std::uint8_t>& bgrTopDown,
                std::string& error);

}  // namespace raster

