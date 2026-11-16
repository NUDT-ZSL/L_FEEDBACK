#include "image.hpp"

#include "error.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

#include "stb_image.h"

namespace preview {
namespace {

int clampIndex(int value, int maximum) {
    return std::max(0, std::min(value, maximum));
}

std::string unicodePath(const std::filesystem::path& path) {
    return path.u8string();
}

} // namespace

Texture loadTexture(const std::filesystem::path& path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    const std::string utf8Path = unicodePath(path);
    stbi_uc* decoded = stbi_load(utf8Path.c_str(), &width, &height, &channels, 3);
    if (decoded == nullptr) {
        const char* reason = stbi_failure_reason();
        throw AppError(
            ErrorCode::ImageDecode,
            "failed to decode PNG '" + utf8Path + "': " +
                (reason != nullptr ? reason : "unknown decoder error"));
    }
    if (width <= 0 || height <= 0 ||
        static_cast<std::size_t>(width) >
            std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(height) / 3U) {
        stbi_image_free(decoded);
        throw AppError(ErrorCode::ImageDecode, "PNG dimensions are invalid: " + utf8Path);
    }

    Texture texture;
    texture.width = width;
    texture.height = height;
    texture.valid = true;
    texture.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t source =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(x)) *
                3U;
            texture.pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                           static_cast<std::size_t>(x)] =
                glm::vec3(
                    static_cast<float>(decoded[source + 0]) / 255.0f,
                    static_cast<float>(decoded[source + 1]) / 255.0f,
                    static_cast<float>(decoded[source + 2]) / 255.0f);
        }
    }
    stbi_image_free(decoded);
    return texture;
}

glm::vec3 sampleBilinearClamp(const Texture& texture, const glm::vec2& uv) {
    if (!texture.valid || texture.width <= 0 || texture.height <= 0) {
        return glm::vec3(0.0f);
    }

    const float clampedU = std::clamp(uv.x, 0.0f, 1.0f);
    const float clampedV = std::clamp(uv.y, 0.0f, 1.0f);
    const float x = clampedU * static_cast<float>(texture.width) - 0.5f;
    const float y = (1.0f - clampedV) * static_cast<float>(texture.height) - 0.5f;
    const float x0f = std::floor(x);
    const float y0f = std::floor(y);
    const float tx = x - x0f;
    const float ty = y - y0f;

    const int maxX = texture.width - 1;
    const int maxY = texture.height - 1;
    const int x0 = clampIndex(static_cast<int>(x0f), maxX);
    const int x1 = clampIndex(static_cast<int>(x0f) + 1, maxX);
    const int y0 = clampIndex(static_cast<int>(y0f), maxY);
    const int y1 = clampIndex(static_cast<int>(y0f) + 1, maxY);

    const auto at = [&texture](int px, int py) -> const glm::vec3& {
        return texture.pixels[
            static_cast<std::size_t>(py) * static_cast<std::size_t>(texture.width) +
            static_cast<std::size_t>(px)];
    };

    const glm::vec3 top = glm::mix(at(x0, y0), at(x1, y0), tx);
    const glm::vec3 bottom = glm::mix(at(x0, y1), at(x1, y1), tx);
    return glm::mix(top, bottom, ty);
}

} // namespace preview
