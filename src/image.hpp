#pragma once

#include <filesystem>
#include <vector>

#include <glm/glm.hpp>

namespace preview {

struct Texture {
    int width = 0;
    int height = 0;
    std::vector<glm::vec3> pixels;
    bool valid = false;
};

Texture loadTexture(const std::filesystem::path& path);
glm::vec3 sampleBilinearClamp(const Texture& texture, const glm::vec2& uv);

} // namespace preview
