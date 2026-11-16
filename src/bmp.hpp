#pragma once

#include <filesystem>
#include <vector>

#include <glm/glm.hpp>

namespace preview {

struct Image {
    int width = 0;
    int height = 0;
    std::vector<glm::vec3> pixels;
};

void writeBmp24(const std::filesystem::path& path, const Image& image);

} // namespace preview
