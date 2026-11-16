#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "image.hpp"

namespace preview {

struct Vertex {
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 normal = glm::vec3(0.0f, 0.0f, 1.0f);
    glm::vec2 uv = glm::vec2(0.0f);
    glm::vec3 tangent = glm::vec3(1.0f, 0.0f, 0.0f);
    float tangentSign = 1.0f;
};

struct Triangle {
    unsigned int indices[3] = {0U, 0U, 0U};
    int materialIndex = -1;
};

struct Material {
    std::string name;
    std::filesystem::path diffusePath;
    std::filesystem::path normalPath;
    glm::vec3 diffuseTint = glm::vec3(1.0f);
    float specularStrength = 0.25f;
    float shininess = 32.0f;
    Texture diffuseTexture;
    Texture normalTexture;
};

struct Model {
    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
};

Model loadModel(const std::filesystem::path& objPath);

} // namespace preview
