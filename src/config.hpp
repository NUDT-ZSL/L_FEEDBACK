#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace preview {

struct CameraConfig {
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 target = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    float verticalFovDegrees = 45.0f;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;
};

struct Attenuation {
    float constant = 1.0f;
    float linear = 0.0f;
    float quadratic = 0.0f;
};

struct LightConfig {
    enum class Type { Directional, Point };

    Type type = Type::Directional;
    glm::vec3 color = glm::vec3(1.0f);
    float intensity = 1.0f;
    glm::vec3 direction = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 position = glm::vec3(0.0f);
    Attenuation attenuation;
};

struct RenderConfig {
    int width = 0;
    int height = 0;
    bool doubleSided = false;
    glm::vec3 ambient = glm::vec3(0.03f);
    glm::vec3 background = glm::vec3(0.0f);
    std::filesystem::path objPath;
    std::filesystem::path outputPath;
    CameraConfig camera;
    std::vector<LightConfig> lights;
};

RenderConfig loadConfig(const std::filesystem::path& configPath);

} // namespace preview
