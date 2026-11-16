#pragma once

#include "common.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace skin {

struct LUTSettings {
    int profileSamples = 128;
    int curvatureSamples = 64;
    int scatteringSamples = 24;
    float curvatureMin = 0.0F;
    float curvatureMax = 0.10F;
    glm::vec3 scatterColor{1.0F, 0.70F, 0.60F};
    glm::vec3 scatterRadius{0.16F, 0.09F, 0.055F};
    float wrap = 0.35F;
    float profilePower = 1.35F;
};

enum class ToneMapMode : int {
    Aces = 0,
    Reinhard = 1,
    None = 2,
};

struct ToneSettings {
    ToneMapMode mode = ToneMapMode::Aces;
    float exposure = 1.0F;
    float gamma = 2.2F;
};

struct CameraSettings {
    glm::vec3 position{0.0F, 0.0F, 6.2F};
    glm::vec3 target{0.0F, 0.0F, 0.0F};
    glm::vec3 up{0.0F, 1.0F, 0.0F};
    float fovYDegrees = 40.0F;
};

struct LightSettings {
    glm::vec3 position{1.55F, 2.25F, 3.15F};
    glm::vec3 target{0.0F, 0.0F, 0.0F};
    glm::vec3 color{1.0F, 0.92F, 0.84F};
    float width = 1.15F;
    float height = 0.62F;
    float intensity = 15.0F;
    int samplesX = 6;
    int samplesY = 6;
};

struct EnvironmentSettings {
    glm::vec3 sky{0.12F, 0.16F, 0.22F};
    glm::vec3 ground{0.025F, 0.018F, 0.014F};
    float intensity = 0.75F;
};

struct SceneSettings {
    float sphereRadius = 1.0F;
    float sphereSpacing = 2.25F;
    glm::vec3 background{0.008F, 0.011F, 0.016F};
};

struct MaterialSpec {
    std::string name = "material";
    glm::vec3 albedo{0.72F, 0.37F, 0.30F};
    float roughness = 0.42F;
    float anisotropy = 0.35F;
    float anisotropyRotation = 0.0F;
    glm::vec3 sssColor{1.0F, 0.78F, 0.68F};
    float sssStrength = 1.0F;
    float sssCurvature = 0.045F;
    float specularF0 = 0.028F;
};

struct GroupSpec {
    int index = -1;
    std::string name;
    MaterialSpec primary;
    MaterialSpec reference;
    std::string baselineConfigured;
    std::filesystem::path baselinePath;
    std::string baselineVersion;
    int errorCode = kErrorNone;
    std::string errorMessage;

    [[nodiscard]] bool valid() const { return errorCode == kErrorNone; }
};

struct RenderConfig {
    int width = 512;
    int height = 512;
    std::uint32_t seed = 1337U;
    LUTSettings lut;
    ToneSettings tone;
    CameraSettings camera;
    LightSettings light;
    EnvironmentSettings environment;
    SceneSettings scene;

    std::filesystem::path sourcePath;
    std::filesystem::path baseDirectory;
    std::filesystem::path outputDirectory;
    std::string outputDirectoryConfigured = "output";
    std::string fingerprint;

    std::vector<GroupSpec> groups;
};

struct ConfigLoadResult {
    bool ok = false;
    std::string error;
    RenderConfig config;
};

ConfigLoadResult loadConfig(const std::filesystem::path& path);

MaterialSpec defaultReferenceMaterial();

}  // namespace skin

