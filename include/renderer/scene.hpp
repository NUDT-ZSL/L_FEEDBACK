#pragma once

#include "renderer/math.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace renderer {

enum class LightType {
    Point = 0,
    Directional = 1,
    Spot = 2,
};

struct Camera {
    Vec3 position{};
    Vec3 target{};
    Vec3 up{};
    float fovYDegrees = 0.0f;
    float nearPlane = 0.0f;
    float farPlane = 0.0f;
};

struct Material {
    std::string name;
    Vec3 albedo{};
    float metallic = 0.0f;
    float roughness = 0.0f;
};

struct Vertex {
    Vec3 position;
    Vec3 normal;
};

struct Mesh {
    std::string name;
    int material = 0;
    Mat4 transform = identity();
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct Light {
    LightType type = LightType::Point;
    Vec3 position{};
    Vec3 direction{};
    Vec3 color{};
    float intensity = 0.0f;
    float range = 0.0f;
    float innerConeDegrees = 0.0f;
    float outerConeDegrees = 0.0f;
    float orthoSize = 0.0f;
    float nearPlane = 0.0f;
    float farPlane = 0.0f;
    float shadowBias = 0.0f;
    bool castShadow = false;
};

struct ShadowAtlasSettings {
    int tileResolution = 0;
    int columns = 0;
    int rows = 0;
    int pcfKernel = 0;
    float pcfStep = 0.0f;
};

struct DenoiseSettings {
    int filterRadius = 0;
    std::vector<float> weights;
};

struct ModeSettings {
    int shadowSamples = 0;
    int postprocessIterations = 0;
    float atlasScale = 0.0f;
    bool gaussianFilter = false;
};

struct RenderSettings {
    int width = 0;
    int height = 0;
    std::string output;
    Vec3 background{};
    float exposure = 0.0f;
    std::string toneMapping;
    ShadowAtlasSettings shadowAtlas;
    DenoiseSettings denoise;
    ModeSettings fast;
    ModeSettings high;
};

struct Scene {
    Camera camera;
    std::vector<Material> materials;
    std::vector<Mesh> meshes;
    std::vector<Light> lights;
    RenderSettings settings;
    Aabb bounds;
    std::string sourcePath;
};

Scene load_scene(const std::string& path);
void generate_samples(const std::string& directory);

int shadow_tile_demand(const Scene& scene);
int shadow_tile_capacity(const Scene& scene);
const ModeSettings& mode_settings(const Scene& scene, const std::string& mode);
int effective_tile_resolution(const Scene& scene, const std::string& mode);
bool is_valid_mode(const std::string& mode);

}  // namespace renderer
