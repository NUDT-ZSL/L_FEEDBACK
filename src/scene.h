#pragma once

#include "math.h"

#include <string>
#include <vector>

namespace pbr {

constexpr int kMaxMaterials = 16;
constexpr int kMaxSpheres = 24;
constexpr int kMaxPointLights = 8;
constexpr int kMaxRectLights = 4;
constexpr int kResolution = 512;

struct Material {
    std::string name;
    Vec3 baseColor{0.8f, 0.8f, 0.8f};
    float metallic = 0.0f;
    float roughness = 0.5f;
    float anisotropy = 0.0f;
    float anisotropyRotationRadians = 0.0f;
    float ior = 1.5f;
};

struct Sphere {
    Vec3 basePosition{};
    float radius = 0.5f;
    int materialId = -1;
};

struct PointLight {
    Vec3 position{};
    Vec3 color{1.0f, 1.0f, 1.0f};
    float power = 100.0f;
};

struct RectLight {
    Vec3 center{};
    Vec3 normal{0.0f, -1.0f, 0.0f};
    Vec3 tangent{1.0f, 0.0f, 0.0f};
    Vec3 bitangent{0.0f, 0.0f, 1.0f};
    float halfWidth = 0.5f;
    float halfHeight = 0.5f;
    Vec3 radiance{1.0f, 1.0f, 1.0f};
};

struct Camera {
    Vec3 position{0.0f, 0.0f, 8.0f};
    Vec3 target{0.0f, 0.0f, 0.0f};
    Vec3 up{0.0f, 1.0f, 0.0f};
    float focalLengthMm = 50.0f;
    float sensorWidthMm = 36.0f;
    float sensorHeightMm = 36.0f;
    float fNumber = 2.8f;
    float focusDistance = 8.0f;
    float exposureTime = 1.0f / 60.0f;
};

struct Motion {
    float lateral = 0.0f;
    float vertical = 0.0f;
    float depth = 0.0f;
    float cycles = 1.0f;
};

struct Environment {
    Vec3 zenith{0.18f, 0.34f, 0.72f};
    Vec3 horizon{0.62f, 0.72f, 0.88f};
    Vec3 ground{0.08f, 0.075f, 0.065f};
    Vec3 sunDirection{0.35f, 0.72f, 0.60f};
    Vec3 sunRadiance{9.0f, 7.8f, 6.2f};
    float sunAngularRadiusDegrees = 2.0f;
};

struct Scene {
    Camera camera;
    Motion motion;
    Environment environment;
    std::vector<Material> materials;
    std::vector<Sphere> spheres;
    std::vector<PointLight> pointLights;
    std::vector<RectLight> rectLights;
};

bool loadScene(const std::string& path, Scene& scene, std::string& error);
Vec3 spherePositionAt(const Scene& scene, int sphereIndex, float normalizedTime);

}  // namespace pbr
