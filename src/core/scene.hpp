#pragma once

#include "core/math.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace uid166 {

enum class LightType : uint32_t { directional = 0, point = 1, spot = 2 };

struct Motion {
    bool enabled = false;
    Vec3 center{};
    float radius = 0.0f;
    float speed = 0.0f;
    float phase = 0.0f;
};

struct SceneObject {
    uint32_t id = 0;
    Vec3 position{};
    float yaw = 0.0f;
    Vec3 scale{1.0f, 1.0f, 1.0f};
    uint32_t material = 0;
    Motion motion{};
    AABB bounds{};
};

struct Light {
    uint32_t id = 0;
    LightType type = LightType::directional;
    Vec3 position{};
    Vec3 direction{0.0f, -1.0f, 0.0f};
    Vec3 color{1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    float range = 20.0f;
    float innerCos = 0.95f;
    float outerCos = 0.85f;
    Motion motion{};
};

struct CameraConfig {
    float fovY = 60.0f * kPi / 180.0f;
    float aspect = 16.0f / 9.0f;
    float nearZ = 0.1f;
    float farZ = 320.0f;
    float orbitRadius = 38.0f;
    float orbitHeight = 18.0f;
    float targetHeight = 2.0f;
    float angularSpeed = 0.22f;
    float startAngle = -2.1f;
    uint32_t holdStartFrame = 45;
    uint32_t holdEndFrame = 90;
};

struct SceneDesc {
    uint32_t version = 1;
    uint64_t seed = 166;
    std::string name;
    std::vector<SceneObject> objects;
    std::vector<Light> lights;
    CameraConfig camera{};
};

struct GenerateOptions {
    uint32_t objectCount = 5000;
    uint64_t seed = 166;
};

[[nodiscard]] AABB localCubeBounds();
[[nodiscard]] Mat4 objectWorldMatrix(const SceneObject& object);
void updateObjectWorldBounds(SceneObject& object);
void updateMovingObjects(SceneDesc& scene, float timeSeconds);
void updateMovingLights(SceneDesc& scene, float timeSeconds);
[[nodiscard]] Camera cameraAtFrame(const CameraConfig& config, uint32_t frame, float frameSeconds);
[[nodiscard]] Camera cameraAtTime(const CameraConfig& config, float timeSeconds);

[[nodiscard]] SceneDesc generateScene(const GenerateOptions& options);
void writeSceneJson(const SceneDesc& scene, const std::filesystem::path& path);
[[nodiscard]] SceneDesc loadSceneJson(const std::filesystem::path& path);

[[nodiscard]] AABB lightInfluenceBounds(const Light& light);
[[nodiscard]] Mat4 lightViewProjection(const Light& light);

} // namespace uid166

