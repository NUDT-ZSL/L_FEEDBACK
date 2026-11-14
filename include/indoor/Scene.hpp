#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace indoor {

enum class MaterialPresetId : int {
    BaseOnly = 0,
    NormalDetail = 1,
    RoughMetal = 2,
    BaseNormal = 3,
    FullLayered = 4,
    Count = 5
};

const char* materialPresetName(MaterialPresetId preset);
MaterialPresetId nextMaterialPreset(MaterialPresetId preset);

struct Light {
    std::uint32_t id = 0;
    glm::vec3 position{0.0f};
    float range = 8.0f;
    glm::vec3 color{1.0f};
    float intensity = 1.0f;
};

struct ObjectTransform {
    glm::vec3 position{0.0f};
    glm::vec3 rotationEuler{0.0f};
    glm::vec3 halfExtents{1.0f};

    [[nodiscard]] glm::mat4 matrix() const;
    [[nodiscard]] float boundingRadius() const;
};

struct Motion {
    glm::vec3 amplitude{0.0f};
    glm::vec3 frequency{0.0f};
    glm::vec3 phase{0.0f};
};

struct MaterialParams {
    glm::vec3 albedo{0.7f};
    float roughness = 0.5f;
    float metallic = 0.0f;
    float uvScale = 1.0f;
    float normalStrength = 1.0f;
    std::uint32_t baseLayer = 0;
    std::uint32_t normalLayer = 3;
    std::uint32_t roughMetalLayer = 5;
};

struct SceneObject {
    std::uint32_t id = 0;
    bool dynamic = false;
    ObjectTransform initialTransform{};
    ObjectTransform transform{};
    Motion motion{};
    MaterialParams material{};
};

struct Scene {
    std::uint64_t seed = 0;
    std::uint64_t generation = 1;
    double animationTime = 0.0;
    std::vector<Light> lights;
    std::vector<SceneObject> objects;
};

struct MovedObject {
    std::uint32_t objectId = 0;
    glm::vec3 oldCenter{0.0f};
    glm::vec3 newCenter{0.0f};
    float boundingRadius = 0.0f;
};

struct Camera {
    glm::vec3 target{0.0f, 2.0f, 0.0f};
    float yaw = 0.70f;
    float pitch = 0.12f;
    float distance = 20.0f;

    [[nodiscard]] glm::vec3 position() const;
    [[nodiscard]] glm::mat4 view() const;
    [[nodiscard]] glm::mat4 projection(float aspect) const;
};

Scene buildDeterministicScene(std::uint64_t seed, std::uint32_t lightCount,
                              std::uint64_t generation = 1);

// Updates all dynamic objects from the scene's absolute deterministic animation
// clock. The returned objects describe exactly which transforms changed.
std::vector<MovedObject> setAnimationTime(Scene& scene, double seconds);
std::vector<MovedObject> advanceAnimation(Scene& scene, double deltaSeconds);

// Conservative swept-volume invalidation: a light is dirty only if the moved
// object's old/new bounding sphere path enters the light's range.
std::vector<std::uint32_t> findAffectedLights(
    const std::vector<Light>& lights,
    const std::vector<MovedObject>& movedObjects);

bool removeLight(Scene& scene, std::uint32_t lightId);
const Light* findLight(const Scene& scene, std::uint32_t lightId);

std::uint64_t sceneDigest(const Scene& scene);
std::string sceneDigestHex(const Scene& scene);

} // namespace indoor
