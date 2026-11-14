#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "config.h"
#include "mesh.h"

struct Material {
    glm::vec3 albedo{0.7f};
    float roughness = 0.65f;
    float metallic = 0.0f;
};

enum class MeshKind : int {
    Plane = 0,
    Cube = 1,
    Sphere = 2,
    Cylinder = 3
};

struct SceneObject {
    MeshKind mesh = MeshKind::Cube;
    glm::mat4 transform{1.0f};
    Material material;
    bool castsShadow = true;
    bool receivesShadow = true;
    std::string name;
};

struct DirectionalLight {
    glm::vec3 direction{0.42f, -1.0f, 0.55f};
    glm::vec3 color{1.0f, 0.92f, 0.78f};
    float intensity = 1.55f;
    float sourceDiameter = 0.22f;
    bool enabled = true;
    bool castsShadow = true;
};

struct PointLight {
    glm::vec3 position{0.0f};
    glm::vec3 color{1.0f};
    float intensity = 20.0f;
    float sourceDiameter = 0.25f;
    float range = 24.0f;
    bool enabled = true;
    bool castsShadow = true;
    glm::vec3 animationCenter{0.0f};
    float animationRadius = 0.0f;
    float animationSpeed = 0.0f;
    float animationPhase = 0.0f;
};

class LightManager {
public:
    DirectionalLight directional;
    std::array<PointLight, kMaxPointLights> points;

    LightManager();

    int activePointCount() const;
    int enabledPointCount() const;
    void togglePoint(int index);
    void setPointEnabled(int index, bool enabled);
    void setAllPointAnimation(bool enabled);
    void update(float elapsedSeconds, bool animate);
    void reset();

private:
    std::array<PointLight, kMaxPointLights> initialPoints_{};
    bool animationEnabled_ = true;
    float animationTime_ = 0.0f;
};

class Scene {
public:
    Scene();

    void initializeGpu();
    void destroyGpu();
    void buildDeterministic(unsigned int seed);

    const std::vector<SceneObject>& objects() const { return objects_; }
    unsigned int seed() const { return seed_; }
    float groundExtent() const { return groundExtent_; }

    const GpuMesh& mesh(MeshKind kind) const;

private:
    std::vector<SceneObject> objects_;
    GpuMesh plane_;
    GpuMesh cube_;
    GpuMesh sphere_;
    GpuMesh cylinder_;
    unsigned int seed_ = kDefaultSeed;
    float groundExtent_ = 26.0f;
};

