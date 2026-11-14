#include "scene.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace {

class DeterministicRandom {
public:
    explicit DeterministicRandom(uint32_t seed) : state_(seed == 0 ? 1u : seed) {}

    uint32_t nextUint() {
        state_ ^= state_ << 13;
        state_ ^= state_ >> 17;
        state_ ^= state_ << 5;
        return state_;
    }

    float uniform(float minimum, float maximum) {
        const float unit = static_cast<float>(nextUint() & 0x00FFFFFFu) /
                           static_cast<float>(0x01000000u);
        return minimum + (maximum - minimum) * unit;
    }

private:
    uint32_t state_;
};

}  // namespace

LightManager::LightManager() {
    points[0].position = {4.6f, 4.3f, 1.1f};
    points[0].color = {1.0f, 0.36f, 0.16f};
    points[0].intensity = 34.0f;
    points[0].sourceDiameter = 0.25f;
    points[0].range = 24.0f;
    points[0].animationCenter = points[0].position;
    points[0].animationRadius = 1.8f;
    points[0].animationSpeed = 0.22f;
    points[0].animationPhase = 0.4f;

    points[1].position = {-4.7f, 3.1f, -3.8f};
    points[1].color = {0.22f, 0.55f, 1.0f};
    points[1].intensity = 28.0f;
    points[1].sourceDiameter = 0.25f;
    points[1].range = 22.0f;
    points[1].animationCenter = points[1].position;
    points[1].animationRadius = 1.3f;
    points[1].animationSpeed = -0.17f;
    points[1].animationPhase = 2.1f;

    initialPoints_ = points;
}

int LightManager::activePointCount() const {
    int count = 0;
    for (const PointLight& light : points) {
        count += (light.enabled && light.castsShadow) ? 1 : 0;
    }
    return count;
}

int LightManager::enabledPointCount() const {
    int count = 0;
    for (const PointLight& light : points) {
        count += light.enabled ? 1 : 0;
    }
    return count;
}

void LightManager::togglePoint(int index) {
    if (index < 0 || index >= kMaxPointLights) {
        return;
    }
    points[index].enabled = !points[index].enabled;
}

void LightManager::setPointEnabled(int index, bool enabled) {
    if (index >= 0 && index < kMaxPointLights) {
        points[index].enabled = enabled;
    }
}

void LightManager::setAllPointAnimation(bool enabled) {
    animationEnabled_ = enabled;
}

void LightManager::update(float elapsedSeconds, bool animate) {
    if (!animate || !animationEnabled_) {
        return;
    }
    animationTime_ += elapsedSeconds;
    for (PointLight& light : points) {
        const float angle = light.animationPhase + animationTime_ * light.animationSpeed;
        light.position.x = light.animationCenter.x + std::cos(angle) * light.animationRadius;
        light.position.z = light.animationCenter.z + std::sin(angle) * light.animationRadius;
        light.position.y = light.animationCenter.y + std::sin(angle * 1.7f) * 0.25f;
    }
}

void LightManager::reset() {
    points = initialPoints_;
    animationEnabled_ = true;
    animationTime_ = 0.0f;
}

Scene::Scene() {
    buildDeterministic(kDefaultSeed);
}

void Scene::initializeGpu() {
    plane_.create(makePlaneMesh(groundExtent_, 32));
    cube_.create(makeCubeMesh());
    sphere_.create(makeSphereMesh(40, 20));
    cylinder_.create(makeCylinderMesh(40, 1.0f, 1.0f));
}

void Scene::destroyGpu() {
    plane_.destroy();
    cube_.destroy();
    sphere_.destroy();
    cylinder_.destroy();
}

void Scene::buildDeterministic(unsigned int seed) {
    seed_ = seed;
    objects_.clear();
    DeterministicRandom rng(seed);

    SceneObject ground;
    ground.mesh = MeshKind::Plane;
    ground.transform = glm::mat4(1.0f);
    ground.material.albedo = {0.42f, 0.45f, 0.47f};
    ground.material.roughness = 0.9f;
    ground.castsShadow = false;
    ground.name = "ground";
    objects_.push_back(ground);

    struct Placement {
        MeshKind kind;
        glm::vec3 position;
        glm::vec3 scale;
        Material material;
        const char* name;
    };

    const std::array<Placement, 9> placements = {{
        {MeshKind::Cube, {-5.2f, 1.15f, 2.4f}, {2.1f, 2.3f, 2.1f},
         {{0.72f, 0.27f, 0.20f}, 0.48f, 0.05f}, "tall_box"},
        {MeshKind::Cube, {2.8f, 0.72f, -1.2f}, {1.45f, 1.45f, 1.45f},
         {{0.20f, 0.52f, 0.72f}, 0.55f, 0.0f}, "blue_box"},
        {MeshKind::Cube, {-1.0f, 2.25f, -5.0f}, {1.2f, 4.5f, 1.2f},
         {{0.70f, 0.62f, 0.25f}, 0.38f, 0.1f}, "pillar"},
        {MeshKind::Sphere, {-1.3f, 1.25f, 1.0f}, {1.25f, 1.25f, 1.25f},
         {{0.87f, 0.77f, 0.60f}, 0.25f, 0.0f}, "ivory_sphere"},
        {MeshKind::Sphere, {6.0f, 1.55f, 3.8f}, {1.55f, 1.55f, 1.55f},
         {{0.42f, 0.68f, 0.38f}, 0.33f, 0.0f}, "green_sphere"},
        {MeshKind::Cylinder, {2.4f, 1.05f, 4.2f}, {0.9f, 1.05f, 0.9f},
         {{0.62f, 0.35f, 0.78f}, 0.42f, 0.08f}, "violet_cylinder"},
        {MeshKind::Cylinder, {-7.2f, 1.5f, -3.6f}, {0.7f, 1.5f, 0.7f},
         {{0.25f, 0.75f, 0.70f}, 0.35f, 0.15f}, "teal_cylinder"},
        {MeshKind::Cube, {5.2f, 0.45f, -4.8f}, {2.7f, 0.9f, 1.4f},
         {{0.73f, 0.58f, 0.46f}, 0.65f, 0.0f}, "low_bench"},
        {MeshKind::Cube, {-3.8f, 0.55f, 7.2f}, {1.0f, 1.1f, 1.0f},
         {{0.68f, 0.30f, 0.50f}, 0.52f, 0.0f}, "pink_cube"},
    }};

    for (const Placement& placement : placements) {
        SceneObject object;
        object.mesh = placement.kind;
        object.material = placement.material;
        object.material.albedo +=
            glm::vec3(rng.uniform(-0.35f, 0.35f) * 0.05f) +
            glm::vec3(rng.uniform(0.0f, 1.0f) * 0.015f);
        object.name = placement.name;

        glm::mat4 transform = glm::translate(glm::mat4(1.0f), placement.position);
        transform = glm::rotate(transform,
                                placement.kind == MeshKind::Cube ? glm::radians(10.0f)
                                                                : glm::radians(0.0f),
                                glm::vec3(0, 1, 0));
        transform = glm::scale(transform, placement.scale);
        object.transform = transform;
        objects_.push_back(object);
    }
}

const GpuMesh& Scene::mesh(MeshKind kind) const {
    switch (kind) {
        case MeshKind::Plane: return plane_;
        case MeshKind::Cube: return cube_;
        case MeshKind::Sphere: return sphere_;
        case MeshKind::Cylinder: return cylinder_;
    }
    return cube_;
}

