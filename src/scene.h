#pragma once

#include "config.h"
#include "math.h"

#include <cstdint>
#include <string>
#include <vector>

namespace df {

enum class PrimitiveType {
    Plane,
    Box,
    Sphere,
};

struct Material {
    Vec3 albedo{0.7f, 0.7f, 0.7f};
    float roughness = 0.7f;
    float metallic = 0.0f;
};

struct Primitive {
    int id = 0;
    PrimitiveType type = PrimitiveType::Sphere;
    Vec3 center{};
    Vec3 halfExtents{1.0f, 1.0f, 1.0f};
    float radius = 1.0f;
    Material material;
    bool transparent = false;
    float opacity = 1.0f;
    Vec3 tint{1.0f, 1.0f, 1.0f};
};

enum class LightType {
    Point,
    Directional,
};

struct Light {
    int id = 0;
    LightType type = LightType::Point;
    Vec3 position{};
    Vec3 direction{0.0f, -1.0f, 0.0f};
    Vec3 color{1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
};

struct Camera {
    Vec3 position{};
    Vec3 target{};
    Vec3 up{0.0f, 1.0f, 0.0f};
    float fovYDegrees = 52.0f;
    float nearPlane = 0.1f;
    float farPlane = 80.0f;
};

struct Scene {
    std::string name;
    Camera camera;
    std::vector<Primitive> opaque;
    std::vector<Primitive> transparent;
    std::vector<Light> lights;
};

Scene makeDeterministicScene(int width, int height);
Scene makeSortingStressScene(int transparentCount);

struct Ray {
    Vec3 origin;
    Vec3 direction;
};

struct Hit {
    bool hit = false;
    float t = std::numeric_limits<float>::max();
    Vec3 position{};
    Vec3 normal{};
    const Primitive* primitive = nullptr;
};

Ray cameraRay(const Camera& camera, int width, int height, int x, int y);
Hit intersectOpaque(const Scene& scene, const Ray& ray, float maxDistance);
Hit intersectPrimitive(const Primitive& primitive, const Ray& ray, float maxDistance);
float viewDepth(const Camera& camera, const Vec3& worldPoint);

} // namespace df

