#include "scene.h"

#include <algorithm>
#include <cmath>

namespace df {
namespace {

Primitive makeSphere(int id, const Vec3& center, float radius,
                     const Material& material, bool transparent = false,
                     float opacity = 1.0f, const Vec3& tint = Vec3{1.0f, 1.0f, 1.0f}) {
    Primitive p;
    p.id = id;
    p.type = PrimitiveType::Sphere;
    p.center = center;
    p.radius = radius;
    p.material = material;
    p.transparent = transparent;
    p.opacity = opacity;
    p.tint = tint;
    return p;
}

Primitive makeBox(int id, const Vec3& center, const Vec3& halfExtents,
                  const Material& material) {
    Primitive p;
    p.id = id;
    p.type = PrimitiveType::Box;
    p.center = center;
    p.halfExtents = halfExtents;
    p.material = material;
    return p;
}

Primitive makeGround() {
    Primitive p;
    p.id = 1;
    p.type = PrimitiveType::Plane;
    p.center = {0.0f, 0.0f, 0.0f};
    p.material.albedo = {0.22f, 0.25f, 0.29f};
    p.material.roughness = 0.82f;
    p.material.metallic = 0.04f;
    return p;
}

bool intersectBox(const Vec3& origin, const Vec3& direction,
                  const Vec3& center, const Vec3& halfExtents, float& tOut) {
    float tMin = -std::numeric_limits<float>::infinity();
    float tMax = std::numeric_limits<float>::infinity();
    for (int axis = 0; axis < 3; ++axis) {
        const float o = (&origin.x)[axis];
        const float d = (&direction.x)[axis];
        const float c = (&center.x)[axis];
        const float h = (&halfExtents.x)[axis];
        if (std::fabs(d) < 1.0e-7f) {
            if (o < c - h || o > c + h) return false;
            continue;
        }
        const float inv = 1.0f / d;
        float t0 = (c - h - o) * inv;
        float t1 = (c + h - o) * inv;
        if (t0 > t1) std::swap(t0, t1);
        tMin = std::max(tMin, t0);
        tMax = std::min(tMax, t1);
        if (tMin > tMax) return false;
    }
    tOut = tMin >= 0.0f ? tMin : tMax;
    return tOut >= 0.0f;
}

} // namespace

Scene makeDeterministicScene(int width, int height) {
    Scene scene;
    scene.name = "deterministic_courtyard_v1";
    scene.camera.position = {0.0f, 1.72f, 6.4f};
    scene.camera.target = {0.0f, 1.05f, -2.4f};
    scene.camera.up = {0.0f, 1.0f, 0.0f};
    scene.camera.fovYDegrees = 52.0f;
    scene.camera.nearPlane = 0.1f;
    scene.camera.farPlane = 80.0f;
    (void)width;
    (void)height;

    scene.opaque.push_back(makeGround());

    Material slate;
    slate.albedo = {0.33f, 0.42f, 0.50f};
    slate.roughness = 0.46f;
    slate.metallic = 0.22f;
    scene.opaque.push_back(makeBox(2, {-3.15f, 1.13f, -2.7f}, {0.9f, 1.13f, 0.9f}, slate));

    Material sandstone;
    sandstone.albedo = {0.67f, 0.51f, 0.34f};
    sandstone.roughness = 0.68f;
    sandstone.metallic = 0.0f;
    scene.opaque.push_back(makeBox(3, {1.55f, 0.68f, -1.3f}, {1.15f, 0.68f, 0.85f}, sandstone));

    Material copper;
    copper.albedo = {0.52f, 0.25f, 0.13f};
    copper.roughness = 0.32f;
    copper.metallic = 0.85f;
    scene.opaque.push_back(makeBox(4, {3.75f, 1.15f, -4.2f}, {1.15f, 1.15f, 0.75f}, copper));

    Material pearl;
    pearl.albedo = {0.77f, 0.79f, 0.86f};
    pearl.roughness = 0.18f;
    pearl.metallic = 0.1f;
    scene.opaque.push_back(makeSphere(5, {-1.4f, 0.95f, -3.85f}, 0.95f, pearl));

    Material moss;
    moss.albedo = {0.29f, 0.46f, 0.28f};
    moss.roughness = 0.9f;
    moss.metallic = 0.0f;
    scene.opaque.push_back(makeSphere(6, {2.45f, 0.72f, -3.1f}, 0.72f, moss));

    Material glassBlue;
    glassBlue.albedo = {0.22f, 0.55f, 0.92f};
    glassBlue.roughness = 0.12f;
    glassBlue.metallic = 0.0f;
    scene.transparent.push_back(
        makeSphere(21, {-2.25f, 1.02f, 0.65f}, 0.72f, glassBlue, true, 0.42f, {0.34f, 0.68f, 1.0f}));

    Material glassAmber;
    glassAmber.albedo = {0.95f, 0.42f, 0.12f};
    glassAmber.roughness = 0.2f;
    scene.transparent.push_back(
        makeSphere(22, {1.05f, 1.12f, -0.45f}, 0.82f, glassAmber, true, 0.36f, {1.0f, 0.52f, 0.18f}));

    Material glassViolet;
    glassViolet.albedo = {0.58f, 0.2f, 0.95f};
    glassViolet.roughness = 0.16f;
    scene.transparent.push_back(
        makeSphere(23, {3.15f, 0.93f, 0.95f}, 0.68f, glassViolet, true, 0.38f, {0.61f, 0.3f, 1.0f}));

    Material glassMint;
    glassMint.albedo = {0.2f, 0.88f, 0.63f};
    glassMint.roughness = 0.22f;
    scene.transparent.push_back(
        makeSphere(24, {-0.8f, 1.55f, -6.2f}, 1.05f, glassMint, true, 0.3f, {0.18f, 1.0f, 0.7f}));

    scene.lights.push_back(
        Light{11, LightType::Point, {2.9f, 3.3f, 1.8f}, {0.0f, -1.0f, 0.0f},
              {1.0f, 0.45f, 0.22f}, 7.5f});
    scene.lights.push_back(
        Light{12, LightType::Point, {-2.8f, 2.5f, -1.3f}, {0.0f, -1.0f, 0.0f},
              {0.25f, 0.67f, 1.0f}, 6.0f});
    scene.lights.push_back(
        Light{13, LightType::Point, {0.2f, 2.8f, -4.8f}, {0.0f, -1.0f, 0.0f},
              {0.35f, 1.0f, 0.7f}, 4.5f});
    scene.lights.push_back(
        Light{14, LightType::Directional, {0.0f, 4.0f, 0.0f}, {-0.45f, -0.82f, 0.3f},
              {0.48f, 0.57f, 0.82f}, 1.15f});
    return scene;
}

Scene makeSortingStressScene(int transparentCount) {
    Scene scene = makeDeterministicScene(256, 144);
    scene.transparent.clear();
    for (int i = 0; i < transparentCount; ++i) {
        const float phase = static_cast<float>(i) * 0.61803398875f;
        const float x = (std::fmod(phase, 1.0f) * 2.0f - 1.0f) * 1.4f;
        const float y = 0.65f + (std::fmod(phase * 3.17f, 1.0f) * 0.45f);
        const float z = -2.5f - (static_cast<float>(i) / static_cast<float>(transparentCount)) * 6.0f;
        Material mat;
        mat.albedo = {0.2f + std::fmod(phase * 5.0f, 0.7f), 0.35f, 0.85f};
        mat.roughness = 0.2f;
        scene.transparent.push_back(
            makeSphere(1000 + i, {x, y, z}, 0.16f, mat, true, 0.16f,
                       {mat.albedo.x, mat.albedo.y, mat.albedo.z}));
    }
    return scene;
}

Ray cameraRay(const Camera& camera, int width, int height, int x, int y) {
    const Vec3 forward = normalize(camera.target - camera.position);
    const Vec3 right = normalize(cross(forward, camera.up));
    const Vec3 up = cross(right, forward);
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float tanHalf = std::tan(camera.fovYDegrees * kPi / 360.0f);
    const float sx = (2.0f * (static_cast<float>(x) + 0.5f) / static_cast<float>(width) - 1.0f) *
                     tanHalf * aspect;
    const float sy = (1.0f - 2.0f * (static_cast<float>(y) + 0.5f) / static_cast<float>(height)) *
                     tanHalf;
    return {camera.position, normalize(forward + right * sx + up * sy)};
}

Hit intersectPrimitive(const Primitive& primitive, const Ray& ray, float maxDistance) {
    Hit out;
    out.primitive = &primitive;
    if (primitive.type == PrimitiveType::Plane) {
        if (std::fabs(ray.direction.y) < 1.0e-7f) return out;
        const float t = (primitive.center.y - ray.origin.y) / ray.direction.y;
        if (t <= 0.0f || t >= maxDistance) return out;
        out.hit = true;
        out.t = t;
        out.position = ray.origin + ray.direction * t;
        out.normal = ray.direction.y < 0.0f ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{0.0f, -1.0f, 0.0f};
        return out;
    }
    if (primitive.type == PrimitiveType::Sphere) {
        const Vec3 oc = ray.origin - primitive.center;
        const float b = dot(oc, ray.direction);
        const float c = dot(oc, oc) - primitive.radius * primitive.radius;
        const float disc = b * b - c;
        if (disc < 0.0f) return out;
        const float root = std::sqrt(disc);
        float t = -b - root;
        if (t <= 0.0f) t = -b + root;
        if (t <= 0.0f || t >= maxDistance) return out;
        out.hit = true;
        out.t = t;
        out.position = ray.origin + ray.direction * t;
        out.normal = normalize(out.position - primitive.center);
        return out;
    }
    float t = 0.0f;
    if (!intersectBox(ray.origin, ray.direction, primitive.center,
                      primitive.halfExtents, t) || t >= maxDistance) {
        return out;
    }
    out.hit = true;
    out.t = t;
    out.position = ray.origin + ray.direction * t;
    const Vec3 local = (out.position - primitive.center) / primitive.halfExtents;
    const float ax = std::fabs(local.x);
    const float ay = std::fabs(local.y);
    const float az = std::fabs(local.z);
    if (ax > ay && ax > az) {
        out.normal = {local.x > 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f};
    } else if (ay > az) {
        out.normal = {0.0f, local.y > 0.0f ? 1.0f : -1.0f, 0.0f};
    } else {
        out.normal = {0.0f, 0.0f, local.z > 0.0f ? 1.0f : -1.0f};
    }
    return out;
}

Hit intersectOpaque(const Scene& scene, const Ray& ray, float maxDistance) {
    Hit nearest;
    for (const Primitive& primitive : scene.opaque) {
        Hit hit = intersectPrimitive(primitive, ray, maxDistance);
        if (hit.hit && hit.t < nearest.t) nearest = hit;
    }
    return nearest;
}

float viewDepth(const Camera& camera, const Vec3& worldPoint) {
    const Vec3 forward = normalize(camera.target - camera.position);
    return dot(worldPoint - camera.position, forward);
}

} // namespace df

