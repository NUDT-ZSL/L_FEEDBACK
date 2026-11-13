#include "scene.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace dungeon {
namespace {

constexpr double kDegToRad = kPi / 180.0;

Triangle makeTriangle(
    const Vec3& p0,
    const Vec3& p1,
    const Vec3& p2,
    uint32_t materialId) {
    const Vec3 normal = normalize(cross(p1 - p0, p2 - p0));
    return {p0, p1, p2, normal, normal, normal, materialId};
}

void addQuad(
    std::vector<Triangle>& triangles,
    const Vec3& a,
    const Vec3& b,
    const Vec3& c,
    const Vec3& d,
    uint32_t materialId) {
    triangles.push_back(makeTriangle(a, b, c, materialId));
    triangles.push_back(makeTriangle(a, c, d, materialId));
}

void addBox(
    std::vector<Triangle>& triangles,
    const Vec3& center,
    const Vec3& halfExtent,
    uint32_t materialId) {
    const double minX = center.x - halfExtent.x;
    const double maxX = center.x + halfExtent.x;
    const double minY = center.y - halfExtent.y;
    const double maxY = center.y + halfExtent.y;
    const double minZ = center.z - halfExtent.z;
    const double maxZ = center.z + halfExtent.z;

    const Vec3 p000{minX, minY, minZ};
    const Vec3 p001{minX, minY, maxZ};
    const Vec3 p010{minX, maxY, minZ};
    const Vec3 p011{minX, maxY, maxZ};
    const Vec3 p100{maxX, minY, minZ};
    const Vec3 p101{maxX, minY, maxZ};
    const Vec3 p110{maxX, maxY, minZ};
    const Vec3 p111{maxX, maxY, maxZ};

    addQuad(triangles, p001, p101, p111, p011, materialId);  // +Z
    addQuad(triangles, p100, p000, p010, p110, materialId);  // -Z
    addQuad(triangles, p101, p100, p110, p111, materialId);  // +X
    addQuad(triangles, p000, p001, p011, p010, materialId);  // -X
    addQuad(triangles, p011, p111, p110, p010, materialId);  // +Y
    addQuad(triangles, p000, p100, p101, p001, materialId);  // -Y
}

}  // namespace

Vec3 Camera::toView(const Vec3& world) const {
    const Vec3 delta = world - position;
    return {
        dot(delta, basis.right),
        dot(delta, basis.up),
        dot(delta, basis.forward),
    };
}

Vec3 Camera::rayDirection(double pixelX, double pixelY, int width, int height) const {
    const double sx = (2.0 * pixelX / static_cast<double>(width)) - 1.0;
    const double sy = 1.0 - (2.0 * pixelY / static_cast<double>(height));
    return normalize(
        basis.forward +
        basis.right * (sx * tanHalfX) +
        basis.up * (sy * tanHalfY));
}

void Camera::project(const Vec3& view, double& screenX, double& screenY) const {
    const double invZ = 1.0 / view.z;
    const double ndcX = (view.x * invZ) / tanHalfX;
    const double ndcY = (view.y * invZ) / tanHalfY;
    screenX = (ndcX * 0.5 + 0.5) * static_cast<double>(width);
    screenY = (0.5 - ndcY * 0.5) * static_cast<double>(height);
}

Vec3 Light::boundingSphereCenter() const {
    return position;
}

double Light::boundingSphereRadius() const {
    return std::max(0.0, range);
}

bool validateCamera(const CameraConfig& config, int width, int height, ValidationIssue& issue) {
    if (width <= 0 || height <= 0) {
        issue = {"E_RESOLUTION_INVALID", "width and height must both be positive"};
        return false;
    }
    constexpr int64_t kMaxPixelCount = 2048LL * 2048LL;
    if (static_cast<int64_t>(width) * static_cast<int64_t>(height) > kMaxPixelCount) {
        issue = {"E_RESOLUTION_INVALID", "resolution exceeds the 2048x2048 offline validation budget"};
        return false;
    }
    if (!std::isfinite(config.fovYDegrees) ||
        config.fovYDegrees <= 0.0 ||
        config.fovYDegrees >= 179.0) {
        issue = {"E_CAMERA_INVALID", "vertical field of view must be in (0, 179) degrees"};
        return false;
    }
    if (!std::isfinite(config.nearZ) || config.nearZ <= 0.0 ||
        !std::isfinite(config.farZ) || config.farZ <= config.nearZ) {
        issue = {"E_CAMERA_INVALID", "near/far planes must satisfy 0 < near < far"};
        return false;
    }
    if (!std::isfinite(config.position.x) || !std::isfinite(config.position.y) ||
        !std::isfinite(config.position.z) || !std::isfinite(config.yawDegrees) ||
        !std::isfinite(config.pitchDegrees) || std::abs(config.pitchDegrees) >= 90.0) {
        issue = {"E_CAMERA_INVALID", "camera position/yaw/pitch is not finite or pitch is degenerate"};
        return false;
    }
    return true;
}

Camera makeCamera(const CameraConfig& config, int width, int height) {
    Camera camera;
    camera.position = config.position;
    camera.width = width;
    camera.height = height;
    camera.fovYDegrees = config.fovYDegrees;
    camera.aspect = static_cast<double>(width) / static_cast<double>(height);
    camera.nearZ = config.nearZ;
    camera.farZ = config.farZ;
    camera.tanHalfY = std::tan(config.fovYDegrees * kDegToRad * 0.5);
    camera.tanHalfX = camera.tanHalfY * camera.aspect;

    const double yaw = config.yawDegrees * kDegToRad;
    const double pitch = config.pitchDegrees * kDegToRad;
    const Vec3 forward{
        std::sin(yaw) * std::cos(pitch),
        std::sin(pitch),
        std::cos(yaw) * std::cos(pitch),
    };
    camera.basis = makeBasis(forward);
    return camera;
}

bool validateGeometry(const GeometryParams& params, ValidationIssue& issue) {
    if (params.version < 0) {
        issue = {"E_GEOMETRY_INVALID", "geometry version must be non-negative"};
        return false;
    }
    if (!std::isfinite(params.scale) || params.scale <= 0.0 || params.scale > 8.0) {
        issue = {"E_GEOMETRY_INVALID", "geometry scale must be in (0, 8]"};
        return false;
    }
    return true;
}

Scene buildDungeonScene(const GeometryParams& params, double lightShift) {
    Scene scene;
    const bool changedVariant = (params.version % 2) != 0;
    const double scale = params.scale;

    auto point = [scale](double x, double y, double z) {
        return Vec3{x * scale, y, z * scale};
    };

    // Room shell. Winding is CCW when viewed from the interior.
    addQuad(
        scene.triangles,
        point(-8.0, 0.0, -4.0),
        point(-8.0, 0.0, 11.0),
        point(8.0, 0.0, 11.0),
        point(8.0, 0.0, -4.0),
        1);
    addQuad(
        scene.triangles,
        point(-8.0, 0.0, 11.0),
        point(-8.0, 5.5, 11.0),
        point(8.0, 5.5, 11.0),
        point(8.0, 0.0, 11.0),
        2);
    addQuad(
        scene.triangles,
        point(-8.0, 0.0, -4.0),
        point(-8.0, 5.5, -4.0),
        point(-8.0, 5.5, 11.0),
        point(-8.0, 0.0, 11.0),
        2);
    addQuad(
        scene.triangles,
        point(8.0, 0.0, -4.0),
        point(8.0, 0.0, 11.0),
        point(8.0, 5.5, 11.0),
        point(8.0, 5.5, -4.0),
        2);
    addQuad(
        scene.triangles,
        point(-8.0, 5.5, -4.0),
        point(-8.0, 5.5, 11.0),
        point(8.0, 5.5, 11.0),
        point(8.0, 5.5, -4.0),
        2);

    const Vec3 pillar1Center =
        changedVariant ? point(-1.95, 1.65, 2.55) : point(-2.45, 1.65, 2.25);
    const Vec3 pillar1Half{0.48 * scale, 1.65, 0.48 * scale};
    addBox(scene.triangles, pillar1Center, pillar1Half, 3);

    const Vec3 pillar2Center = point(2.35, 1.25, 4.45);
    const Vec3 pillar2Half{0.62 * scale, 1.25, 0.62 * scale};
    addBox(scene.triangles, pillar2Center, pillar2Half, 3);

    if (changedVariant) {
        const Vec3 crateCenter = point(0.25, 0.55, 3.25);
        const Vec3 crateHalf{0.72 * scale, 0.55, 0.72 * scale};
        addBox(scene.triangles, crateCenter, crateHalf, 4);
    }

    Light warmSpot;
    warmSpot.id = 1;
    warmSpot.type = LightType::Spot;
    warmSpot.position = point(-2.8, 3.9, 0.3);
    warmSpot.direction = normalize(Vec3{2.7, -3.7, 4.0});
    warmSpot.color = {1.0, 0.72, 0.42};
    warmSpot.intensity = 6.8;
    warmSpot.range = 4.5;
    warmSpot.innerConeDegrees = 15.0;
    warmSpot.outerConeDegrees = 34.0;
    warmSpot.castsShadow = true;

    Light pointLight;
    pointLight.id = 2;
    pointLight.type = LightType::Point;
    pointLight.position = point(3.0 + lightShift, 2.25, 1.5 + lightShift * 0.35);
    pointLight.direction = {0.0, -1.0, 0.0};
    pointLight.color = {0.38, 0.78, 1.0};
    pointLight.intensity = 5.4;
    pointLight.range = 4.5;
    pointLight.castsShadow = true;

    Light coolSpot;
    coolSpot.id = 3;
    coolSpot.type = LightType::Spot;
    coolSpot.position = point(0.55, 2.8, 1.8);
    coolSpot.direction = normalize(Vec3{-0.35, -2.8, 3.0});
    coolSpot.color = {0.58, 0.66, 1.0};
    coolSpot.intensity = 8.0;
    coolSpot.range = 4.5;
    coolSpot.innerConeDegrees = 10.0;
    coolSpot.outerConeDegrees = 29.0;
    coolSpot.castsShadow = true;

    Light filler;
    filler.id = 4;
    filler.type = LightType::Point;
    filler.position = point(-4.6, 1.45, 6.2);
    filler.direction = {0.0, -1.0, 0.0};
    filler.color = {1.0, 0.35, 0.18};
    filler.intensity = 2.7;
    filler.range = 2.8;
    filler.castsShadow = false;

    scene.lights = {warmSpot, pointLight, coolSpot, filler};
    std::sort(
        scene.lights.begin(),
        scene.lights.end(),
        [](const Light& a, const Light& b) { return a.id < b.id; });

    uint64_t hash = 1469598103934665603ULL;
    for (const Triangle& triangle : scene.triangles) {
        hash = hashVec3(hash, triangle.p0);
        hash = hashVec3(hash, triangle.p1);
        hash = hashVec3(hash, triangle.p2);
        hash = hashVec3(hash, triangle.n0);
        hash = hashVec3(hash, triangle.n1);
        hash = hashVec3(hash, triangle.n2);
        hash = fnv1aU32(hash, triangle.materialId);
    }
    scene.geometryHash = hash;
    return scene;
}

uint64_t lightTransformHash(const Light& light) {
    uint64_t hash = 1469598103934665603ULL;
    hash = fnv1aU32(hash, static_cast<uint32_t>(light.type));
    hash = hashVec3(hash, light.position);
    hash = hashVec3(hash, light.direction);
    hash = hashDouble(hash, light.range);
    hash = hashDouble(hash, light.innerConeDegrees);
    hash = hashDouble(hash, light.outerConeDegrees);
    hash = fnv1aByte(hash, light.castsShadow ? 1u : 0u);
    return hash;
}

}  // namespace dungeon
