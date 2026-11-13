#pragma once

#include "math.hpp"

#include <cstdint>
#include <vector>

namespace dungeon {

enum class LightType {
    Point,
    Spot,
};

struct CameraConfig {
    Vec3 position{0.0, 2.05, -6.8};
    double yawDegrees = 0.0;
    double pitchDegrees = -7.0;
    double fovYDegrees = 62.0;
    double nearZ = 0.08;
    double farZ = 40.0;
};

struct Camera {
    Vec3 position;
    Basis basis;
    int width = 0;
    int height = 0;
    double fovYDegrees = 0.0;
    double aspect = 1.0;
    double nearZ = 0.0;
    double farZ = 0.0;
    double tanHalfY = 0.0;
    double tanHalfX = 0.0;

    Vec3 toView(const Vec3& world) const;
    Vec3 rayDirection(double pixelX, double pixelY, int width, int height) const;
    void project(const Vec3& view, double& screenX, double& screenY) const;
};

struct Triangle {
    Vec3 p0;
    Vec3 p1;
    Vec3 p2;
    Vec3 n0;
    Vec3 n1;
    Vec3 n2;
    uint32_t materialId = 0;
};

struct Light {
    uint32_t id = 0;
    LightType type = LightType::Point;
    Vec3 position;
    Vec3 direction{0.0, 0.0, 1.0};
    Vec3 color{1.0, 1.0, 1.0};
    double intensity = 1.0;
    double range = 10.0;
    double innerConeDegrees = 15.0;
    double outerConeDegrees = 30.0;
    bool castsShadow = false;

    Vec3 boundingSphereCenter() const;
    double boundingSphereRadius() const;
};

struct GeometryParams {
    int version = 0;
    double scale = 1.0;
};

struct Scene {
    std::vector<Triangle> triangles;
    std::vector<Light> lights;
    uint64_t geometryHash = 1469598103934665603ULL;
};

struct ValidationIssue {
    std::string code;
    std::string message;
};

bool validateCamera(const CameraConfig& config, int width, int height, ValidationIssue& issue);
Camera makeCamera(const CameraConfig& config, int width, int height);
bool validateGeometry(const GeometryParams& params, ValidationIssue& issue);
Scene buildDungeonScene(const GeometryParams& params, double lightShift);
uint64_t lightTransformHash(const Light& light);

}  // namespace dungeon
