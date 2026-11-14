#pragma once

#include "common.hpp"
#include "material.hpp"

#include <string>

namespace cr {

enum class MaterialZone : int {
    Skin = 0,
    Cloth = 1,
    Metal = 2,
    Hair = 3,
    Eyes = 4,
};

inline constexpr const char* kRequiredMaterialNames[] = {"skin", "cloth", "metal", "hair",
                                                         "eyes"};
inline constexpr int kMaterialZoneCount = 5;

struct SceneMapResult {
    float distance = 1.0e9F;
    MaterialZone zone = MaterialZone::Skin;
};

struct SurfaceMapping {
    std::string materialName;
    float u = 0.0F;
    float v = 0.0F;
    Vec3 tangent{1.0F, 0.0F, 0.0F};
    Vec3 bitangent{0.0F, 1.0F, 0.0F};
};

struct CameraFrame {
    Vec3 origin;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    float tanHalfFov = 1.0F;
    float aspect = 1.0F;
};

CameraFrame makeCameraFrame(const Camera& camera);
Vec3 rayDirection(const CameraFrame& frame, int x, int y, int width, int height);
float linearDepthFromPosition(const Camera& camera, const Vec3& position);
Vec3 positionFromLinearDepth(const CameraFrame& frame, const Vec3& rayDirection, float depth,
                             const Camera& camera);

SceneMapResult mapScene(const Vec3& position);
Vec3 sceneNormal(const Vec3& position, float epsilon = 0.0015F);
float traceScene(const Vec3& origin, const Vec3& direction, float maximumDistance,
                 MaterialZone* zone);
SurfaceMapping surfaceMapping(const Vec3& position, const Vec3& normal, MaterialZone zone);
Vec3 worldNormalFromTangent(const SurfaceMapping& mapping, const Vec3& tangentNormal);

}  // namespace cr

