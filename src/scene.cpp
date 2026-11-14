#include "scene.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cr {
namespace {

constexpr float kPi = 3.14159265358979323846F;

float sdEllipsoid(const Vec3& position, const Vec3& center, const Vec3& radii) {
    const Vec3 local = position - center;
    const Vec3 scaled{local.x / radii.x, local.y / radii.y, local.z / radii.z};
    const Vec3 squaredScaled{scaled.x / radii.x, scaled.y / radii.y, scaled.z / radii.z};
    const float k0 = length(scaled);
    const float k1 = length(squaredScaled);
    if (k1 <= std::numeric_limits<float>::epsilon()) {
        return -std::min({radii.x, radii.y, radii.z});
    }
    return k0 * (k0 - 1.0F) / k1;
}

float sdCapsule(const Vec3& position, const Vec3& a, const Vec3& b, float radius) {
    const Vec3 pa = position - a;
    const Vec3 ba = b - a;
    const float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0F, 1.0F);
    return length(pa - ba * h) - radius;
}

float sdTorusY(const Vec3& position, const Vec3& center, float majorRadius,
               float minorRadius) {
    const Vec3 local = position - center;
    const float radial = std::sqrt(local.x * local.x + local.z * local.z) - majorRadius;
    return std::sqrt(radial * radial + local.y * local.y) - minorRadius;
}

void consider(SceneMapResult& result, float distance, MaterialZone zone) {
    if (distance < result.distance) {
        result.distance = distance;
        result.zone = zone;
    }
}

Vec3 tangentBasis(const Vec3& normal) {
    const Vec3 reference =
        std::abs(normal.y) < 0.92F ? Vec3(0.0F, 1.0F, 0.0F) : Vec3(1.0F, 0.0F, 0.0F);
    return normalize(cross(reference, normal));
}

}  // namespace

CameraFrame makeCameraFrame(const Camera& camera) {
    CameraFrame frame;
    frame.origin = camera.position;
    frame.forward = normalize(camera.target - camera.position);
    frame.right = normalize(cross(frame.forward, camera.up));
    frame.up = normalize(cross(frame.right, frame.forward));
    frame.tanHalfFov = std::tan(camera.fovYDegrees * kPi / 360.0F);
    frame.aspect =
        static_cast<float>(camera.width) / static_cast<float>(std::max(camera.height, 1));
    return frame;
}

Vec3 rayDirection(const CameraFrame& frame, int x, int y, int width, int height) {
    const float screenX =
        (2.0F * (static_cast<float>(x) + 0.5F) / static_cast<float>(width) - 1.0F) *
        frame.aspect * frame.tanHalfFov;
    const float screenY =
        (1.0F - 2.0F * (static_cast<float>(y) + 0.5F) / static_cast<float>(height)) *
        frame.tanHalfFov;
    return normalize(frame.forward + frame.right * screenX + frame.up * screenY);
}

float linearDepthFromPosition(const Camera& camera, const Vec3& position) {
    const CameraFrame frame = makeCameraFrame(camera);
    return dot(position - frame.origin, frame.forward);
}

Vec3 positionFromLinearDepth(const CameraFrame& frame, const Vec3& direction, float depth,
                             const Camera& camera) {
    const float forwardComponent = dot(direction, frame.forward);
    if (std::abs(forwardComponent) < 1.0e-6F) {
        return frame.origin;
    }
    const float distance = depth / forwardComponent;
    return frame.origin + direction * std::max(distance, camera.nearPlane);
}

SceneMapResult mapScene(const Vec3& position) {
    SceneMapResult result;
    consider(result,
             sdEllipsoid(position, Vec3(0.0F, 0.38F, 0.0F),
                         Vec3(0.64F, 0.76F, 0.60F)),
             MaterialZone::Skin);
    consider(result,
             sdEllipsoid(position, Vec3(0.0F, 0.69F, -0.04F),
                         Vec3(0.69F, 0.50F, 0.65F)),
             MaterialZone::Hair);
    consider(result,
             sdCapsule(position, Vec3(0.0F, -0.09F, 0.0F), Vec3(0.0F, -0.43F, 0.0F),
                       0.205F),
             MaterialZone::Skin);
    consider(result,
             sdEllipsoid(position, Vec3(0.0F, -1.04F, 0.0F),
                         Vec3(1.03F, 0.70F, 0.52F)),
             MaterialZone::Cloth);
    consider(result,
             sdCapsule(position, Vec3(-0.78F, -0.68F, 0.02F),
                       Vec3(-1.03F, -0.90F, 0.02F), 0.25F),
             MaterialZone::Cloth);
    consider(result,
             sdCapsule(position, Vec3(0.78F, -0.68F, 0.02F),
                       Vec3(1.03F, -0.90F, 0.02F), 0.25F),
             MaterialZone::Cloth);
    consider(result,
             sdEllipsoid(position, Vec3(-0.87F, -0.66F, 0.02F),
                         Vec3(0.38F, 0.29F, 0.34F)),
             MaterialZone::Metal);
    consider(result,
             sdEllipsoid(position, Vec3(0.87F, -0.66F, 0.02F),
                         Vec3(0.38F, 0.29F, 0.34F)),
             MaterialZone::Metal);
    consider(result,
             sdEllipsoid(position, Vec3(-0.24F, 0.45F, 0.555F),
                         Vec3(0.112F, 0.073F, 0.052F)),
             MaterialZone::Eyes);
    consider(result,
             sdEllipsoid(position, Vec3(0.24F, 0.45F, 0.555F),
                         Vec3(0.112F, 0.073F, 0.052F)),
             MaterialZone::Eyes);
    consider(result,
             sdEllipsoid(position, Vec3(0.0F, 0.31F, 0.575F),
                         Vec3(0.095F, 0.135F, 0.105F)),
             MaterialZone::Skin);
    consider(result,
             sdEllipsoid(position, Vec3(-0.62F, 0.38F, 0.0F),
                         Vec3(0.075F, 0.15F, 0.07F)),
             MaterialZone::Skin);
    consider(result,
             sdEllipsoid(position, Vec3(0.62F, 0.38F, 0.0F),
                         Vec3(0.075F, 0.15F, 0.07F)),
             MaterialZone::Skin);
    consider(result, sdTorusY(position, Vec3(0.0F, 0.67F, 0.0F), 0.69F, 0.045F),
             MaterialZone::Metal);
    consider(result,
             sdEllipsoid(position, Vec3(0.0F, -0.91F, 0.50F),
                         Vec3(0.17F, 0.21F, 0.065F)),
             MaterialZone::Metal);
    return result;
}

Vec3 sceneNormal(const Vec3& position, float epsilon) {
    const Vec3 dx =
        Vec3(mapScene(position + Vec3(epsilon, 0.0F, 0.0F)).distance,
             mapScene(position + Vec3(0.0F, epsilon, 0.0F)).distance,
             mapScene(position + Vec3(0.0F, 0.0F, epsilon)).distance);
    const Vec3 dy =
        Vec3(mapScene(position - Vec3(epsilon, 0.0F, 0.0F)).distance,
             mapScene(position - Vec3(0.0F, epsilon, 0.0F)).distance,
             mapScene(position - Vec3(0.0F, 0.0F, epsilon)).distance);
    return normalize(dx - dy);
}

float traceScene(const Vec3& origin, const Vec3& direction, float maximumDistance,
                 MaterialZone* zone) {
    float distanceAlongRay = 0.5F;
    for (int step = 0; step < 170; ++step) {
        const Vec3 position = origin + direction * distanceAlongRay;
        const SceneMapResult mapped = mapScene(position);
        if (mapped.distance < 0.0007F) {
            if (zone != nullptr) {
                *zone = mapped.zone;
            }
            return distanceAlongRay;
        }
        distanceAlongRay += std::max(mapped.distance * 0.62F, 0.0008F);
        if (distanceAlongRay > maximumDistance) {
            break;
        }
    }
    return -1.0F;
}

SurfaceMapping surfaceMapping(const Vec3& position, const Vec3& normal, MaterialZone zone) {
    SurfaceMapping mapping;
    mapping.materialName = kRequiredMaterialNames[static_cast<int>(zone)];
    const Vec3 tangent = tangentBasis(normal);
    mapping.tangent = tangent;
    mapping.bitangent = normalize(cross(normal, tangent));

    switch (zone) {
        case MaterialZone::Skin:
        case MaterialZone::Hair:
        case MaterialZone::Eyes: {
            if (zone == MaterialZone::Eyes) {
                const float centerX = position.x < 0.0F ? -0.24F : 0.24F;
                mapping.u = clamp(0.5F + (position.x - centerX) / 0.224F, 0.0F, 1.0F);
                mapping.v = clamp(0.5F - (position.y - 0.45F) / 0.146F, 0.0F, 1.0F);
            } else {
                mapping.u =
                    std::atan2(normal.x, normal.z) / (2.0F * kPi) + 0.5F;
                mapping.v = clamp((position.y - -0.38F) / 1.52F, 0.0F, 1.0F);
            }
            break;
        }
        case MaterialZone::Cloth:
        case MaterialZone::Metal: {
            mapping.u = std::atan2(position.x, position.z) / (2.0F * kPi) + 0.5F;
            mapping.v = clamp((position.y - -1.76F) / 2.40F, 0.0F, 1.0F);
            break;
        }
    }
    return mapping;
}

Vec3 worldNormalFromTangent(const SurfaceMapping& mapping, const Vec3& tangentNormal) {
    return normalize(mapping.tangent * tangentNormal.x +
                     mapping.bitangent * tangentNormal.y +
                     normalize(cross(mapping.tangent, mapping.bitangent)) * tangentNormal.z);
}

}  // namespace cr

