#include "terrain/lod.hpp"

#include <algorithm>
#include <cmath>

namespace terrain {

Vec3 Camera::forward() const {
    const float yaw = radians(yawDegrees);
    const float pitch = radians(pitchDegrees);
    const float cosPitch = std::cos(pitch);
    return normalize({
        std::sin(yaw) * cosPitch,
        std::sin(pitch),
        std::cos(yaw) * cosPitch,
    });
}

Mat4 Camera::viewMatrix() const {
    const Vec3 f = forward();
    const Vec3 approximateUp{0.0f, 1.0f, 0.0f};
    Vec3 s = cross(f, approximateUp);
    if (lengthSquared(s) < 1.0e-8f) {
        s = {1.0f, 0.0f, 0.0f};
    }
    s = normalize(s);
    const Vec3 u = cross(s, f);

    Mat4 view = Mat4::identity();
    view(0, 0) = s.x;
    view(0, 1) = s.y;
    view(0, 2) = s.z;
    view(0, 3) = -dot(s, position);
    view(1, 0) = u.x;
    view(1, 1) = u.y;
    view(1, 2) = u.z;
    view(1, 3) = -dot(u, position);
    view(2, 0) = -f.x;
    view(2, 1) = -f.y;
    view(2, 2) = -f.z;
    view(2, 3) = dot(f, position);
    return view;
}

Mat4 Camera::projectionMatrix(float aspect) const {
    const float safeAspect = std::max(aspect, 1.0e-4f);
    const float tanHalfFov = std::tan(radians(fovYDegrees) * 0.5f);
    const float f = 1.0f / std::max(tanHalfFov, 1.0e-6f);

    Mat4 projection;
    projection(0, 0) = f / safeAspect;
    projection(1, 1) = f;
    projection(2, 2) = (farPlane + nearPlane) / (nearPlane - farPlane);
    projection(2, 3) =
        (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
    projection(3, 2) = -1.0f;
    return projection;
}

Mat4 Camera::viewProjectionMatrix(float aspect) const {
    return projectionMatrix(aspect) * viewMatrix();
}

Frustum Camera::frustum(float aspect) const {
    return Frustum::fromViewProjection(viewProjectionMatrix(aspect));
}

}  // namespace terrain

