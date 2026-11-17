#include "cv_math.h"

#include <cstring>

namespace cv {

Mat4 Mat4::identity() {
    Mat4 result;
    result.at(0, 0) = 1.0f;
    result.at(1, 1) = 1.0f;
    result.at(2, 2) = 1.0f;
    result.at(3, 3) = 1.0f;
    return result;
}

Mat4 Mat4::translation(Vec3 t) {
    Mat4 result = identity();
    result.at(0, 3) = t.x;
    result.at(1, 3) = t.y;
    result.at(2, 3) = t.z;
    return result;
}

Mat4 Mat4::scale(Vec3 s) {
    Mat4 result = identity();
    result.at(0, 0) = s.x;
    result.at(1, 1) = s.y;
    result.at(2, 2) = s.z;
    return result;
}

Mat4 Mat4::rotationX(float radians) {
    Mat4 result = identity();
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    result.at(1, 1) = c;
    result.at(1, 2) = -s;
    result.at(2, 1) = s;
    result.at(2, 2) = c;
    return result;
}

Mat4 Mat4::rotationY(float radians) {
    Mat4 result = identity();
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    result.at(0, 0) = c;
    result.at(0, 2) = s;
    result.at(2, 0) = -s;
    result.at(2, 2) = c;
    return result;
}

Mat4 Mat4::rotationZ(float radians) {
    Mat4 result = identity();
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    result.at(0, 0) = c;
    result.at(0, 1) = -s;
    result.at(1, 0) = s;
    result.at(1, 1) = c;
    return result;
}

Mat4 Mat4::lookAtLH(Vec3 eye, Vec3 target, Vec3 up) {
    const Vec3 forward = normalize(target - eye);
    const Vec3 right = normalize(cross(up, forward));
    const Vec3 cameraUp = cross(forward, right);

    Mat4 result = identity();
    result.at(0, 0) = right.x;
    result.at(0, 1) = right.y;
    result.at(0, 2) = right.z;
    result.at(0, 3) = -dot(right, eye);
    result.at(1, 0) = cameraUp.x;
    result.at(1, 1) = cameraUp.y;
    result.at(1, 2) = cameraUp.z;
    result.at(1, 3) = -dot(cameraUp, eye);
    result.at(2, 0) = forward.x;
    result.at(2, 1) = forward.y;
    result.at(2, 2) = forward.z;
    result.at(2, 3) = -dot(forward, eye);
    return result;
}

Mat4 Mat4::perspectiveD3D(float fovYRadians, float aspect, float nearZ, float farZ) {
    Mat4 result;
    const float yScale = 1.0f / std::tan(fovYRadians * 0.5f);
    const float xScale = yScale / aspect;
    result.at(0, 0) = xScale;
    result.at(1, 1) = yScale;
    result.at(2, 2) = farZ / (farZ - nearZ);
    result.at(2, 3) = -(nearZ * farZ) / (farZ - nearZ);
    result.at(3, 2) = 1.0f;
    return result;
}

Mat4 Mat4::orthographicD3D(float left, float right, float bottom, float top,
                           float nearZ, float farZ) {
    Mat4 result = identity();
    result.at(0, 0) = 2.0f / (right - left);
    result.at(1, 1) = 2.0f / (top - bottom);
    result.at(2, 2) = 1.0f / (farZ - nearZ);
    result.at(0, 3) = -(right + left) / (right - left);
    result.at(1, 3) = -(top + bottom) / (top - bottom);
    result.at(2, 3) = -nearZ / (farZ - nearZ);
    return result;
}

Mat4 Mat4::transpose() const {
    Mat4 result;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            result.at(c, r) = at(r, c);
        }
    }
    return result;
}

Mat4 Mat4::inverse() const {
    const float* a = m.data();
    float inv[16];

    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] +
             a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] -
             a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] +
             a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] -
              a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] -
             a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] +
             a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] -
             a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] +
              a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] +
             a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] -
             a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] +
              a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] -
              a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] -
             a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] +
             a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] -
              a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] +
              a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];

    const float det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    Mat4 result;
    if (std::abs(det) < 1.0e-12f) {
        return identity();
    }
    const float invDet = 1.0f / det;
    for (int i = 0; i < 16; ++i) {
        result.m[static_cast<size_t>(i)] = inv[i] * invDet;
    }
    return result;
}

Vec4 Mat4::transform(Vec4 v) const {
    return {
        at(0, 0) * v.x + at(0, 1) * v.y + at(0, 2) * v.z + at(0, 3) * v.w,
        at(1, 0) * v.x + at(1, 1) * v.y + at(1, 2) * v.z + at(1, 3) * v.w,
        at(2, 0) * v.x + at(2, 1) * v.y + at(2, 2) * v.z + at(2, 3) * v.w,
        at(3, 0) * v.x + at(3, 1) * v.y + at(3, 2) * v.z + at(3, 3) * v.w,
    };
}

Vec3 Mat4::transformPoint(Vec3 v) const {
    const Vec4 r = transform({v.x, v.y, v.z, 1.0f});
    return {r.x, r.y, r.z};
}

Vec3 Mat4::transformVector(Vec3 v) const {
    return {
        at(0, 0) * v.x + at(0, 1) * v.y + at(0, 2) * v.z,
        at(1, 0) * v.x + at(1, 1) * v.y + at(1, 2) * v.z,
        at(2, 0) * v.x + at(2, 1) * v.y + at(2, 2) * v.z,
    };
}

Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 result;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a.at(r, k) * b.at(k, c);
            }
            result.at(r, c) = sum;
        }
    }
    return result;
}

bool Frustum::intersects(const AABB& box) const {
    if (!box.valid()) {
        return false;
    }
    const Vec3 c = box.center();
    const Vec3 e = box.extent();
    for (const Plane& p : planes) {
        const float radius = std::abs(p.n.x) * e.x + std::abs(p.n.y) * e.y +
                             std::abs(p.n.z) * e.z;
        if (dot(p.n, c) + p.d + radius < 0.0f) {
            return false;
        }
    }
    return true;
}

Frustum extractFrustum(const Mat4& vp) {
    Frustum f;
    // D3D depth range [0, w]. Planes point toward the interior of the frustum.
    auto set = [&vp](Plane& p, int a, int b, float sa, float sb) {
        p.n = {
            vp.at(3, 0) + sa * vp.at(a, 0) + sb * vp.at(b, 0),
            vp.at(3, 1) + sa * vp.at(a, 1) + sb * vp.at(b, 1),
            vp.at(3, 2) + sa * vp.at(a, 2) + sb * vp.at(b, 2),
        };
        p.d = vp.at(3, 3) + sa * vp.at(a, 3) + sb * vp.at(b, 3);
        const float len = length(p.n);
        if (len > kEpsilon) {
            p.n = p.n / len;
            p.d /= len;
        }
    };
    set(f.planes[0], 0, 2, 1.0f, 0.0f);   // left:  w + x
    set(f.planes[1], 0, 2, -1.0f, 0.0f);  // right: w - x
    set(f.planes[2], 1, 2, 1.0f, 0.0f);   // bottom: w + y
    set(f.planes[3], 1, 2, -1.0f, 0.0f);  // top: w - y
    set(f.planes[5], 2, 3, 0.0f, -1.0f);  // far: w - z

    // Near: z >= 0. This is row 2 alone, unlike the w +/- row form above.
    f.planes[4].n = {vp.at(2, 0), vp.at(2, 1), vp.at(2, 2)};
    f.planes[4].d = vp.at(2, 3);
    const float nearLength = length(f.planes[4].n);
    if (nearLength > kEpsilon) {
        f.planes[4].n = f.planes[4].n / nearLength;
        f.planes[4].d /= nearLength;
    }
    return f;
}

Vec3 Camera::position() const {
    const float cp = std::cos(pitchRadians);
    return target + Vec3{std::sin(yawRadians) * cp, std::sin(pitchRadians),
                         std::cos(yawRadians) * cp} *
                        distance;
}

Vec3 Camera::forward() const { return normalize(target - position()); }

Mat4 Camera::view() const { return Mat4::lookAtLH(position(), target, {0.0f, 1.0f, 0.0f}); }

Mat4 Camera::projection(float aspect) const {
    return Mat4::perspectiveD3D(fovYRadians, aspect, nearZ, farZ);
}

Mat4 Camera::viewProjection(float aspect) const { return projection(aspect) * view(); }

float clamp01(float v) { return std::clamp(v, 0.0f, 1.0f); }

Vec3 clamp01(Vec3 v) {
    return {std::clamp(v.x, 0.0f, 1.0f), std::clamp(v.y, 0.0f, 1.0f),
            std::clamp(v.z, 0.0f, 1.0f)};
}

}  // namespace cv

