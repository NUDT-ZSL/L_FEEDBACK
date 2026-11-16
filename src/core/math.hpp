#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace uid166 {

constexpr float kPi = 3.14159265358979323846f;

inline bool finite(float v) { return std::isfinite(v); }

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3 operator+(const Vec3& r) const { return {x + r.x, y + r.y, z + r.z}; }
    Vec3 operator-(const Vec3& r) const { return {x - r.x, y - r.y, z - r.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    Vec3& operator+=(const Vec3& r) { x += r.x; y += r.y; z += r.z; return *this; }
};

inline Vec3 operator*(float s, const Vec3& v) { return v * s; }
inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(const Vec3& v) { return std::sqrt(dot(v, v)); }
inline Vec3 normalize(const Vec3& v) {
    const float l = length(v);
    return l > 1.0e-8f ? v / l : Vec3{0.0f, 0.0f, 1.0f};
}
inline Vec3 minVec(const Vec3& a, const Vec3& b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}
inline Vec3 maxVec(const Vec3& a, const Vec3& b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}
inline float maxComponent(const Vec3& v) { return std::max(v.x, std::max(v.y, v.z)); }
inline Vec3 lerp(const Vec3& a, const Vec3& b, float t) { return a + (b - a) * t; }

struct Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

struct Quat {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

inline Quat normalizeQuat(Quat q) {
    const float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (n < 1.0e-8f) return {};
    q.x /= n;
    q.y /= n;
    q.z /= n;
    q.w /= n;
    return q;
}

inline Quat slerp(Quat a, Quat b, float t) {
    float d = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    if (d < 0.0f) {
        b = {-b.x, -b.y, -b.z, -b.w};
        d = -d;
    }
    if (d > 0.9995f) {
        return normalizeQuat({a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
                              a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t});
    }
    const float theta0 = std::acos(std::clamp(d, -1.0f, 1.0f));
    const float theta = theta0 * t;
    const float sinTheta = std::sin(theta);
    const float sinTheta0 = std::sin(theta0);
    const float s0 = std::cos(theta) - d * sinTheta / sinTheta0;
    const float s1 = sinTheta / sinTheta0;
    return normalizeQuat({a.x * s0 + b.x * s1, a.y * s0 + b.y * s1,
                          a.z * s0 + b.z * s1, a.w * s0 + b.w * s1});
}

struct Mat4 {
    std::array<float, 16> m{1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1};

    float& at(int row, int col) { return m[static_cast<size_t>(row * 4 + col)]; }
    float at(int row, int col) const { return m[static_cast<size_t>(row * 4 + col)]; }

    static Mat4 identity() { return {}; }
    static Mat4 zero() {
        Mat4 r;
        r.m.fill(0.0f);
        return r;
    }
    static Mat4 translation(const Vec3& t) {
        Mat4 r;
        r.at(3, 0) = t.x;
        r.at(3, 1) = t.y;
        r.at(3, 2) = t.z;
        return r;
    }
    static Mat4 scale(const Vec3& s) {
        Mat4 r;
        r.at(0, 0) = s.x;
        r.at(1, 1) = s.y;
        r.at(2, 2) = s.z;
        return r;
    }
    static Mat4 rotationX(float a) {
        Mat4 r;
        const float c = std::cos(a);
        const float s = std::sin(a);
        r.at(1, 1) = c; r.at(1, 2) = s;
        r.at(2, 1) = -s; r.at(2, 2) = c;
        return r;
    }
    static Mat4 rotationY(float a) {
        Mat4 r;
        const float c = std::cos(a);
        const float s = std::sin(a);
        r.at(0, 0) = c; r.at(0, 2) = -s;
        r.at(2, 0) = s; r.at(2, 2) = c;
        return r;
    }
    static Mat4 rotationZ(float a) {
        Mat4 r;
        const float c = std::cos(a);
        const float s = std::sin(a);
        r.at(0, 0) = c; r.at(0, 1) = s;
        r.at(1, 0) = -s; r.at(1, 1) = c;
        return r;
    }
    static Mat4 rotationQuat(const Quat& q) {
        const Quat n = normalizeQuat(q);
        const float xx = n.x * n.x, yy = n.y * n.y, zz = n.z * n.z;
        const float xy = n.x * n.y, xz = n.x * n.z, yz = n.y * n.z;
        const float wx = n.w * n.x, wy = n.w * n.y, wz = n.w * n.z;
        Mat4 r;
        r.at(0, 0) = 1.0f - 2.0f * (yy + zz);
        r.at(0, 1) = 2.0f * (xy + wz);
        r.at(0, 2) = 2.0f * (xz - wy);
        r.at(1, 0) = 2.0f * (xy - wz);
        r.at(1, 1) = 1.0f - 2.0f * (xx + zz);
        r.at(1, 2) = 2.0f * (yz + wx);
        r.at(2, 0) = 2.0f * (xz + wy);
        r.at(2, 1) = 2.0f * (yz - wx);
        r.at(2, 2) = 1.0f - 2.0f * (xx + yy);
        return r;
    }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r = Mat4::zero();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float v = 0.0f;
            for (int k = 0; k < 4; ++k) v += a.at(i, k) * b.at(k, j);
            r.at(i, j) = v;
        }
    }
    return r;
}

inline Vec4 transform(const Vec4& v, const Mat4& m) {
    return {
        v.x * m.at(0, 0) + v.y * m.at(1, 0) + v.z * m.at(2, 0) + v.w * m.at(3, 0),
        v.x * m.at(0, 1) + v.y * m.at(1, 1) + v.z * m.at(2, 1) + v.w * m.at(3, 1),
        v.x * m.at(0, 2) + v.y * m.at(1, 2) + v.z * m.at(2, 2) + v.w * m.at(3, 2),
        v.x * m.at(0, 3) + v.y * m.at(1, 3) + v.z * m.at(2, 3) + v.w * m.at(3, 3),
    };
}

inline Vec3 transformPoint(const Mat4& m, const Vec3& p) {
    const Vec4 r = transform({p.x, p.y, p.z, 1.0f}, m);
    return {r.x, r.y, r.z};
}

inline Vec3 transformVector(const Mat4& m, const Vec3& v) {
    const Vec4 r = transform({v.x, v.y, v.z, 0.0f}, m);
    return {r.x, r.y, r.z};
}

inline Mat4 lookAtLH(const Vec3& eye, const Vec3& target, const Vec3& up) {
    const Vec3 zaxis = normalize(target - eye);
    Vec3 xaxis = cross(up, zaxis);
    if (length(xaxis) < 1.0e-6f) xaxis = cross({1.0f, 0.0f, 0.0f}, zaxis);
    xaxis = normalize(xaxis);
    const Vec3 yaxis = cross(zaxis, xaxis);
    Mat4 r;
    r.at(0, 0) = xaxis.x; r.at(0, 1) = yaxis.x; r.at(0, 2) = zaxis.x;
    r.at(1, 0) = xaxis.y; r.at(1, 1) = yaxis.y; r.at(1, 2) = zaxis.y;
    r.at(2, 0) = xaxis.z; r.at(2, 1) = yaxis.z; r.at(2, 2) = zaxis.z;
    r.at(3, 0) = -dot(xaxis, eye);
    r.at(3, 1) = -dot(yaxis, eye);
    r.at(3, 2) = -dot(zaxis, eye);
    return r;
}

inline Mat4 perspectiveLH(float fovY, float aspect, float zn, float zf) {
    Mat4 r = Mat4::zero();
    const float yScale = 1.0f / std::tan(fovY * 0.5f);
    const float xScale = yScale / aspect;
    r.at(0, 0) = xScale;
    r.at(1, 1) = yScale;
    r.at(2, 2) = zf / (zf - zn);
    r.at(2, 3) = 1.0f;
    r.at(3, 2) = -zn * zf / (zf - zn);
    return r;
}

inline Mat4 orthoLH(float l, float r, float b, float t, float zn, float zf) {
    Mat4 m = Mat4::zero();
    m.at(0, 0) = 2.0f / (r - l);
    m.at(1, 1) = 2.0f / (t - b);
    m.at(2, 2) = 1.0f / (zf - zn);
    m.at(3, 0) = (l + r) / (l - r);
    m.at(3, 1) = (t + b) / (b - t);
    m.at(3, 2) = -zn / (zf - zn);
    m.at(3, 3) = 1.0f;
    return m;
}

struct AABB {
    Vec3 mn{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max()};
    Vec3 mx{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
            -std::numeric_limits<float>::max()};

    [[nodiscard]] bool valid() const {
        return mn.x <= mx.x && mn.y <= mx.y && mn.z <= mx.z;
    }
    [[nodiscard]] Vec3 center() const { return (mn + mx) * 0.5f; }
    [[nodiscard]] Vec3 extent() const { return (mx - mn) * 0.5f; }
    [[nodiscard]] float radius() const { return length(extent()); }
    [[nodiscard]] Vec3 corner(int i) const {
        return {(i & 1) ? mx.x : mn.x, (i & 2) ? mx.y : mn.y, (i & 4) ? mx.z : mn.z};
    }
    void expand(const Vec3& p) {
        mn = minVec(mn, p);
        mx = maxVec(mx, p);
    }
    void expand(const AABB& b) {
        if (!b.valid()) return;
        expand(b.mn);
        expand(b.mx);
    }
};

inline AABB merge(const AABB& a, const AABB& b) {
    AABB r = a;
    r.expand(b);
    return r;
}

inline AABB transformedAABB(const AABB& local, const Mat4& world) {
    AABB r;
    for (int i = 0; i < 8; ++i) r.expand(transformPoint(world, local.corner(i)));
    return r;
}

struct Plane {
    Vec3 normal;
    float d = 0.0f;
};

struct Frustum {
    std::array<Plane, 6> planes{};

    [[nodiscard]] bool intersects(const AABB& box) const {
        if (!box.valid()) return false;
        for (const Plane& p : planes) {
            const Vec3 positive{
                p.normal.x >= 0.0f ? box.mx.x : box.mn.x,
                p.normal.y >= 0.0f ? box.mx.y : box.mn.y,
                p.normal.z >= 0.0f ? box.mx.z : box.mn.z,
            };
            if (dot(p.normal, positive) + p.d < 0.0f) return false;
        }
        return true;
    }
};

inline Plane normalizedPlane(Vec4 p) {
    const float n = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (n < 1.0e-8f) return {};
    return {{p.x / n, p.y / n, p.z / n}, p.w / n};
}

inline Frustum frustumFromViewProj(const Mat4& vp) {
    Frustum f;
    auto column = [&vp](int c, float sign) {
        return Vec4{
            vp.at(0, 3) + sign * vp.at(0, c),
            vp.at(1, 3) + sign * vp.at(1, c),
            vp.at(2, 3) + sign * vp.at(2, c),
            vp.at(3, 3) + sign * vp.at(3, c),
        };
    };
    f.planes[0] = normalizedPlane(column(0, 1.0f));
    f.planes[1] = normalizedPlane(column(0, -1.0f));
    f.planes[2] = normalizedPlane(column(1, 1.0f));
    f.planes[3] = normalizedPlane(column(1, -1.0f));
    f.planes[4] = normalizedPlane(column(2, 1.0f));
    f.planes[5] = normalizedPlane(column(2, -1.0f));
    return f;
}

struct Camera {
    Vec3 eye{0.0f, 8.0f, -25.0f};
    Vec3 target{0.0f, 2.0f, 0.0f};
    Vec3 up{0.0f, 1.0f, 0.0f};
    float fovY = 60.0f * kPi / 180.0f;
    float aspect = 16.0f / 9.0f;
    float nearZ = 0.1f;
    float farZ = 300.0f;

    [[nodiscard]] Mat4 view() const { return lookAtLH(eye, target, up); }
    [[nodiscard]] Mat4 projection() const {
        return perspectiveLH(fovY, aspect, nearZ, farZ);
    }
    [[nodiscard]] Mat4 viewProjection() const { return view() * projection(); }
    [[nodiscard]] Frustum frustum() const { return frustumFromViewProj(viewProjection()); }
};

inline std::array<Vec3, 8> frustumCorners(const Camera& camera, float splitNear, float splitFar) {
    const Vec3 forward = normalize(camera.target - camera.eye);
    Vec3 right = cross(camera.up, forward);
    right = normalize(right);
    const Vec3 up = cross(forward, right);
    const float tanHalf = std::tan(camera.fovY * 0.5f);
    const float halfNearH = splitNear * tanHalf;
    const float halfNearW = halfNearH * camera.aspect;
    const float halfFarH = splitFar * tanHalf;
    const float halfFarW = halfFarH * camera.aspect;
    const Vec3 nearCenter = camera.eye + forward * splitNear;
    const Vec3 farCenter = camera.eye + forward * splitFar;
    return {
        nearCenter - right * halfNearW - up * halfNearH,
        nearCenter + right * halfNearW - up * halfNearH,
        nearCenter + right * halfNearW + up * halfNearH,
        nearCenter - right * halfNearW + up * halfNearH,
        farCenter - right * halfFarW - up * halfFarH,
        farCenter + right * halfFarW - up * halfFarH,
        farCenter + right * halfFarW + up * halfFarH,
        farCenter - right * halfFarW + up * halfFarH,
    };
}

inline uint64_t hashBytes(uint64_t h, const void* data, size_t size) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

constexpr uint64_t kFnvOffset = 1469598103934665603ull;

} // namespace uid166

