#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace cv {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kEpsilon = 1.0e-7f;

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

inline Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
inline Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
inline Vec2 operator*(Vec2 a, float s) { return {a.x * s, a.y * s}; }
inline Vec2 operator*(float s, Vec2 a) { return {a.x * s, a.y * s}; }

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(float s, Vec3 a) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(Vec3 a, Vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
inline Vec3 operator/(Vec3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }
inline Vec3& operator+=(Vec3& a, Vec3 b) {
    a = a + b;
    return a;
}
inline Vec3& operator*=(Vec3& a, float s) {
    a = a * s;
    return a;
}

inline float dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float lengthSquared(Vec3 a) { return dot(a, a); }
inline float length(Vec3 a) { return std::sqrt(lengthSquared(a)); }
inline Vec3 normalize(Vec3 a) {
    float l = length(a);
    return l > kEpsilon ? a / l : Vec3{0.0f, 0.0f, 0.0f};
}
inline Vec3 lerp(Vec3 a, Vec3 b, float t) { return a + (b - a) * t; }
inline float lerp(float a, float b, float t) { return a + (b - a) * t; }
inline Vec3 reflect(Vec3 v, Vec3 n) { return v - n * (2.0f * dot(v, n)); }

struct Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

inline Vec4 operator+(Vec4 a, Vec4 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
inline Vec4 operator-(Vec4 a, Vec4 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
inline Vec4 operator*(Vec4 a, float s) { return {a.x * s, a.y * s, a.z * s, a.w * s}; }
inline Vec4 operator*(float s, Vec4 a) { return a * s; }
inline float dot(Vec4 a, Vec4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

struct Mat4 {
    // Row-major storage. Vectors are column vectors: clip = M * position.
    std::array<float, 16> m{};

    static Mat4 identity();
    static Mat4 translation(Vec3 t);
    static Mat4 scale(Vec3 s);
    static Mat4 rotationX(float radians);
    static Mat4 rotationY(float radians);
    static Mat4 rotationZ(float radians);
    static Mat4 lookAtLH(Vec3 eye, Vec3 target, Vec3 up);
    static Mat4 perspectiveD3D(float fovYRadians, float aspect, float nearZ, float farZ);
    static Mat4 orthographicD3D(float left, float right, float bottom, float top,
                                float nearZ, float farZ);

    float& at(int row, int col) { return m[static_cast<size_t>(row * 4 + col)]; }
    float at(int row, int col) const { return m[static_cast<size_t>(row * 4 + col)]; }
    Mat4 transpose() const;
    Mat4 inverse() const;
    Vec4 transform(Vec4 v) const;
    Vec3 transformPoint(Vec3 v) const;
    Vec3 transformVector(Vec3 v) const;
};

Mat4 operator*(const Mat4& a, const Mat4& b);

struct AABB {
    Vec3 min{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max()};
    Vec3 max{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
             -std::numeric_limits<float>::max()};

    bool valid() const { return max.x >= min.x && max.y >= min.y && max.z >= min.z; }
    Vec3 center() const { return (min + max) * 0.5f; }
    Vec3 extent() const { return (max - min) * 0.5f; }
    void expand(Vec3 p) {
        min.x = std::min(min.x, p.x);
        min.y = std::min(min.y, p.y);
        min.z = std::min(min.z, p.z);
        max.x = std::max(max.x, p.x);
        max.y = std::max(max.y, p.y);
        max.z = std::max(max.z, p.z);
    }
    void pad(float amount) {
        min = min - Vec3{amount, amount, amount};
        max = max + Vec3{amount, amount, amount};
    }
};

struct Plane {
    Vec3 n;
    float d = 0.0f;
};

struct Frustum {
    std::array<Plane, 6> planes{};
    bool intersects(const AABB& box) const;
};

Frustum extractFrustum(const Mat4& viewProjection);

struct Camera {
    Vec3 target{0.0f, 1.13f, 0.0f};
    float yawRadians = 0.55f;
    float pitchRadians = 0.13f;
    float distance = 5.1f;
    float fovYRadians = 50.0f * kPi / 180.0f;
    float nearZ = 0.08f;
    float farZ = 50.0f;

    Vec3 position() const;
    Vec3 forward() const;
    Mat4 view() const;
    Mat4 projection(float aspect) const;
    Mat4 viewProjection(float aspect) const;
};

float clamp01(float v);
Vec3 clamp01(Vec3 v);

}  // namespace cv

