#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cloud {

constexpr float kPi = 3.14159265358979323846f;

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3() = default;
    Vec3(float xValue, float yValue, float zValue) : x(xValue), y(yValue), z(zValue) {}

    Vec3 operator+(const Vec3& other) const { return {x + other.x, y + other.y, z + other.z}; }
    Vec3 operator-(const Vec3& other) const { return {x - other.x, y - other.y, z - other.z}; }
    Vec3 operator*(float scalar) const { return {x * scalar, y * scalar, z * scalar}; }
    Vec3 operator*(const Vec3& other) const { return {x * other.x, y * other.y, z * other.z}; }
    Vec3 operator/(float scalar) const { return {x / scalar, y / scalar, z / scalar}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    bool operator==(const Vec3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    Vec3& operator*=(const Vec3& other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return *this;
    }
};

inline Vec3 operator*(float scalar, const Vec3& value) { return value * scalar; }
inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(const Vec3& value) { return std::sqrt(dot(value, value)); }
inline Vec3 normalize(const Vec3& value) {
    const float len = length(value);
    return len > 1.0e-20f ? value / len : Vec3{0.0f, 0.0f, 0.0f};
}
inline Vec3 clamp(const Vec3& value, float low, float high) {
    return {std::clamp(value.x, low, high), std::clamp(value.y, low, high),
            std::clamp(value.z, low, high)};
}
inline Vec3 lerp(const Vec3& a, const Vec3& b, float t) { return a + (b - a) * t; }
inline Vec3 mix(const Vec3& a, const Vec3& b, float t) { return lerp(a, b, t); }

inline float clampf(float value, float low, float high) { return std::clamp(value, low, high); }
inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
inline float smoothstep(float edge0, float edge1, float x) {
    const float t = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

struct Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

struct Mat4 {
    std::array<float, 16> m{};

    static Mat4 identity();
    static Mat4 lookAt(const Vec3& eye, const Vec3& target, const Vec3& up);
    static Mat4 perspective(float fovYRadians, float aspect, float nearPlane, float farPlane);

    const float* data() const { return m.data(); }
    float* data() { return m.data(); }
    float operator()(int row, int column) const { return m[static_cast<std::size_t>(row * 4 + column)]; }
    float& operator()(int row, int column) { return m[static_cast<std::size_t>(row * 4 + column)]; }
    Mat4 operator*(const Mat4& other) const;
    Vec4 operator*(const Vec4& value) const;
    Mat4 inverse() const;
    Vec3 transformPoint(const Vec3& point) const;
    Vec3 transformDirection(const Vec3& direction) const;
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
};

struct Camera {
    Vec3 position;
    Mat4 view = Mat4::identity();
    Mat4 projection = Mat4::identity();
    Mat4 inverseViewProjection = Mat4::identity();

    static Camera fromLookAt(const Vec3& position, const Vec3& target, const Vec3& up, float fovDegrees,
                             float aspect, float nearPlane, float farPlane);
    static Camera fromMatrices(const Mat4& view, const Mat4& projection);
    Ray rayForPixel(float pixelX, float pixelY, int width, int height) const;
};

inline bool finite(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

}  // namespace cloud

