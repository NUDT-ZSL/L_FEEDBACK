#pragma once

#include <array>
#include <cmath>

namespace dr {

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3 operator+(const Vec3& other) const;
    Vec3 operator-(const Vec3& other) const;
    Vec3 operator*(float scalar) const;
    Vec3& operator+=(const Vec3& other);
};

struct Mat4 {
    std::array<float, 16> m{};

    static Mat4 identity();
    static Mat4 perspective(float fovYDegrees, float aspect, float nearPlane, float farPlane);
    static Mat4 lookAt(const Vec3& eye, const Vec3& target, const Vec3& up);
    static Mat4 translationScale(const Vec3& translation, float scale);

    Mat4 operator*(const Mat4& other) const;
    const float* data() const { return m.data(); }
};

float radians(float degrees);
float dot(const Vec3& a, const Vec3& b);
Vec3 cross(const Vec3& a, const Vec3& b);
Vec3 normalize(const Vec3& value);

} // namespace dr
