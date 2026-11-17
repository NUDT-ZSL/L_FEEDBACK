#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace softraster {

struct Vec2 {
    float x = 0.0F;
    float y = 0.0F;
};

struct Vec3 {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
};

struct Vec4 {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    float w = 0.0F;
};

[[nodiscard]] Vec2 operator+(Vec2 a, Vec2 b);
[[nodiscard]] Vec2 operator-(Vec2 a, Vec2 b);
[[nodiscard]] Vec2 operator*(Vec2 a, float scalar);
[[nodiscard]] Vec2 operator/(Vec2 a, float scalar);

[[nodiscard]] Vec3 operator+(Vec3 a, Vec3 b);
[[nodiscard]] Vec3 operator-(Vec3 a, Vec3 b);
[[nodiscard]] Vec3 operator-(Vec3 a);
[[nodiscard]] Vec3 operator*(Vec3 a, float scalar);
[[nodiscard]] Vec3 operator*(float scalar, Vec3 a);
[[nodiscard]] Vec3 operator/(Vec3 a, float scalar);
[[nodiscard]] Vec3 operator*(Vec3 a, Vec3 b);

[[nodiscard]] Vec4 operator+(Vec4 a, Vec4 b);
[[nodiscard]] Vec4 operator-(Vec4 a, Vec4 b);
[[nodiscard]] Vec4 operator*(Vec4 a, float scalar);

[[nodiscard]] float dot(Vec2 a, Vec2 b);
[[nodiscard]] float dot(Vec3 a, Vec3 b);
[[nodiscard]] Vec3 cross(Vec3 a, Vec3 b);
[[nodiscard]] float lengthSquared(Vec3 value);
[[nodiscard]] float length(Vec3 value);
[[nodiscard]] Vec3 normalize(Vec3 value, Vec3 fallback = Vec3{0.0F, 0.0F, 1.0F});
[[nodiscard]] Vec2 normalize(Vec2 value, Vec2 fallback = Vec2{0.0F, 0.0F});
[[nodiscard]] Vec3 minComponents(Vec3 a, Vec3 b);
[[nodiscard]] Vec3 maxComponents(Vec3 a, Vec3 b);
[[nodiscard]] Vec3 clamp01(Vec3 value);
[[nodiscard]] float clamp01(float value);

struct Mat4 {
    // Row-major storage. A point is transformed as matrix * column-vector.
    std::array<float, 16> m{};

    [[nodiscard]] static Mat4 identity();
};

[[nodiscard]] Mat4 operator*(const Mat4& a, const Mat4& b);
[[nodiscard]] Vec4 operator*(const Mat4& matrix, Vec4 value);
[[nodiscard]] Vec4 transformPoint(const Mat4& matrix, Vec3 point);
[[nodiscard]] Vec3 transformVector(const Mat4& matrix, Vec3 vector);
[[nodiscard]] Mat4 transpose(const Mat4& matrix);
[[nodiscard]] Mat4 inverse(const Mat4& matrix, bool& invertible);

[[nodiscard]] Mat4 translation(Vec3 value);
[[nodiscard]] Mat4 scale(Vec3 value);
[[nodiscard]] Mat4 rotationX(float radians);
[[nodiscard]] Mat4 rotationY(float radians);
[[nodiscard]] Mat4 rotationZ(float radians);

[[nodiscard]] Mat4 lookAtRh(Vec3 eye, Vec3 target, Vec3 up, bool& valid);
[[nodiscard]] Mat4 perspectiveRh(float fovYDegrees, float aspect, float nearPlane, float farPlane);
[[nodiscard]] Mat4 orthographicRh(float left, float right, float bottom, float top,
                                  float nearPlane, float farPlane);

[[nodiscard]] bool isFinite(Vec2 value);
[[nodiscard]] bool isFinite(Vec3 value);
[[nodiscard]] bool isFinite(const Mat4& value);
[[nodiscard]] double radians(double degrees);

}  // namespace softraster
