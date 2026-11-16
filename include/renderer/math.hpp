#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace renderer {

constexpr float kPi = 3.14159265358979323846f;

inline float radians(float degrees) {
    return degrees * (kPi / 180.0f);
}

inline float clampf(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

inline Vec3 operator+(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(Vec3 value, float scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

inline Vec3 operator/(Vec3 value, float scalar) {
    return {value.x / scalar, value.y / scalar, value.z / scalar};
}

inline float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

inline float length(Vec3 value) {
    return std::sqrt(dot(value, value));
}

inline Vec3 normalize(Vec3 value) {
    const float len = length(value);
    if (len <= 1.0e-20f) {
        return {0.0f, 0.0f, 0.0f};
    }
    return value / len;
}

struct Mat3 {
    std::array<float, 9> m{};
};

struct Mat4 {
    // Column-major storage, matching OpenGL uniform uploads.
    std::array<float, 16> m{};

    float& operator()(int row, int column) {
        return m[static_cast<std::size_t>(column * 4 + row)];
    }

    float operator()(int row, int column) const {
        return m[static_cast<std::size_t>(column * 4 + row)];
    }
};

inline Mat4 identity() {
    Mat4 result{};
    result(0, 0) = 1.0f;
    result(1, 1) = 1.0f;
    result(2, 2) = 1.0f;
    result(3, 3) = 1.0f;
    return result;
}

inline Mat4 mat4FromRowMajor(const std::array<float, 16>& values) {
    Mat4 result{};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            result(row, column) = values[static_cast<std::size_t>(row * 4 + column)];
        }
    }
    return result;
}

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 result{};
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            float value = 0.0f;
            for (int k = 0; k < 4; ++k) {
                value += a(row, k) * b(k, column);
            }
            result(row, column) = value;
        }
    }
    return result;
}

inline Vec3 transformPoint(const Mat4& matrix, Vec3 point) {
    const float x = matrix(0, 0) * point.x + matrix(0, 1) * point.y
                  + matrix(0, 2) * point.z + matrix(0, 3);
    const float y = matrix(1, 0) * point.x + matrix(1, 1) * point.y
                  + matrix(1, 2) * point.z + matrix(1, 3);
    const float z = matrix(2, 0) * point.x + matrix(2, 1) * point.y
                  + matrix(2, 2) * point.z + matrix(2, 3);
    const float w = matrix(3, 0) * point.x + matrix(3, 1) * point.y
                  + matrix(3, 2) * point.z + matrix(3, 3);
    if (std::abs(w) > 1.0e-20f && std::abs(w - 1.0f) > 1.0e-20f) {
        return {x / w, y / w, z / w};
    }
    return {x, y, z};
}

inline Mat3 normalMatrix(const Mat4& matrix) {
    const float a00 = matrix(0, 0);
    const float a01 = matrix(0, 1);
    const float a02 = matrix(0, 2);
    const float a10 = matrix(1, 0);
    const float a11 = matrix(1, 1);
    const float a12 = matrix(1, 2);
    const float a20 = matrix(2, 0);
    const float a21 = matrix(2, 1);
    const float a22 = matrix(2, 2);

    const float c00 = a11 * a22 - a12 * a21;
    const float c01 = -(a10 * a22 - a12 * a20);
    const float c02 = a10 * a21 - a11 * a20;
    const float c10 = -(a01 * a22 - a02 * a21);
    const float c11 = a00 * a22 - a02 * a20;
    const float c12 = -(a00 * a21 - a01 * a20);
    const float c20 = a01 * a12 - a02 * a11;
    const float c21 = -(a00 * a12 - a02 * a10);
    const float c22 = a00 * a11 - a01 * a10;

    const float determinant = a00 * c00 + a01 * c01 + a02 * c02;
    Mat3 result{};
    if (std::abs(determinant) <= 1.0e-20f) {
        result.m = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        return result;
    }
    const float inv = 1.0f / determinant;
    result.m = {
        c00 * inv, c10 * inv, c20 * inv,
        c01 * inv, c11 * inv, c21 * inv,
        c02 * inv, c12 * inv, c22 * inv,
    };
    return result;
}

inline float determinant3(const Mat4& matrix) {
    const float a00 = matrix(0, 0);
    const float a01 = matrix(0, 1);
    const float a02 = matrix(0, 2);
    const float a10 = matrix(1, 0);
    const float a11 = matrix(1, 1);
    const float a12 = matrix(1, 2);
    const float a20 = matrix(2, 0);
    const float a21 = matrix(2, 1);
    const float a22 = matrix(2, 2);
    return a00 * (a11 * a22 - a12 * a21)
         - a01 * (a10 * a22 - a12 * a20)
         + a02 * (a10 * a21 - a11 * a20);
}

inline Mat4 perspective(float fovYRadians, float aspect, float nearPlane, float farPlane) {
    const float f = 1.0f / std::tan(fovYRadians * 0.5f);
    Mat4 result{};
    result(0, 0) = f / aspect;
    result(1, 1) = f;
    result(2, 2) = (farPlane + nearPlane) / (nearPlane - farPlane);
    result(2, 3) = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
    result(3, 2) = -1.0f;
    return result;
}

inline Mat4 orthographic(float left, float right, float bottom, float top,
                         float nearPlane, float farPlane) {
    Mat4 result = identity();
    result(0, 0) = 2.0f / (right - left);
    result(1, 1) = 2.0f / (top - bottom);
    result(2, 2) = -2.0f / (farPlane - nearPlane);
    result(0, 3) = -(right + left) / (right - left);
    result(1, 3) = -(top + bottom) / (top - bottom);
    result(2, 3) = -(farPlane + nearPlane) / (farPlane - nearPlane);
    return result;
}

inline Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
    const Vec3 forward = normalize(center - eye);
    const Vec3 side = normalize(cross(forward, up));
    const Vec3 correctedUp = cross(side, forward);

    Mat4 result = identity();
    result(0, 0) = side.x;
    result(0, 1) = side.y;
    result(0, 2) = side.z;
    result(1, 0) = correctedUp.x;
    result(1, 1) = correctedUp.y;
    result(1, 2) = correctedUp.z;
    result(2, 0) = -forward.x;
    result(2, 1) = -forward.y;
    result(2, 2) = -forward.z;
    result(0, 3) = -dot(side, eye);
    result(1, 3) = -dot(correctedUp, eye);
    result(2, 3) = dot(forward, eye);
    return result;
}

struct Aabb {
    Vec3 minimum{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
    };
    Vec3 maximum{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
    };

    bool valid() const {
        return minimum.x <= maximum.x && minimum.y <= maximum.y && minimum.z <= maximum.z;
    }

    void expand(Vec3 point) {
        minimum.x = std::min(minimum.x, point.x);
        minimum.y = std::min(minimum.y, point.y);
        minimum.z = std::min(minimum.z, point.z);
        maximum.x = std::max(maximum.x, point.x);
        maximum.y = std::max(maximum.y, point.y);
        maximum.z = std::max(maximum.z, point.z);
    }

    Vec3 center() const {
        return (minimum + maximum) * 0.5f;
    }
};

}  // namespace renderer
