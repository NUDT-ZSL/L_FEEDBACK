#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace vlr {

constexpr double kPi = 3.14159265358979323846;

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    float& operator[](std::size_t index) { return (&x)[index]; }
    const float& operator[](std::size_t index) const { return (&x)[index]; }
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(float s, Vec3 a) { return a * s; }
inline Vec3 operator/(Vec3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }
inline Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }

inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}
inline float lengthSquared(Vec3 v) { return dot(v, v); }
inline float length(Vec3 v) { return std::sqrt(lengthSquared(v)); }
inline Vec3 normalize(Vec3 v) {
    const float len = length(v);
    return len > 0.0f ? v / len : Vec3{};
}

struct alignas(16) Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

struct alignas(16) Mat4 {
    std::array<Vec4, 4> columns{};

    Vec4& operator[](std::size_t column) { return columns[column]; }
    const Vec4& operator[](std::size_t column) const { return columns[column]; }
};

inline Mat4 identity() {
    Mat4 result{};
    result[0].x = 1.0f;
    result[1].y = 1.0f;
    result[2].z = 1.0f;
    result[3].w = 1.0f;
    return result;
}

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 result{};
    for (std::size_t column = 0; column < 4; ++column) {
        for (std::size_t row = 0; row < 4; ++row) {
            float value = 0.0f;
            for (std::size_t k = 0; k < 4; ++k) {
                const float av = (&a.columns[k].x)[row];
                const float bv = (&b.columns[column].x)[k];
                value += av * bv;
            }
            (&result.columns[column].x)[row] = value;
        }
    }
    return result;
}

inline Vec4 operator*(const Mat4& matrix, const Vec4& vector) {
    Vec4 result{};
    result.x = matrix[0].x * vector.x + matrix[1].x * vector.y +
               matrix[2].x * vector.z + matrix[3].x * vector.w;
    result.y = matrix[0].y * vector.x + matrix[1].y * vector.y +
               matrix[2].y * vector.z + matrix[3].y * vector.w;
    result.z = matrix[0].z * vector.x + matrix[1].z * vector.y +
               matrix[2].z * vector.z + matrix[3].z * vector.w;
    result.w = matrix[0].w * vector.x + matrix[1].w * vector.y +
               matrix[2].w * vector.z + matrix[3].w * vector.w;
    return result;
}

inline Mat4 perspectiveVulkan(float fovYDegrees, float aspect, float nearPlane,
                              float farPlane, bool flipY = false) {
    const float f = 1.0f / std::tan(fovYDegrees * static_cast<float>(kPi) / 360.0f);
    Mat4 result{};
    result[0].x = f / aspect;
    result[1].y = flipY ? -f : f;
    result[2].z = farPlane / (nearPlane - farPlane);
    result[2].w = -1.0f;
    result[3].z = nearPlane * farPlane / (nearPlane - farPlane);
    return result;
}

inline Mat4 orthoVulkan(float left, float right, float bottom, float top,
                        float nearPlane, float farPlane) {
    Mat4 result = identity();
    result[0].x = 2.0f / (right - left);
    result[1].y = 2.0f / (top - bottom);
    result[2].z = 1.0f / (nearPlane - farPlane);
    result[3].x = -(right + left) / (right - left);
    result[3].y = -(top + bottom) / (top - bottom);
    result[3].z = nearPlane / (nearPlane - farPlane);
    return result;
}

inline Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 up) {
    const Vec3 f = normalize(target - eye);
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);
    Mat4 result = identity();
    result[0].x = s.x;
    result[1].x = s.y;
    result[2].x = s.z;
    result[3].x = -dot(s, eye);
    result[0].y = u.x;
    result[1].y = u.y;
    result[2].y = u.z;
    result[3].y = -dot(u, eye);
    result[0].z = -f.x;
    result[1].z = -f.y;
    result[2].z = -f.z;
    result[3].z = dot(f, eye);
    return result;
}

inline Mat4 viewFromBasis(Vec3 right, Vec3 up, Vec3 forward, Vec3 eye) {
    Mat4 result = identity();
    result[0].x = right.x;
    result[1].x = right.y;
    result[2].x = right.z;
    result[3].x = -dot(right, eye);
    result[0].y = up.x;
    result[1].y = up.y;
    result[2].y = up.z;
    result[3].y = -dot(up, eye);
    result[0].z = -forward.x;
    result[1].z = -forward.y;
    result[2].z = -forward.z;
    result[3].z = dot(forward, eye);
    return result;
}

inline Mat4 inverse(const Mat4& matrix) {
    const float* m = &matrix[0].x;
    float inv[16];

    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] -
             m[9] * m[6] * m[15] + m[9] * m[7] * m[14] +
             m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] +
             m[8] * m[6] * m[15] - m[8] * m[7] * m[14] -
             m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] -
             m[8] * m[5] * m[15] + m[8] * m[7] * m[13] +
             m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] +
              m[8] * m[5] * m[14] - m[8] * m[6] * m[13] -
              m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] +
             m[9] * m[2] * m[15] - m[9] * m[3] * m[14] -
             m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] -
             m[8] * m[2] * m[15] + m[8] * m[3] * m[14] +
             m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] +
             m[8] * m[1] * m[15] - m[8] * m[3] * m[13] -
             m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] -
              m[8] * m[1] * m[14] + m[8] * m[2] * m[13] +
              m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] -
             m[5] * m[2] * m[15] + m[5] * m[3] * m[14] +
             m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] +
             m[4] * m[2] * m[15] - m[4] * m[3] * m[14] -
             m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] -
              m[4] * m[1] * m[15] + m[4] * m[3] * m[13] +
              m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] +
              m[4] * m[1] * m[14] - m[4] * m[2] * m[13] -
              m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] +
             m[5] * m[2] * m[11] - m[5] * m[3] * m[10] -
             m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] + m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] +
              m[4] * m[1] * m[11] - m[4] * m[3] * m[9] -
              m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] + m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

    const float determinant =
        m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    Mat4 result{};
    if (std::abs(determinant) < 1.0e-20f) {
        return identity();
    }
    const float invDet = 1.0f / determinant;
    float* output = &result[0].x;
    for (int i = 0; i < 16; ++i) {
        output[i] = inv[i] * invDet;
    }
    return result;
}

} // namespace vlr
