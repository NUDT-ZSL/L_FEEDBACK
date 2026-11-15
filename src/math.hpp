#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace scene {

constexpr float kPi = 3.14159265358979323846f;

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(float s, Vec3 a) { return a * s; }
inline Vec3 operator*(Vec3 a, Vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
inline Vec3& operator+=(Vec3& a, Vec3 b) { a = a + b; return a; }

inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
inline float length(Vec3 v) { return std::sqrt(dot(v, v)); }
inline Vec3 normalize(Vec3 v) {
    const float len = length(v);
    return len > 1.0e-8f ? v * (1.0f / len) : Vec3{0.0f, 0.0f, 0.0f};
}
inline Vec3 lerp(Vec3 a, Vec3 b, float t) { return a + (b - a) * t; }

struct Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

struct Mat3 {
    std::array<float, 9> m{};

    static Mat3 identity() {
        Mat3 r;
        r.m[0] = r.m[4] = r.m[8] = 1.0f;
        return r;
    }
};

struct Mat4 {
    std::array<float, 16> m{};

    static Mat4 identity() {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a.m[k * 4 + row] * b.m[col * 4 + k];
            }
            r.m[col * 4 + row] = sum;
        }
    }
    return r;
}

inline Vec4 operator*(const Mat4& a, Vec4 v) {
    return {
        a.m[0] * v.x + a.m[4] * v.y + a.m[8] * v.z + a.m[12] * v.w,
        a.m[1] * v.x + a.m[5] * v.y + a.m[9] * v.z + a.m[13] * v.w,
        a.m[2] * v.x + a.m[6] * v.y + a.m[10] * v.z + a.m[14] * v.w,
        a.m[3] * v.x + a.m[7] * v.y + a.m[11] * v.z + a.m[15] * v.w
    };
}

inline Mat3 upperLeft3(const Mat4& a) {
    Mat3 r;
    r.m[0] = a.m[0]; r.m[1] = a.m[1]; r.m[2] = a.m[2];
    r.m[3] = a.m[4]; r.m[4] = a.m[5]; r.m[5] = a.m[6];
    r.m[6] = a.m[8]; r.m[7] = a.m[9]; r.m[8] = a.m[10];
    return r;
}

inline Mat4 translate(Vec3 t) {
    Mat4 r = Mat4::identity();
    r.m[12] = t.x;
    r.m[13] = t.y;
    r.m[14] = t.z;
    return r;
}

inline Mat4 scale(Vec3 s) {
    Mat4 r = Mat4::identity();
    r.m[0] = s.x;
    r.m[5] = s.y;
    r.m[10] = s.z;
    return r;
}

inline Mat4 rotateX(float radians) {
    Mat4 r = Mat4::identity();
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    r.m[5] = c;
    r.m[6] = s;
    r.m[9] = -s;
    r.m[10] = c;
    return r;
}

inline Mat4 rotateY(float radians) {
    Mat4 r = Mat4::identity();
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    r.m[0] = c;
    r.m[2] = -s;
    r.m[8] = s;
    r.m[10] = c;
    return r;
}

inline Mat4 rotateZ(float radians) {
    Mat4 r = Mat4::identity();
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    r.m[0] = c;
    r.m[1] = s;
    r.m[4] = -s;
    r.m[5] = c;
    return r;
}

inline Mat4 perspective(float fovYRadians, float aspect, float nearPlane, float farPlane) {
    const float f = 1.0f / std::tan(fovYRadians * 0.5f);
    Mat4 r;
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (farPlane + nearPlane) / (nearPlane - farPlane);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
    return r;
}

inline Mat4 orthographic(float left, float right, float bottom, float top,
                         float nearPlane, float farPlane) {
    Mat4 r = Mat4::identity();
    r.m[0] = 2.0f / (right - left);
    r.m[5] = 2.0f / (top - bottom);
    r.m[10] = -2.0f / (farPlane - nearPlane);
    r.m[12] = -(right + left) / (right - left);
    r.m[13] = -(top + bottom) / (top - bottom);
    r.m[14] = -(farPlane + nearPlane) / (farPlane - nearPlane);
    return r;
}

inline Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
    const Vec3 f = normalize(center - eye);
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);
    Mat4 r = Mat4::identity();
    r.m[0] = s.x;  r.m[4] = s.y;  r.m[8] = s.z;
    r.m[1] = u.x;  r.m[5] = u.y;  r.m[9] = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -dot(s, eye);
    r.m[13] = -dot(u, eye);
    r.m[14] = dot(f, eye);
    return r;
}

inline Mat4 inverse(const Mat4& mat) {
    const auto& m = mat.m;
    Mat4 out;
    auto& o = out.m;

    const float b00 = m[0] * m[5] - m[1] * m[4];
    const float b01 = m[0] * m[6] - m[2] * m[4];
    const float b02 = m[0] * m[7] - m[3] * m[4];
    const float b03 = m[1] * m[6] - m[2] * m[5];
    const float b04 = m[1] * m[7] - m[3] * m[5];
    const float b05 = m[2] * m[7] - m[3] * m[6];
    const float b06 = m[8] * m[13] - m[9] * m[12];
    const float b07 = m[8] * m[14] - m[10] * m[12];
    const float b08 = m[8] * m[15] - m[11] * m[12];
    const float b09 = m[9] * m[14] - m[10] * m[13];
    const float b10 = m[9] * m[15] - m[11] * m[13];
    const float b11 = m[10] * m[15] - m[11] * m[14];

    const float det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08
                    - b04 * b07 + b05 * b06;
    if (std::abs(det) < 1.0e-12f) {
        return Mat4::identity();
    }
    const float invDet = 1.0f / det;

    o[0] = (m[5] * b11 - m[6] * b10 + m[7] * b09) * invDet;
    o[1] = (m[2] * b10 - m[1] * b11 - m[3] * b09) * invDet;
    o[2] = (m[13] * b05 - m[14] * b04 + m[15] * b03) * invDet;
    o[3] = (m[10] * b04 - m[9] * b05 - m[11] * b03) * invDet;
    o[4] = (m[6] * b08 - m[4] * b11 - m[7] * b07) * invDet;
    o[5] = (m[0] * b11 - m[2] * b08 + m[3] * b07) * invDet;
    o[6] = (m[14] * b02 - m[12] * b05 - m[15] * b01) * invDet;
    o[7] = (m[8] * b05 - m[10] * b02 + m[11] * b01) * invDet;
    o[8] = (m[4] * b10 - m[5] * b08 + m[7] * b06) * invDet;
    o[9] = (m[1] * b08 - m[0] * b10 - m[3] * b06) * invDet;
    o[10] = (m[12] * b04 - m[13] * b02 + m[15] * b00) * invDet;
    o[11] = (m[9] * b02 - m[8] * b04 - m[11] * b00) * invDet;
    o[12] = (m[5] * b07 - m[4] * b09 - m[6] * b06) * invDet;
    o[13] = (m[0] * b09 - m[1] * b07 + m[2] * b06) * invDet;
    o[14] = (m[13] * b01 - m[12] * b03 - m[14] * b00) * invDet;
    o[15] = (m[8] * b03 - m[9] * b01 + m[10] * b00) * invDet;
    return out;
}

inline Vec3 transformPoint(const Mat4& mat, Vec3 p) {
    const Vec4 r = mat * Vec4{p.x, p.y, p.z, 1.0f};
    return {r.x, r.y, r.z};
}

inline Vec3 transformDirection(const Mat4& mat, Vec3 d) {
    const Vec4 r = mat * Vec4{d.x, d.y, d.z, 0.0f};
    return {r.x, r.y, r.z};
}

} // namespace scene

