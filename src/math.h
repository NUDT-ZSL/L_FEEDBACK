#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace df {

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
    Vec3(float vx, float vy, float vz) : x(vx), y(vy), z(vz) {}
    explicit Vec3(float v) : x(v), y(v), z(v) {}

    Vec3 operator+(const Vec3& rhs) const { return {x + rhs.x, y + rhs.y, z + rhs.z}; }
    Vec3 operator-(const Vec3& rhs) const { return {x - rhs.x, y - rhs.y, z - rhs.z}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator*(const Vec3& rhs) const { return {x * rhs.x, y * rhs.y, z * rhs.z}; }
    Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    Vec3 operator/(const Vec3& rhs) const { return {x / rhs.x, y / rhs.y, z / rhs.z}; }
    Vec3& operator+=(const Vec3& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }
    Vec3& operator*=(float s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }
};

inline Vec3 operator*(float s, const Vec3& v) { return v * s; }

struct Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline float length(const Vec3& v) { return std::sqrt(dot(v, v)); }

inline Vec3 normalize(const Vec3& v) {
    const float len = length(v);
    return len > 1.0e-8f ? v / len : Vec3{};
}

inline Vec3 clamp(const Vec3& v, float lo, float hi) {
    return {std::clamp(v.x, lo, hi), std::clamp(v.y, lo, hi), std::clamp(v.z, lo, hi)};
}

inline Vec3 mix(const Vec3& a, const Vec3& b, float t) {
    return a * (1.0f - t) + b * t;
}

inline float lerp(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

inline Vec3 reflect(const Vec3& i, const Vec3& n) {
    return i - n * (2.0f * dot(i, n));
}

inline float smoothstep(float edge0, float edge1, float x) {
    const float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

struct Mat4 {
    float m[16]{};

    static Mat4 identity() {
        Mat4 out;
        out.m[0] = out.m[5] = out.m[10] = out.m[15] = 1.0f;
        return out;
    }

    static Mat4 perspective(float fovYRadians, float aspect, float nearPlane, float farPlane) {
        Mat4 out;
        const float f = 1.0f / std::tan(fovYRadians * 0.5f);
        out.m[0] = f / aspect;
        out.m[5] = f;
        out.m[10] = (farPlane + nearPlane) / (nearPlane - farPlane);
        out.m[11] = -1.0f;
        out.m[14] = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
        return out;
    }

    static Mat4 lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
        const Vec3 f = normalize(target - eye);
        const Vec3 s = normalize(cross(f, up));
        const Vec3 u = cross(s, f);
        Mat4 out = identity();
        out.m[0] = s.x;  out.m[4] = s.y;  out.m[8] = s.z;
        out.m[1] = u.x;  out.m[5] = u.y;  out.m[9] = u.z;
        out.m[2] = -f.x; out.m[6] = -f.y; out.m[10] = -f.z;
        out.m[12] = -dot(s, eye);
        out.m[13] = -dot(u, eye);
        out.m[14] = dot(f, eye);
        return out;
    }

    static Mat4 translation(const Vec3& translation) {
        Mat4 out = identity();
        out.m[12] = translation.x;
        out.m[13] = translation.y;
        out.m[14] = translation.z;
        return out;
    }

    static Mat4 scaling(const Vec3& scale) {
        Mat4 out = identity();
        out.m[0] = scale.x;
        out.m[5] = scale.y;
        out.m[10] = scale.z;
        return out;
    }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 out;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            out.m[col * 4 + row] =
                a.m[row] * b.m[col * 4] +
                a.m[4 + row] * b.m[col * 4 + 1] +
                a.m[8 + row] * b.m[col * 4 + 2] +
                a.m[12 + row] * b.m[col * 4 + 3];
        }
    }
    return out;
}

inline Vec4 operator*(const Mat4& m, const Vec4& v) {
    return {
        m.m[0] * v.x + m.m[4] * v.y + m.m[8] * v.z + m.m[12] * v.w,
        m.m[1] * v.x + m.m[5] * v.y + m.m[9] * v.z + m.m[13] * v.w,
        m.m[2] * v.x + m.m[6] * v.y + m.m[10] * v.z + m.m[14] * v.w,
        m.m[3] * v.x + m.m[7] * v.y + m.m[11] * v.z + m.m[15] * v.w,
    };
}

inline Vec3 transformPoint(const Mat4& m, const Vec3& p) {
    const Vec4 out = m * Vec4{p.x, p.y, p.z, 1.0f};
    return {out.x, out.y, out.z};
}

inline Mat4 inverse(const Mat4& input) {
    const float* a = input.m;
    float inv[16];
    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] -
             a[9] * a[6] * a[15] + a[9] * a[7] * a[14] +
             a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] +
             a[8] * a[6] * a[15] - a[8] * a[7] * a[14] -
             a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] -
             a[8] * a[5] * a[15] + a[8] * a[7] * a[13] +
             a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] +
              a[8] * a[5] * a[14] - a[8] * a[6] * a[13] -
              a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] +
             a[9] * a[2] * a[15] - a[9] * a[3] * a[14] -
             a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] -
             a[8] * a[2] * a[15] + a[8] * a[3] * a[14] +
             a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] +
             a[8] * a[1] * a[15] - a[8] * a[3] * a[13] -
             a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] -
              a[8] * a[1] * a[14] + a[8] * a[2] * a[13] +
              a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] -
             a[5] * a[2] * a[15] + a[5] * a[3] * a[14] +
             a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] +
             a[4] * a[2] * a[15] - a[4] * a[3] * a[14] -
             a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] -
              a[4] * a[1] * a[15] + a[4] * a[3] * a[13] +
              a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] +
              a[4] * a[1] * a[14] - a[4] * a[2] * a[13] -
              a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] +
             a[5] * a[2] * a[11] - a[5] * a[3] * a[10] -
             a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] -
             a[4] * a[2] * a[11] + a[4] * a[3] * a[10] +
             a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] +
              a[4] * a[1] * a[11] - a[4] * a[3] * a[9] -
              a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] -
              a[4] * a[1] * a[10] + a[4] * a[2] * a[9] +
              a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
    const float det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    Mat4 out;
    if (std::fabs(det) < 1.0e-12f) return out;
    const float reciprocal = 1.0f / det;
    for (int i = 0; i < 16; ++i) out.m[i] = inv[i] * reciprocal;
    return out;
}

inline bool approxEqual(float a, float b, float eps = 1.0e-6f) {
    return std::fabs(a - b) <= eps * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
}

inline bool approxEqual(const Vec3& a, const Vec3& b, float eps = 1.0e-6f) {
    return approxEqual(a.x, b.x, eps) && approxEqual(a.y, b.y, eps) &&
           approxEqual(a.z, b.z, eps);
}

} // namespace df

