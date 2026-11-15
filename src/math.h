#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace pbr {

constexpr double kPi = 3.14159265358979323846;

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3& operator+=(const Vec3& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    Vec3& operator-=(const Vec3& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    Vec3& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    Vec3& operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator-(const Vec3& v) {
    return {-v.x, -v.y, -v.z};
}

inline Vec3 operator*(const Vec3& a, const Vec3& b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

inline Vec3 operator*(const Vec3& v, float scalar) {
    return {v.x * scalar, v.y * scalar, v.z * scalar};
}

inline Vec3 operator*(float scalar, const Vec3& v) {
    return v * scalar;
}

inline Vec3 operator/(const Vec3& v, float scalar) {
    return {v.x / scalar, v.y / scalar, v.z / scalar};
}

inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

inline float lengthSquared(const Vec3& v) {
    return dot(v, v);
}

inline float length(const Vec3& v) {
    return std::sqrt(lengthSquared(v));
}

inline Vec3 normalize(const Vec3& v) {
    const float len = length(v);
    if (len <= 1.0e-20f) {
        return {0.0f, 0.0f, 0.0f};
    }
    return v / len;
}

inline Vec3 reflect(const Vec3& incident, const Vec3& normal) {
    return incident - 2.0f * dot(incident, normal) * normal;
}

inline float clampf(float value, float lo, float hi) {
    return std::max(lo, std::min(value, hi));
}

inline Vec3 clampColor(const Vec3& value, float lo, float hi) {
    return {
        clampf(value.x, lo, hi),
        clampf(value.y, lo, hi),
        clampf(value.z, lo, hi),
    };
}

inline float luminance(const Vec3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

static_assert(sizeof(Vec3) == sizeof(float) * 3, "Vec3 must remain tightly packed for GL uploads");

}  // namespace pbr
