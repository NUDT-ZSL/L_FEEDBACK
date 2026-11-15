#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace subdiv {

constexpr float kPi = 3.14159265358979323846f;

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    constexpr Vec3() = default;
    constexpr Vec3(float xValue, float yValue, float zValue)
        : x(xValue), y(yValue), z(zValue) {}

    constexpr Vec3 operator+(const Vec3& rhs) const {
        return {x + rhs.x, y + rhs.y, z + rhs.z};
    }
    constexpr Vec3 operator-(const Vec3& rhs) const {
        return {x - rhs.x, y - rhs.y, z - rhs.z};
    }
    constexpr Vec3 operator*(float scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }
    constexpr Vec3 operator/(float scalar) const {
        return {x / scalar, y / scalar, z / scalar};
    }
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
};

inline constexpr Vec3 operator*(float scalar, const Vec3& value) {
    return value * scalar;
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

inline float lengthSquared(const Vec3& value) {
    return dot(value, value);
}

inline float length(const Vec3& value) {
    return std::sqrt(lengthSquared(value));
}

inline Vec3 normalize(const Vec3& value) {
    const float len = length(value);
    if (len <= 1.0e-20f) {
        return {0.0f, 0.0f, 0.0f};
    }
    return value / len;
}

inline Vec3 lerp(const Vec3& a, const Vec3& b, float t) {
    return a * (1.0f - t) + b * t;
}

inline float clamp(float value, float low, float high) {
    return std::max(low, std::min(high, value));
}

struct Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

struct Mat4 {
    float values[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };

    static Mat4 identity() {
        return {};
    }

    static Mat4 perspective(float fovRadians, float aspect, float nearPlane, float farPlane) {
        Mat4 result{};
        const float f = 1.0f / std::tan(fovRadians * 0.5f);
        result.values[0] = f / aspect;
        result.values[5] = f;
        result.values[10] = (farPlane + nearPlane) / (nearPlane - farPlane);
        result.values[11] = -1.0f;
        result.values[14] = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
        result.values[15] = 0.0f;
        return result;
    }

    static Mat4 lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
        const Vec3 forward = normalize(target - eye);
        Vec3 side = normalize(cross(forward, up));
        if (lengthSquared(side) < 1.0e-12f) {
            side = normalize(cross(forward, Vec3{0.0f, 0.0f, 1.0f}));
        }
        const Vec3 correctedUp = cross(side, forward);

        Mat4 result{};
        result.values[0] = side.x;
        result.values[1] = correctedUp.x;
        result.values[2] = -forward.x;
        result.values[3] = 0.0f;

        result.values[4] = side.y;
        result.values[5] = correctedUp.y;
        result.values[6] = -forward.y;
        result.values[7] = 0.0f;

        result.values[8] = side.z;
        result.values[9] = correctedUp.z;
        result.values[10] = -forward.z;
        result.values[11] = 0.0f;

        result.values[12] = -dot(side, eye);
        result.values[13] = -dot(correctedUp, eye);
        result.values[14] = dot(forward, eye);
        result.values[15] = 1.0f;
        return result;
    }

    static Mat4 multiply(const Mat4& a, const Mat4& b) {
        Mat4 result{};
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row) {
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    sum += a.values[k * 4 + row] * b.values[column * 4 + k];
                }
                result.values[column * 4 + row] = sum;
            }
        }
        return result;
    }
};

struct CameraSettings {
    Vec3 position{4.8f, 3.6f, 5.4f};
    Vec3 target{0.0f, 0.55f, 0.0f};
    Vec3 up{0.0f, 1.0f, 0.0f};
    float fovDegrees = 42.0f;
};

}  // namespace subdiv

