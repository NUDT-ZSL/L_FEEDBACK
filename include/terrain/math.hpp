#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace terrain {

constexpr float kPi = 3.14159265358979323846f;

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    constexpr Vec3() = default;
    constexpr Vec3(float xValue, float yValue, float zValue)
        : x(xValue), y(yValue), z(zValue) {}

    constexpr Vec3 operator+(const Vec3& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    constexpr Vec3 operator-(const Vec3& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    constexpr Vec3 operator*(float scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }

    constexpr Vec3 operator/(float scalar) const {
        return {x / scalar, y / scalar, z / scalar};
    }

    constexpr Vec3 operator-() const {
        return {-x, -y, -z};
    }
};

inline Vec3 operator*(float scalar, const Vec3& value) {
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

inline Vec3 normalize(const Vec3& value) {
    const float length = std::sqrt(lengthSquared(value));
    if (length <= std::numeric_limits<float>::epsilon()) {
        return {};
    }
    return value / length;
}

inline float clampf(float value, float low, float high) {
    return std::max(low, std::min(high, value));
}

inline float radians(float degrees) {
    return degrees * (kPi / 180.0f);
}

struct Mat4 {
    std::array<float, 16> m{};

    static Mat4 identity() {
        Mat4 result;
        result.m[0] = 1.0f;
        result.m[5] = 1.0f;
        result.m[10] = 1.0f;
        result.m[15] = 1.0f;
        return result;
    }

    const float& operator()(int row, int column) const {
        return m[static_cast<std::size_t>(column * 4 + row)];
    }

    float& operator()(int row, int column) {
        return m[static_cast<std::size_t>(column * 4 + row)];
    }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 result;
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

struct Plane {
    Vec3 normal;
    float distance = 0.0f;

    void normalizePlane() {
        const float length = std::sqrt(lengthSquared(normal));
        if (length > 0.0f) {
            normal = normal / length;
            distance /= length;
        }
    }
};

struct Aabb {
    Vec3 minimum;
    Vec3 maximum;

    Vec3 center() const {
        return (minimum + maximum) * 0.5f;
    }
};

struct Frustum {
    std::array<Plane, 6> planes{};

    static Frustum fromViewProjection(const Mat4& m) {
        Frustum result;
        const auto row = [&m](int index) {
            return Vec3(m(index, 0), m(index, 1), m(index, 2));
        };
        const auto rowW = [&m](int index) {
            return m(index, 3);
        };

        result.planes[0] = {row(3) + row(0), rowW(3) + rowW(0)};
        result.planes[1] = {row(3) - row(0), rowW(3) - rowW(0)};
        result.planes[2] = {row(3) + row(1), rowW(3) + rowW(1)};
        result.planes[3] = {row(3) - row(1), rowW(3) - rowW(1)};
        result.planes[4] = {row(3) + row(2), rowW(3) + rowW(2)};
        result.planes[5] = {row(3) - row(2), rowW(3) - rowW(2)};

        for (Plane& plane : result.planes) {
            plane.normalizePlane();
        }
        return result;
    }

    bool intersects(const Aabb& box) const {
        for (const Plane& plane : planes) {
            const Vec3 positive{
                plane.normal.x >= 0.0f ? box.maximum.x : box.minimum.x,
                plane.normal.y >= 0.0f ? box.maximum.y : box.minimum.y,
                plane.normal.z >= 0.0f ? box.maximum.z : box.minimum.z,
            };
            if (dot(plane.normal, positive) + plane.distance < 0.0f) {
                return false;
            }
        }
        return true;
    }
};

inline float distanceSquaredToAabb(const Vec3& point, const Aabb& box) {
    const auto axisDistance = [](float value, float low, float high) {
        if (value < low) {
            return low - value;
        }
        if (value > high) {
            return value - high;
        }
        return 0.0f;
    };

    const float dx = axisDistance(point.x, box.minimum.x, box.maximum.x);
    const float dy = axisDistance(point.y, box.minimum.y, box.maximum.y);
    const float dz = axisDistance(point.z, box.minimum.z, box.maximum.z);
    return dx * dx + dy * dy + dz * dz;
}

}  // namespace terrain

