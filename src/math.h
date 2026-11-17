#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace ct {

constexpr double kPi = 3.14159265358979323846;

struct Vec2 {
    double x = 0.0;
    double y = 0.0;

    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}

    Vec2 operator+(const Vec2& rhs) const { return {x + rhs.x, y + rhs.y}; }
    Vec2 operator-(const Vec2& rhs) const { return {x - rhs.x, y - rhs.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    Vec2 operator/(double s) const { return {x / s, y / s}; }
    Vec2& operator+=(const Vec2& rhs) {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }
};

inline double dot(const Vec2& a, const Vec2& b) {
    return a.x * b.x + a.y * b.y;
}

inline double length(const Vec2& v) {
    return std::sqrt(dot(v, v));
}

inline Vec2 normalize(const Vec2& v) {
    const double len = length(v);
    return len > 1e-20 ? v / len : Vec2{};
}

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& rhs) const { return {x + rhs.x, y + rhs.y, z + rhs.z}; }
    Vec3 operator-(const Vec3& rhs) const { return {x - rhs.x, y - rhs.y, z - rhs.z}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
    Vec3& operator+=(const Vec3& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }
    Vec3& operator*=(double s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }
};

inline Vec3 operator*(double s, const Vec3& v) {
    return v * s;
}

inline double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

inline double length(const Vec3& v) {
    return std::sqrt(dot(v, v));
}

inline Vec3 normalize(const Vec3& v) {
    const double len = length(v);
    return len > 1e-20 ? v / len : Vec3{};
}

inline Vec3 reflect(const Vec3& incident, const Vec3& normal) {
    return incident - normal * (2.0 * dot(incident, normal));
}

inline Vec3 clampColor(const Vec3& v) {
    return {
        std::clamp(v.x, 0.0, 1.0),
        std::clamp(v.y, 0.0, 1.0),
        std::clamp(v.z, 0.0, 1.0),
    };
}

struct Vec4 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 0.0;

    Vec4() = default;
    Vec4(double x_, double y_, double z_, double w_) : x(x_), y(y_), z(z_), w(w_) {}
    Vec4(const Vec3& v, double w_) : x(v.x), y(v.y), z(v.z), w(w_) {}

    Vec4 operator+(const Vec4& rhs) const {
        return {x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w};
    }
    Vec4 operator-(const Vec4& rhs) const {
        return {x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w};
    }
    Vec4 operator*(double s) const { return {x * s, y * s, z * s, w * s}; }
};

inline double dot(const Vec4& a, const Vec4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

struct Mat4 {
    double m[4][4]{};

    static Mat4 identity() {
        Mat4 r;
        r.m[0][0] = 1.0;
        r.m[1][1] = 1.0;
        r.m[2][2] = 1.0;
        r.m[3][3] = 1.0;
        return r;
    }
};

inline Vec4 operator*(const Mat4& a, const Vec4& v) {
    return {
        a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z + a.m[0][3] * v.w,
        a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z + a.m[1][3] * v.w,
        a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z + a.m[2][3] * v.w,
        a.m[3][0] * v.x + a.m[3][1] * v.y + a.m[3][2] * v.z + a.m[3][3] * v.w,
    };
}

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            for (int k = 0; k < 4; ++k) {
                r.m[row][col] += a.m[row][k] * b.m[k][col];
            }
        }
    }
    return r;
}

inline Mat4 lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
    const Vec3 f = normalize(target - eye);
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);

    Mat4 r = Mat4::identity();
    r.m[0][0] = s.x;
    r.m[0][1] = s.y;
    r.m[0][2] = s.z;
    r.m[0][3] = -dot(s, eye);
    r.m[1][0] = u.x;
    r.m[1][1] = u.y;
    r.m[1][2] = u.z;
    r.m[1][3] = -dot(u, eye);
    r.m[2][0] = -f.x;
    r.m[2][1] = -f.y;
    r.m[2][2] = -f.z;
    r.m[2][3] = dot(f, eye);
    return r;
}

inline Mat4 perspective(double fov_y_radians, double aspect, double near_plane, double far_plane) {
    const double f = 1.0 / std::tan(fov_y_radians * 0.5);
    Mat4 r;
    r.m[0][0] = f / aspect;
    r.m[1][1] = f;
    r.m[2][2] = (far_plane + near_plane) / (near_plane - far_plane);
    r.m[2][3] = (2.0 * far_plane * near_plane) / (near_plane - far_plane);
    r.m[3][2] = -1.0;
    return r;
}

inline Vec3 transformPoint(const Mat4& m, const Vec3& p) {
    const Vec4 r = m * Vec4(p, 1.0);
    if (std::abs(r.w) < 1e-20) {
        return {r.x, r.y, r.z};
    }
    return {r.x / r.w, r.y / r.w, r.z / r.w};
}

inline Vec3 minVec(const Vec3& a, const Vec3& b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

inline Vec3 maxVec(const Vec3& a, const Vec3& b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

}  // namespace ct
