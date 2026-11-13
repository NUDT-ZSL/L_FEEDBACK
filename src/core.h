#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& other) const { return {x + other.x, y + other.y, z + other.z}; }
    Vec3 operator-(const Vec3& other) const { return {x - other.x, y - other.y, z - other.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

inline double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline double lengthSquared(const Vec3& v) { return dot(v, v); }
inline double length(const Vec3& v) { return std::sqrt(lengthSquared(v)); }

inline Vec3 normalized(const Vec3& v) {
    const double len = length(v);
    if (!(len > 1e-15) || !std::isfinite(len)) {
        return {};
    }
    return v / len;
}

inline bool finite(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

inline double clamp01(double v) {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

struct Vec4 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 0.0;

    Vec4() = default;
    Vec4(double x_, double y_, double z_, double w_) : x(x_), y(y_), z(z_), w(w_) {}
};

struct Mat4 {
    // Column-major, matching the layout used by OpenGL-style projection matrices.
    double m[16]{};

    static Mat4 identity() {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0;
        return r;
    }

    double& at(int row, int col) { return m[col * 4 + row]; }
    double at(int row, int col) const { return m[col * 4 + row]; }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            double v = 0.0;
            for (int k = 0; k < 4; ++k) {
                v += a.at(row, k) * b.at(k, col);
            }
            r.at(row, col) = v;
        }
    }
    return r;
}

inline Vec4 operator*(const Mat4& m, const Vec4& v) {
    return {
        m.at(0, 0) * v.x + m.at(0, 1) * v.y + m.at(0, 2) * v.z + m.at(0, 3) * v.w,
        m.at(1, 0) * v.x + m.at(1, 1) * v.y + m.at(1, 2) * v.z + m.at(1, 3) * v.w,
        m.at(2, 0) * v.x + m.at(2, 1) * v.y + m.at(2, 2) * v.z + m.at(2, 3) * v.w,
        m.at(3, 0) * v.x + m.at(3, 1) * v.y + m.at(3, 2) * v.z + m.at(3, 3) * v.w,
    };
}

Vec3 transformPoint(const Mat4& m, const Vec3& p);
Vec3 transformVector(const Mat4& m, const Vec3& v);
Mat4 makeTRS(const Vec3& translation, const Vec3& rotationDeg, const Vec3& scale);

struct RenderError : std::runtime_error {
    std::string code;
    int exitCode;

    RenderError(std::string code_, std::string message_, int exitCode_ = 2)
        : std::runtime_error(std::move(message_)), code(std::move(code_)), exitCode(exitCode_) {}
};

uint64_t fnv1a64(const void* data, size_t size, uint64_t seed = 1469598103934665603ULL);
std::string hex64(uint64_t value);

std::string formatDouble(double value, int precision = 6);

struct Framebuffer {
    int width = 0;
    int height = 0;
    int tileSize = 16;
    int tilesX = 0;
    int tilesY = 0;
    std::vector<uint8_t> color;
    std::vector<uint32_t> depth;

    Framebuffer() = default;
    Framebuffer(int width_, int height_, int tileSize_);

    int tileCount() const { return tilesX * tilesY; }
    int tileIndex(int tx, int ty) const { return ty * tilesX + tx; }
    void tileBounds(int index, int& x0, int& y0, int& x1, int& y1) const;
    uint64_t colorHash() const;
    uint64_t depthHash() const;
    uint64_t tileColorHash(int index) const;
};
