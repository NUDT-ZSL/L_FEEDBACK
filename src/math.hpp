#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>

namespace dungeon {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kInfinity = std::numeric_limits<double>::infinity();

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator-(const Vec3& a) {
    return {-a.x, -a.y, -a.z};
}

inline Vec3 operator*(const Vec3& a, double s) {
    return {a.x * s, a.y * s, a.z * s};
}

inline Vec3 operator*(double s, const Vec3& a) {
    return a * s;
}

inline Vec3 operator*(const Vec3& a, const Vec3& b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

inline Vec3 operator/(const Vec3& a, double s) {
    return {a.x / s, a.y / s, a.z / s};
}

inline Vec3& operator+=(Vec3& a, const Vec3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

inline Vec3& operator*=(Vec3& a, double s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
    return a;
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

inline double lengthSquared(const Vec3& a) {
    return dot(a, a);
}

inline double length(const Vec3& a) {
    return std::sqrt(lengthSquared(a));
}

inline Vec3 normalize(const Vec3& a) {
    const double len = length(a);
    if (len <= 1e-30) {
        return {};
    }
    return a / len;
}

inline Vec3 clampColor(const Vec3& c, double low, double high) {
    return {
        std::clamp(c.x, low, high),
        std::clamp(c.y, low, high),
        std::clamp(c.z, low, high),
    };
}

struct Ray {
    Vec3 origin;
    Vec3 direction;
};

inline uint64_t fnv1aByte(uint64_t hash, uint8_t value) {
    hash ^= static_cast<uint64_t>(value);
    hash *= 1099511628211ULL;
    return hash;
}

inline uint64_t fnv1aU32(uint64_t hash, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        hash = fnv1aByte(hash, static_cast<uint8_t>((value >> shift) & 0xFFu));
    }
    return hash;
}

inline uint64_t fnv1aU64(uint64_t hash, uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        hash = fnv1aByte(hash, static_cast<uint8_t>((value >> shift) & 0xFFu));
    }
    return hash;
}

inline uint64_t hashDouble(uint64_t hash, double value) {
    static_assert(sizeof(double) == sizeof(uint64_t), "double must be 64-bit");
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return fnv1aU64(hash, bits);
}

inline uint64_t hashVec3(uint64_t hash, const Vec3& v) {
    hash = hashDouble(hash, v.x);
    hash = hashDouble(hash, v.y);
    return hashDouble(hash, v.z);
}

inline std::string hex64(uint64_t value) {
    std::ostringstream out;
    out << "0x";
    const char* digits = "0123456789abcdef";
    for (int shift = 60; shift >= 0; shift -= 4) {
        out << digits[(value >> shift) & 0xFu];
    }
    return out.str();
}

inline uint32_t quantizeLuminance(double r, double g, double b) {
    const double luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const double scaled = std::clamp(luma, 0.0, 1000.0) * 1000000.0;
    const auto rounded = static_cast<uint64_t>(scaled + 0.5);
    return static_cast<uint32_t>(std::min<uint64_t>(rounded, 0xFFFFFFFFULL));
}

struct Basis {
    Vec3 forward;
    Vec3 right;
    Vec3 up;
};

inline Basis makeBasis(const Vec3& direction) {
    Basis basis;
    basis.forward = normalize(direction);
    if (lengthSquared(basis.forward) <= 1e-30) {
        basis.forward = {0.0, 0.0, 1.0};
    }

    Vec3 referenceUp{0.0, 1.0, 0.0};
    if (std::abs(dot(basis.forward, referenceUp)) > 0.999) {
        referenceUp = {1.0, 0.0, 0.0};
    }
    basis.right = normalize(cross(referenceUp, basis.forward));
    basis.up = normalize(cross(basis.forward, basis.right));
    return basis;
}

inline double clamp01(double value) {
    return std::clamp(value, 0.0, 1.0);
}

inline double smoothStep01(double value) {
    const double t = clamp01(value);
    return t * t * (3.0 - 2.0 * t);
}

}  // namespace dungeon
