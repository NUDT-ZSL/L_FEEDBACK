#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace skinbake {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Vec2 {
    double x = 0.0;
    double y = 0.0;

    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}
};

inline Vec2 operator+(const Vec2& a, const Vec2& b) {
    return {a.x + b.x, a.y + b.y};
}
inline Vec2 operator-(const Vec2& a, const Vec2& b) {
    return {a.x - b.x, a.y - b.y};
}
inline Vec2 operator*(const Vec2& a, double s) {
    return {a.x * s, a.y * s};
}
inline bool is_finite(const Vec2& v) {
    return std::isfinite(v.x) && std::isfinite(v.y);
}

struct Mat3 {
    std::array<double, 9> m{1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0};

    static Mat3 identity() { return Mat3{}; }

    double at(std::size_t row, std::size_t col) const {
        return m[row * 3 + col];
    }
    void set(std::size_t row, std::size_t col, double value) {
        m[row * 3 + col] = value;
    }

    Vec2 transform_point(const Vec2& p) const {
        return {at(0, 0) * p.x + at(0, 1) * p.y + at(0, 2),
                at(1, 0) * p.x + at(1, 1) * p.y + at(1, 2)};
    }
};

inline Mat3 operator*(const Mat3& a, const Mat3& b) {
    Mat3 out;
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            double value = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                value += a.at(r, k) * b.at(k, c);
            }
            out.set(r, c, value);
        }
    }
    return out;
}

struct Transform {
    Vec2 translation{0.0, 0.0};
    double rotationRadians = 0.0;
    Vec2 scale{1.0, 1.0};

    Mat3 to_matrix() const {
        const double c = std::cos(rotationRadians);
        const double s = std::sin(rotationRadians);
        Mat3 out;
        out.m = {scale.x * c, -scale.y * s, translation.x,
                 scale.x * s,  scale.y * c, translation.y,
                 0.0,          0.0,         1.0};
        return out;
    }
};

inline bool is_finite(const Mat3& m) {
    for (double v : m.m) {
        if (!std::isfinite(v)) {
            return false;
        }
    }
    return true;
}

inline Mat3 lerp(const Mat3& a, const Mat3& b, double t) {
    Mat3 out;
    for (std::size_t i = 0; i < out.m.size(); ++i) {
        out.m[i] = a.m[i] + (b.m[i] - a.m[i]) * t;
    }
    return out;
}

inline Vec2 lerp(const Vec2& a, const Vec2& b, double t) {
    return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t};
}

inline std::array<float, 12> matrix_to_float12(const Mat3& m) {
    return {
        static_cast<float>(m.at(0, 0)), static_cast<float>(m.at(0, 1)),
        static_cast<float>(m.at(0, 2)), 0.0f,
        static_cast<float>(m.at(1, 0)), static_cast<float>(m.at(1, 1)),
        static_cast<float>(m.at(1, 2)), 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
}

inline Mat3 float12_to_matrix(const float* values) {
    Mat3 out;
    out.set(0, 0, values[0]);
    out.set(0, 1, values[1]);
    out.set(0, 2, values[2]);
    out.set(1, 0, values[4]);
    out.set(1, 1, values[5]);
    out.set(1, 2, values[6]);
    out.set(2, 0, values[8]);
    out.set(2, 1, values[9]);
    out.set(2, 2, values[10]);
    return out;
}

}

