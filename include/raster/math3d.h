#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace raster {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kEpsilon = 1.0e-12;

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Vec4 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 1.0;
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

inline Vec3 operator*(const Vec3& v, double s) {
    return {v.x * s, v.y * s, v.z * s};
}

inline Vec3 operator*(double s, const Vec3& v) {
    return v * s;
}

inline Vec3 operator*(const Vec3& a, const Vec3& b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

inline Vec3 operator/(const Vec3& v, double s) {
    return {v.x / s, v.y / s, v.z / s};
}

inline Vec3& operator+=(Vec3& a, const Vec3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
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

inline double lengthSquared(const Vec3& v) {
    return dot(v, v);
}

inline double length(const Vec3& v) {
    return std::sqrt(lengthSquared(v));
}

inline Vec3 normalize(const Vec3& v) {
    const double len = length(v);
    if (!(len > kEpsilon) || !std::isfinite(len)) {
        return {};
    }
    return v / len;
}

inline Vec3 minComponents(const Vec3& a, const Vec3& b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

inline Vec3 maxComponents(const Vec3& a, const Vec3& b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

inline Vec3 clamp01(const Vec3& v) {
    return {
        std::clamp(v.x, 0.0, 1.0),
        std::clamp(v.y, 0.0, 1.0),
        std::clamp(v.z, 0.0, 1.0),
    };
}

inline Vec4 clamp01(const Vec4& v) {
    return {
        std::clamp(v.x, 0.0, 1.0),
        std::clamp(v.y, 0.0, 1.0),
        std::clamp(v.z, 0.0, 1.0),
        std::clamp(v.w, 0.0, 1.0),
    };
}

inline bool isFinite(const Vec2& v) {
    return std::isfinite(v.x) && std::isfinite(v.y);
}

inline bool isFinite(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

inline bool isFinite(const Vec4& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) &&
           std::isfinite(v.w);
}

inline Vec3 lerp(const Vec3& a, const Vec3& b, double t) {
    return a + (b - a) * t;
}

inline Vec2 lerp(const Vec2& a, const Vec2& b, double t) {
    return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t};
}

inline Vec4 lerp(const Vec4& a, const Vec4& b, double t) {
    return {
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
        a.w + (b.w - a.w) * t,
    };
}

struct Mat4 {
    std::array<double, 16> m{};

    static Mat4 identity() {
        Mat4 result;
        result.m = {1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0};
        return result;
    }

    double& at(std::size_t row, std::size_t column) {
        return m[row * 4 + column];
    }

    double at(std::size_t row, std::size_t column) const {
        return m[row * 4 + column];
    }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 result;
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            double sum = 0.0;
            for (std::size_t k = 0; k < 4; ++k) {
                sum += a.at(row, k) * b.at(k, column);
            }
            result.at(row, column) = sum;
        }
    }
    return result;
}

inline Vec4 transform(const Mat4& matrix, const Vec4& v) {
    return {
        matrix.at(0, 0) * v.x + matrix.at(0, 1) * v.y +
            matrix.at(0, 2) * v.z + matrix.at(0, 3) * v.w,
        matrix.at(1, 0) * v.x + matrix.at(1, 1) * v.y +
            matrix.at(1, 2) * v.z + matrix.at(1, 3) * v.w,
        matrix.at(2, 0) * v.x + matrix.at(2, 1) * v.y +
            matrix.at(2, 2) * v.z + matrix.at(2, 3) * v.w,
        matrix.at(3, 0) * v.x + matrix.at(3, 1) * v.y +
            matrix.at(3, 2) * v.z + matrix.at(3, 3) * v.w,
    };
}

inline Vec3 transformPoint(const Mat4& matrix, const Vec3& point) {
    const Vec4 result = transform(matrix, {point.x, point.y, point.z, 1.0});
    if (std::abs(result.w) > kEpsilon && result.w != 1.0) {
        return {result.x / result.w, result.y / result.w, result.z / result.w};
    }
    return {result.x, result.y, result.z};
}

inline Vec3 transformVector(const Mat4& matrix, const Vec3& vector) {
    const Vec4 result = transform(matrix, {vector.x, vector.y, vector.z, 0.0});
    return {result.x, result.y, result.z};
}

inline bool invert(const Mat4& input, Mat4& output) {
    Mat4 augmented = input;
    output = Mat4::identity();

    for (std::size_t column = 0; column < 4; ++column) {
        std::size_t pivot = column;
        double pivotMagnitude = std::abs(augmented.at(pivot, column));
        for (std::size_t row = column + 1; row < 4; ++row) {
            const double magnitude = std::abs(augmented.at(row, column));
            if (magnitude > pivotMagnitude) {
                pivot = row;
                pivotMagnitude = magnitude;
            }
        }

        if (!(pivotMagnitude > 1.0e-15) || !std::isfinite(pivotMagnitude)) {
            return false;
        }

        if (pivot != column) {
            for (std::size_t k = 0; k < 4; ++k) {
                std::swap(augmented.at(column, k), augmented.at(pivot, k));
                std::swap(output.at(column, k), output.at(pivot, k));
            }
        }

        const double divisor = augmented.at(column, column);
        for (std::size_t k = 0; k < 4; ++k) {
            augmented.at(column, k) /= divisor;
            output.at(column, k) /= divisor;
        }

        for (std::size_t row = 0; row < 4; ++row) {
            if (row == column) {
                continue;
            }
            const double factor = augmented.at(row, column);
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t k = 0; k < 4; ++k) {
                augmented.at(row, k) -= factor * augmented.at(column, k);
                output.at(row, k) -= factor * output.at(column, k);
            }
        }
    }

    return true;
}

inline bool isFinite(const Mat4& matrix) {
    for (double value : matrix.m) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

inline Mat4 normalMatrix(const Mat4& model) {
    Mat4 inverse;
    if (!invert(model, inverse)) {
        return Mat4::identity();
    }
    Mat4 transpose;
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            transpose.at(row, column) = inverse.at(column, row);
        }
    }
    return transpose;
}

inline Mat4 lookAtLH(const Vec3& position, const Vec3& forward,
                     const Vec3& up) {
    const Vec3 zAxis = normalize(forward);
    const Vec3 xAxis = normalize(cross(up, zAxis));
    const Vec3 yAxis = cross(zAxis, xAxis);

    Mat4 result = Mat4::identity();
    result.at(0, 0) = xAxis.x;
    result.at(0, 1) = xAxis.y;
    result.at(0, 2) = xAxis.z;
    result.at(0, 3) = -dot(xAxis, position);
    result.at(1, 0) = yAxis.x;
    result.at(1, 1) = yAxis.y;
    result.at(1, 2) = yAxis.z;
    result.at(1, 3) = -dot(yAxis, position);
    result.at(2, 0) = zAxis.x;
    result.at(2, 1) = zAxis.y;
    result.at(2, 2) = zAxis.z;
    result.at(2, 3) = -dot(zAxis, position);
    return result;
}

inline Mat4 perspectiveFovLH(double fovYRadians, double aspect, double nearPlane,
                             double farPlane) {
    Mat4 result{};
    const double yScale = 1.0 / std::tan(fovYRadians * 0.5);
    const double xScale = yScale / aspect;
    result.at(0, 0) = xScale;
    result.at(1, 1) = yScale;
    result.at(2, 2) = farPlane / (farPlane - nearPlane);
    result.at(2, 3) = -nearPlane * farPlane / (farPlane - nearPlane);
    result.at(3, 2) = 1.0;
    result.at(3, 3) = 0.0;
    return result;
}

}  // namespace raster

