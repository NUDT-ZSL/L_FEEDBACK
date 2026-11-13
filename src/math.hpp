#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace rigskin {

constexpr double kPi = 3.141592653589793238462643383279502884;

inline double radians(double degrees) {
    return degrees * kPi / 180.0;
}

inline bool finite(double value) {
    return std::isfinite(value);
}

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Vec3() = default;
    Vec3(double xValue, double yValue, double zValue)
        : x(xValue), y(yValue), z(zValue) {}

    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    Vec3& operator*=(double scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }
};

inline Vec3 operator+(Vec3 lhs, const Vec3& rhs) {
    lhs += rhs;
    return lhs;
}

inline Vec3 operator-(Vec3 lhs, const Vec3& rhs) {
    lhs -= rhs;
    return lhs;
}

inline Vec3 operator-(const Vec3& value) {
    return {-value.x, -value.y, -value.z};
}

inline Vec3 operator*(Vec3 value, double scalar) {
    value *= scalar;
    return value;
}

inline Vec3 operator*(double scalar, Vec3 value) {
    value *= scalar;
    return value;
}

inline Vec3 operator/(Vec3 value, double scalar) {
    return value * (1.0 / scalar);
}

inline double dot(const Vec3& lhs, const Vec3& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline Vec3 cross(const Vec3& lhs, const Vec3& rhs) {
    return {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x,
    };
}

inline double lengthSquared(const Vec3& value) {
    return dot(value, value);
}

inline double length(const Vec3& value) {
    return std::sqrt(lengthSquared(value));
}

inline Vec3 normalized(const Vec3& value) {
    const double valueLength = length(value);
    if (valueLength <= 1e-30 || !finite(valueLength)) {
        return {0.0, 0.0, 0.0};
    }
    return value / valueLength;
}

inline Vec3 lerp(const Vec3& lhs, const Vec3& rhs, double t) {
    return lhs * (1.0 - t) + rhs * t;
}

struct Quat {
    double w = 1.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Quat() = default;
    Quat(double wValue, double xValue, double yValue, double zValue)
        : w(wValue), x(xValue), y(yValue), z(zValue) {}
};

inline Quat operator*(const Quat& lhs, const Quat& rhs) {
    return {
        lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
        lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
        lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w,
    };
}

inline Quat normalized(const Quat& value) {
    const double norm = std::sqrt(
        value.w * value.w + value.x * value.x + value.y * value.y + value.z * value.z);
    if (norm <= 1e-30 || !finite(norm)) {
        return {};
    }
    return {value.w / norm, value.x / norm, value.y / norm, value.z / norm};
}

inline Quat conjugate(const Quat& value) {
    return {value.w, -value.x, -value.y, -value.z};
}

inline Quat axisAngle(const Vec3& axis, double angleRadians) {
    const Vec3 unitAxis = normalized(axis);
    const double halfAngle = angleRadians * 0.5;
    const double sine = std::sin(halfAngle);
    return normalized({
        std::cos(halfAngle),
        unitAxis.x * sine,
        unitAxis.y * sine,
        unitAxis.z * sine,
    });
}

inline Quat slerp(const Quat& rawLhs, const Quat& rawRhs, double t) {
    Quat lhs = normalized(rawLhs);
    Quat rhs = normalized(rawRhs);
    double cosine =
        lhs.w * rhs.w + lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;

    if (cosine < 0.0) {
        rhs = {-rhs.w, -rhs.x, -rhs.y, -rhs.z};
        cosine = -cosine;
    }

    if (cosine > 0.9995) {
        return normalized({
            lhs.w + t * (rhs.w - lhs.w),
            lhs.x + t * (rhs.x - lhs.x),
            lhs.y + t * (rhs.y - lhs.y),
            lhs.z + t * (rhs.z - lhs.z),
        });
    }

    cosine = std::max(-1.0, std::min(1.0, cosine));
    const double angle = std::acos(cosine);
    const double sine = std::sin(angle);
    const double lhsWeight = std::sin((1.0 - t) * angle) / sine;
    const double rhsWeight = std::sin(t * angle) / sine;
    return normalized({
        lhs.w * lhsWeight + rhs.w * rhsWeight,
        lhs.x * lhsWeight + rhs.x * rhsWeight,
        lhs.y * lhsWeight + rhs.y * rhsWeight,
        lhs.z * lhsWeight + rhs.z * rhsWeight,
    });
}

struct Mat4;
inline Mat4 operator*(const Mat4& lhs, const Mat4& rhs);

struct Mat4 {
    double m[4][4]{};

    static Mat4 identity() {
        Mat4 result;
        for (int row = 0; row < 4; ++row) {
            result.m[row][row] = 1.0;
        }
        return result;
    }

    static Mat4 translation(const Vec3& value) {
        Mat4 result = identity();
        result.m[0][3] = value.x;
        result.m[1][3] = value.y;
        result.m[2][3] = value.z;
        return result;
    }

    static Mat4 rotation(const Quat& rawValue) {
        const Quat value = normalized(rawValue);
        const double xx = value.x * value.x;
        const double yy = value.y * value.y;
        const double zz = value.z * value.z;
        const double xy = value.x * value.y;
        const double xz = value.x * value.z;
        const double yz = value.y * value.z;
        const double wx = value.w * value.x;
        const double wy = value.w * value.y;
        const double wz = value.w * value.z;

        Mat4 result = identity();
        result.m[0][0] = 1.0 - 2.0 * (yy + zz);
        result.m[0][1] = 2.0 * (xy - wz);
        result.m[0][2] = 2.0 * (xz + wy);
        result.m[1][0] = 2.0 * (xy + wz);
        result.m[1][1] = 1.0 - 2.0 * (xx + zz);
        result.m[1][2] = 2.0 * (yz - wx);
        result.m[2][0] = 2.0 * (xz - wy);
        result.m[2][1] = 2.0 * (yz + wx);
        result.m[2][2] = 1.0 - 2.0 * (xx + yy);
        return result;
    }

    static Mat4 localTransform(const Vec3& translationValue, const Quat& rotationValue) {
        return translation(translationValue) * rotation(rotationValue);
    }
};

inline Mat4 operator*(const Mat4& lhs, const Mat4& rhs) {
    Mat4 result;
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            double value = 0.0;
            for (int inner = 0; inner < 4; ++inner) {
                value += lhs.m[row][inner] * rhs.m[inner][column];
            }
            result.m[row][column] = value;
        }
    }
    return result;
}

inline Vec3 transformPoint(const Mat4& matrix, const Vec3& point) {
    const double x =
        matrix.m[0][0] * point.x + matrix.m[0][1] * point.y +
        matrix.m[0][2] * point.z + matrix.m[0][3];
    const double y =
        matrix.m[1][0] * point.x + matrix.m[1][1] * point.y +
        matrix.m[1][2] * point.z + matrix.m[1][3];
    const double z =
        matrix.m[2][0] * point.x + matrix.m[2][1] * point.y +
        matrix.m[2][2] * point.z + matrix.m[2][3];
    const double w =
        matrix.m[3][0] * point.x + matrix.m[3][1] * point.y +
        matrix.m[3][2] * point.z + matrix.m[3][3];
    if (std::abs(w) > 1e-30) {
        return {x / w, y / w, z / w};
    }
    return {x, y, z};
}

inline Vec3 transformVector(const Mat4& matrix, const Vec3& vector) {
    return {
        matrix.m[0][0] * vector.x + matrix.m[0][1] * vector.y +
            matrix.m[0][2] * vector.z,
        matrix.m[1][0] * vector.x + matrix.m[1][1] * vector.y +
            matrix.m[1][2] * vector.z,
        matrix.m[2][0] * vector.x + matrix.m[2][1] * vector.y +
            matrix.m[2][2] * vector.z,
    };
}

inline Mat4 inverse(const Mat4& matrix) {
    std::array<std::array<double, 8>, 4> augmented{};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            augmented[row][column] = matrix.m[row][column];
        }
        augmented[row][4 + row] = 1.0;
    }

    for (int pivotColumn = 0; pivotColumn < 4; ++pivotColumn) {
        int pivotRow = pivotColumn;
        double pivotMagnitude = std::abs(augmented[pivotRow][pivotColumn]);
        for (int row = pivotColumn + 1; row < 4; ++row) {
            const double magnitude = std::abs(augmented[row][pivotColumn]);
            if (magnitude > pivotMagnitude) {
                pivotMagnitude = magnitude;
                pivotRow = row;
            }
        }
        if (pivotMagnitude <= 1e-14) {
            throw std::runtime_error("singular matrix cannot be inverted");
        }
        if (pivotRow != pivotColumn) {
            std::swap(augmented[pivotRow], augmented[pivotColumn]);
        }

        const double pivot = augmented[pivotColumn][pivotColumn];
        for (int column = 0; column < 8; ++column) {
            augmented[pivotColumn][column] /= pivot;
        }

        for (int row = 0; row < 4; ++row) {
            if (row == pivotColumn) {
                continue;
            }
            const double factor = augmented[row][pivotColumn];
            if (std::abs(factor) <= 0.0) {
                continue;
            }
            for (int column = 0; column < 8; ++column) {
                augmented[row][column] -= factor * augmented[pivotColumn][column];
            }
        }
    }

    Mat4 result = Mat4::identity();
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            result.m[row][column] = augmented[row][4 + column];
        }
    }
    return result;
}

struct Bounds {
    Vec3 minimum{0.0, 0.0, 0.0};
    Vec3 maximum{0.0, 0.0, 0.0};
    bool valid = false;

    void include(const Vec3& point) {
        if (!valid) {
            minimum = point;
            maximum = point;
            valid = true;
            return;
        }
        minimum.x = std::min(minimum.x, point.x);
        minimum.y = std::min(minimum.y, point.y);
        minimum.z = std::min(minimum.z, point.z);
        maximum.x = std::max(maximum.x, point.x);
        maximum.y = std::max(maximum.y, point.y);
        maximum.z = std::max(maximum.z, point.z);
    }

    Vec3 center() const {
        return (minimum + maximum) * 0.5;
    }
};

}  // namespace rigskin

