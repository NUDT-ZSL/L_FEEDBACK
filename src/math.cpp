#include "math.h"

#include <algorithm>
#include <limits>

namespace softraster {

Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
Vec2 operator*(Vec2 a, float scalar) { return {a.x * scalar, a.y * scalar}; }
Vec2 operator/(Vec2 a, float scalar) { return {a.x / scalar, a.y / scalar}; }

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
Vec3 operator*(Vec3 a, float scalar) { return {a.x * scalar, a.y * scalar, a.z * scalar}; }
Vec3 operator*(float scalar, Vec3 a) { return a * scalar; }
Vec3 operator/(Vec3 a, float scalar) { return {a.x / scalar, a.y / scalar, a.z / scalar}; }
Vec3 operator*(Vec3 a, Vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }

Vec4 operator+(Vec4 a, Vec4 b) { return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
Vec4 operator-(Vec4 a, Vec4 b) { return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }
Vec4 operator*(Vec4 a, float scalar) {
    return {a.x * scalar, a.y * scalar, a.z * scalar, a.w * scalar};
}

float dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
float lengthSquared(Vec3 value) { return dot(value, value); }
float length(Vec3 value) { return std::sqrt(lengthSquared(value)); }

Vec3 normalize(Vec3 value, Vec3 fallback) {
    const float len = length(value);
    if (!(len > 1.0e-12F) || !std::isfinite(len)) {
        return fallback;
    }
    return value / len;
}

Vec2 normalize(Vec2 value, Vec2 fallback) {
    const float len = std::sqrt(dot(value, value));
    if (!(len > 1.0e-12F) || !std::isfinite(len)) {
        return fallback;
    }
    return value / len;
}

Vec3 minComponents(Vec3 a, Vec3 b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

Vec3 maxComponents(Vec3 a, Vec3 b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

float clamp01(float value) { return std::clamp(value, 0.0F, 1.0F); }
Vec3 clamp01(Vec3 value) {
    return {clamp01(value.x), clamp01(value.y), clamp01(value.z)};
}

Mat4 Mat4::identity() {
    Mat4 result;
    result.m[0] = 1.0F;
    result.m[5] = 1.0F;
    result.m[10] = 1.0F;
    result.m[15] = 1.0F;
    return result;
}

Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 result;
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            float sum = 0.0F;
            for (int k = 0; k < 4; ++k) {
                sum += a.m[static_cast<std::size_t>(row * 4 + k)] *
                       b.m[static_cast<std::size_t>(k * 4 + column)];
            }
            result.m[static_cast<std::size_t>(row * 4 + column)] = sum;
        }
    }
    return result;
}

Vec4 operator*(const Mat4& matrix, Vec4 value) {
    return {
        matrix.m[0] * value.x + matrix.m[1] * value.y + matrix.m[2] * value.z +
            matrix.m[3] * value.w,
        matrix.m[4] * value.x + matrix.m[5] * value.y + matrix.m[6] * value.z +
            matrix.m[7] * value.w,
        matrix.m[8] * value.x + matrix.m[9] * value.y + matrix.m[10] * value.z +
            matrix.m[11] * value.w,
        matrix.m[12] * value.x + matrix.m[13] * value.y + matrix.m[14] * value.z +
            matrix.m[15] * value.w,
    };
}

Vec4 transformPoint(const Mat4& matrix, Vec3 point) {
    return matrix * Vec4{point.x, point.y, point.z, 1.0F};
}

Vec3 transformVector(const Mat4& matrix, Vec3 vector) {
    return {
        matrix.m[0] * vector.x + matrix.m[1] * vector.y + matrix.m[2] * vector.z,
        matrix.m[4] * vector.x + matrix.m[5] * vector.y + matrix.m[6] * vector.z,
        matrix.m[8] * vector.x + matrix.m[9] * vector.y + matrix.m[10] * vector.z,
    };
}

Mat4 transpose(const Mat4& matrix) {
    Mat4 result;
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            result.m[static_cast<std::size_t>(row * 4 + column)] =
                matrix.m[static_cast<std::size_t>(column * 4 + row)];
        }
    }
    return result;
}

Mat4 inverse(const Mat4& matrix, bool& invertible) {
    const float a00 = matrix.m[0];
    const float a01 = matrix.m[1];
    const float a02 = matrix.m[2];
    const float a03 = matrix.m[3];
    const float a10 = matrix.m[4];
    const float a11 = matrix.m[5];
    const float a12 = matrix.m[6];
    const float a13 = matrix.m[7];
    const float a20 = matrix.m[8];
    const float a21 = matrix.m[9];
    const float a22 = matrix.m[10];
    const float a23 = matrix.m[11];
    const float a30 = matrix.m[12];
    const float a31 = matrix.m[13];
    const float a32 = matrix.m[14];
    const float a33 = matrix.m[15];

    const float b00 = a00 * a11 - a01 * a10;
    const float b01 = a00 * a12 - a02 * a10;
    const float b02 = a00 * a13 - a03 * a10;
    const float b03 = a01 * a12 - a02 * a11;
    const float b04 = a01 * a13 - a03 * a11;
    const float b05 = a02 * a13 - a03 * a12;
    const float b06 = a20 * a31 - a21 * a30;
    const float b07 = a20 * a32 - a22 * a30;
    const float b08 = a20 * a33 - a23 * a30;
    const float b09 = a21 * a32 - a22 * a31;
    const float b10 = a21 * a33 - a23 * a31;
    const float b11 = a22 * a33 - a23 * a32;

    const float determinant =
        b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    invertible = std::isfinite(determinant) && std::abs(determinant) > 1.0e-12F;
    if (!invertible) {
        return Mat4::identity();
    }
    const float invDet = 1.0F / determinant;

    Mat4 result;
    result.m[0] = (a11 * b11 - a12 * b10 + a13 * b09) * invDet;
    result.m[1] = (-a01 * b11 + a02 * b10 - a03 * b09) * invDet;
    result.m[2] = (a31 * b05 - a32 * b04 + a33 * b03) * invDet;
    result.m[3] = (-a21 * b05 + a22 * b04 - a23 * b03) * invDet;
    result.m[4] = (-a10 * b11 + a12 * b08 - a13 * b07) * invDet;
    result.m[5] = (a00 * b11 - a02 * b08 + a03 * b07) * invDet;
    result.m[6] = (-a30 * b05 + a32 * b02 - a33 * b01) * invDet;
    result.m[7] = (a20 * b05 - a22 * b02 + a23 * b01) * invDet;
    result.m[8] = (a10 * b10 - a11 * b08 + a13 * b06) * invDet;
    result.m[9] = (-a00 * b10 + a01 * b08 - a03 * b06) * invDet;
    result.m[10] = (a30 * b04 - a31 * b02 + a33 * b00) * invDet;
    result.m[11] = (-a20 * b04 + a21 * b02 - a23 * b00) * invDet;
    result.m[12] = (-a10 * b09 + a11 * b07 - a12 * b06) * invDet;
    result.m[13] = (a00 * b09 - a01 * b07 + a02 * b06) * invDet;
    result.m[14] = (-a30 * b03 + a31 * b01 - a32 * b00) * invDet;
    result.m[15] = (a20 * b03 - a21 * b01 + a22 * b00) * invDet;
    return result;
}

Mat4 translation(Vec3 value) {
    Mat4 result = Mat4::identity();
    result.m[3] = value.x;
    result.m[7] = value.y;
    result.m[11] = value.z;
    return result;
}

Mat4 scale(Vec3 value) {
    Mat4 result = Mat4::identity();
    result.m[0] = value.x;
    result.m[5] = value.y;
    result.m[10] = value.z;
    return result;
}

Mat4 rotationX(float radiansValue) {
    Mat4 result = Mat4::identity();
    const float c = std::cos(radiansValue);
    const float s = std::sin(radiansValue);
    result.m[5] = c;
    result.m[6] = -s;
    result.m[9] = s;
    result.m[10] = c;
    return result;
}

Mat4 rotationY(float radiansValue) {
    Mat4 result = Mat4::identity();
    const float c = std::cos(radiansValue);
    const float s = std::sin(radiansValue);
    result.m[0] = c;
    result.m[2] = s;
    result.m[8] = -s;
    result.m[10] = c;
    return result;
}

Mat4 rotationZ(float radiansValue) {
    Mat4 result = Mat4::identity();
    const float c = std::cos(radiansValue);
    const float s = std::sin(radiansValue);
    result.m[0] = c;
    result.m[1] = -s;
    result.m[4] = s;
    result.m[5] = c;
    return result;
}

Mat4 lookAtRh(Vec3 eye, Vec3 target, Vec3 up, bool& valid) {
    const Vec3 forward = normalize(target - eye, {0.0F, 0.0F, 0.0F});
    const Vec3 side = normalize(cross(forward, up), {0.0F, 0.0F, 0.0F});
    const Vec3 correctedUp = cross(side, forward);
    valid = lengthSquared(forward) > 0.0F && lengthSquared(side) > 0.0F;
    if (!valid) {
        return Mat4::identity();
    }
    Mat4 result = Mat4::identity();
    result.m[0] = side.x;
    result.m[1] = side.y;
    result.m[2] = side.z;
    result.m[3] = -dot(side, eye);
    result.m[4] = correctedUp.x;
    result.m[5] = correctedUp.y;
    result.m[6] = correctedUp.z;
    result.m[7] = -dot(correctedUp, eye);
    result.m[8] = -forward.x;
    result.m[9] = -forward.y;
    result.m[10] = -forward.z;
    result.m[11] = dot(forward, eye);
    return result;
}

Mat4 perspectiveRh(float fovYDegrees, float aspect, float nearPlane, float farPlane) {
    const float f = 1.0F / std::tan(static_cast<float>(radians(fovYDegrees)) * 0.5F);
    Mat4 result;
    result.m[0] = f / aspect;
    result.m[5] = f;
    result.m[10] = (farPlane + nearPlane) / (nearPlane - farPlane);
    result.m[11] = (2.0F * farPlane * nearPlane) / (nearPlane - farPlane);
    result.m[14] = -1.0F;
    return result;
}

Mat4 orthographicRh(float left, float right, float bottom, float top, float nearPlane,
                    float farPlane) {
    Mat4 result = Mat4::identity();
    result.m[0] = 2.0F / (right - left);
    result.m[5] = 2.0F / (top - bottom);
    result.m[10] = -2.0F / (farPlane - nearPlane);
    result.m[3] = -(right + left) / (right - left);
    result.m[7] = -(top + bottom) / (top - bottom);
    result.m[11] = -(farPlane + nearPlane) / (farPlane - nearPlane);
    return result;
}

bool isFinite(Vec2 value) { return std::isfinite(value.x) && std::isfinite(value.y); }
bool isFinite(Vec3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
bool isFinite(const Mat4& value) {
    for (const float element : value.m) {
        if (!std::isfinite(element)) {
            return false;
        }
    }
    return true;
}

double radians(double degrees) { return degrees * 3.1415926535897932384626433832795 / 180.0; }

}  // namespace softraster
