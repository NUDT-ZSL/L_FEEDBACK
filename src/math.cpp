#include "math.h"

#include <algorithm>

namespace dr {

Vec3 Vec3::operator+(const Vec3& other) const {
    return {x + other.x, y + other.y, z + other.z};
}

Vec3 Vec3::operator-(const Vec3& other) const {
    return {x - other.x, y - other.y, z - other.z};
}

Vec3 Vec3::operator*(float scalar) const {
    return {x * scalar, y * scalar, z * scalar};
}

Vec3& Vec3::operator+=(const Vec3& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}

Mat4 Mat4::identity() {
    Mat4 result;
    result.m = {1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f};
    return result;
}

Mat4 Mat4::perspective(float fovYDegrees, float aspect, float nearPlane, float farPlane) {
    const float f = 1.0f / std::tan(radians(fovYDegrees) * 0.5f);
    Mat4 result;
    result.m = {f / aspect, 0.0f, 0.0f, 0.0f,
                0.0f, f, 0.0f, 0.0f,
                0.0f, 0.0f, (farPlane + nearPlane) / (nearPlane - farPlane), -1.0f,
                0.0f, 0.0f, (2.0f * farPlane * nearPlane) / (nearPlane - farPlane), 0.0f};
    return result;
}

Mat4 Mat4::lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
    const Vec3 f = normalize(target - eye);
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);

    Mat4 result = identity();
    result.m = {s.x, u.x, -f.x, 0.0f,
                s.y, u.y, -f.y, 0.0f,
                s.z, u.z, -f.z, 0.0f,
                -dot(s, eye), -dot(u, eye), dot(f, eye), 1.0f};
    return result;
}

Mat4 Mat4::translationScale(const Vec3& translation, float scale) {
    Mat4 result = identity();
    result.m[0] = scale;
    result.m[5] = scale;
    result.m[10] = scale;
    result.m[12] = translation.x;
    result.m[13] = translation.y;
    result.m[14] = translation.z;
    return result;
}

Mat4 Mat4::operator*(const Mat4& other) const {
    Mat4 result;
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            float value = 0.0f;
            for (int k = 0; k < 4; ++k) {
                value += m[k * 4 + row] * other.m[column * 4 + k];
            }
            result.m[column * 4 + row] = value;
        }
    }
    return result;
}

float radians(float degrees) {
    return degrees * 3.14159265358979323846f / 180.0f;
}

float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

Vec3 normalize(const Vec3& value) {
    const float length = std::sqrt(dot(value, value));
    if (length <= 1.0e-8f) {
        return {};
    }
    return value * (1.0f / length);
}

} // namespace dr
