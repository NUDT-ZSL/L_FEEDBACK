#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace orv {

constexpr float kPi = 3.14159265358979323846f;

struct Vec2 {
  float x = 0.0f;
  float y = 0.0f;
};

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;

  Vec3 operator+(const Vec3& other) const { return {x + other.x, y + other.y, z + other.z}; }
  Vec3 operator-(const Vec3& other) const { return {x - other.x, y - other.y, z - other.z}; }
  Vec3 operator*(float scalar) const { return {x * scalar, y * scalar, z * scalar}; }
  Vec3 operator/(float scalar) const { return {x / scalar, y / scalar, z / scalar}; }
  Vec3& operator+=(const Vec3& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }
};

inline float dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline float lengthSquared(const Vec3& value) { return dot(value, value); }
inline float length(const Vec3& value) { return std::sqrt(lengthSquared(value)); }

inline Vec3 normalize(const Vec3& value) {
  const float magnitude = length(value);
  if (magnitude <= 1.0e-8f) {
    return {};
  }
  return value / magnitude;
}

inline bool finite(const Vec3& value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

struct Mat4 {
  std::array<float, 16> m{};

  static Mat4 identity() {
    Mat4 result;
    result.m = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    return result;
  }

  const float* data() const { return m.data(); }
  float* data() { return m.data(); }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
  Mat4 result;
  for (int column = 0; column < 4; ++column) {
    for (int row = 0; row < 4; ++row) {
      float value = 0.0f;
      for (int k = 0; k < 4; ++k) {
        value += a.m[k * 4 + row] * b.m[column * 4 + k];
      }
      result.m[column * 4 + row] = value;
    }
  }
  return result;
}

inline Vec3 transformPoint(const Mat4& matrix, const Vec3& point) {
  const float x = matrix.m[0] * point.x + matrix.m[4] * point.y + matrix.m[8] * point.z +
                  matrix.m[12];
  const float y = matrix.m[1] * point.x + matrix.m[5] * point.y + matrix.m[9] * point.z +
                  matrix.m[13];
  const float z = matrix.m[2] * point.x + matrix.m[6] * point.y + matrix.m[10] * point.z +
                  matrix.m[14];
  const float w = matrix.m[3] * point.x + matrix.m[7] * point.y + matrix.m[11] * point.z +
                  matrix.m[15];
  if (std::abs(w) > 1.0e-8f) {
    return {x / w, y / w, z / w};
  }
  return {x, y, z};
}

inline Mat4 perspective(float fovYRadians, float aspect, float nearPlane, float farPlane) {
  const float f = 1.0f / std::tan(fovYRadians * 0.5f);
  Mat4 result;
  result.m = {f / aspect, 0, 0, 0,
              0, f, 0, 0,
              0, 0, (farPlane + nearPlane) / (nearPlane - farPlane), -1,
              0, 0, (2.0f * farPlane * nearPlane) / (nearPlane - farPlane), 0};
  return result;
}

inline Mat4 orthographic(float left, float right, float bottom, float top, float nearPlane,
                         float farPlane) {
  Mat4 result = Mat4::identity();
  result.m[0] = 2.0f / (right - left);
  result.m[5] = 2.0f / (top - bottom);
  result.m[10] = -2.0f / (farPlane - nearPlane);
  result.m[12] = -(right + left) / (right - left);
  result.m[13] = -(top + bottom) / (top - bottom);
  result.m[14] = -(farPlane + nearPlane) / (farPlane - nearPlane);
  return result;
}

inline Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
  const Vec3 forward = normalize(center - eye);
  Vec3 side = normalize(cross(forward, up));
  if (lengthSquared(side) <= 1.0e-10f) {
    const Vec3 fallbackUp = std::abs(forward.y) > 0.9f ? Vec3{0, 0, 1} : Vec3{0, 1, 0};
    side = normalize(cross(forward, fallbackUp));
  }
  const Vec3 correctedUp = cross(side, forward);
  Mat4 result = Mat4::identity();
  result.m[0] = side.x;
  result.m[4] = side.y;
  result.m[8] = side.z;
  result.m[1] = correctedUp.x;
  result.m[5] = correctedUp.y;
  result.m[9] = correctedUp.z;
  result.m[2] = -forward.x;
  result.m[6] = -forward.y;
  result.m[10] = -forward.z;
  result.m[12] = -dot(side, eye);
  result.m[13] = -dot(correctedUp, eye);
  result.m[14] = dot(forward, eye);
  return result;
}

inline Mat4 translation(const Vec3& value) {
  Mat4 result = Mat4::identity();
  result.m[12] = value.x;
  result.m[13] = value.y;
  result.m[14] = value.z;
  return result;
}

inline Mat4 scaling(const Vec3& value) {
  Mat4 result = Mat4::identity();
  result.m[0] = value.x;
  result.m[5] = value.y;
  result.m[10] = value.z;
  return result;
}

inline float radians(float degrees) { return degrees * kPi / 180.0f; }

}  // namespace orv
