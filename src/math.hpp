#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace terrain {

constexpr float kPi = 3.14159265358979323846f;

inline float radians(float degrees) {
  return degrees * (kPi / 180.0f);
}

inline float clamp01(float value) {
  return std::max(0.0f, std::min(1.0f, value));
}

inline float clampf(float value, float low, float high) {
  return std::max(low, std::min(high, value));
}

inline float smoothstep(float edge0, float edge1, float value) {
  if (edge0 == edge1) {
    return value < edge0 ? 0.0f : 1.0f;
  }
  const float t = clamp01((value - edge0) / (edge1 - edge0));
  return t * t * (3.0f - 2.0f * t);
}

struct Vec2 {
  float x = 0.0f;
  float y = 0.0f;

  constexpr Vec2() = default;
  constexpr Vec2(float xValue, float yValue) : x(xValue), y(yValue) {}
};

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;

  constexpr Vec3() = default;
  constexpr Vec3(float xValue, float yValue, float zValue)
      : x(xValue), y(yValue), z(zValue) {}
  explicit constexpr Vec3(float value) : x(value), y(value), z(value) {}
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator-(const Vec3& value) {
  return {-value.x, -value.y, -value.z};
}

inline Vec3 operator*(const Vec3& a, float scalar) {
  return {a.x * scalar, a.y * scalar, a.z * scalar};
}

inline Vec3 operator*(float scalar, const Vec3& a) {
  return a * scalar;
}

inline Vec3 operator*(const Vec3& a, const Vec3& b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}

inline Vec3& operator+=(Vec3& a, const Vec3& b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

inline Vec3& operator*=(Vec3& a, float scalar) {
  a.x *= scalar;
  a.y *= scalar;
  a.z *= scalar;
  return a;
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

inline float length(const Vec3& value) {
  return std::sqrt(lengthSquared(value));
}

inline Vec3 normalize(const Vec3& value) {
  const float magnitude = length(value);
  if (magnitude <= 1.0e-20f) {
    return {0.0f, 1.0f, 0.0f};
  }
  return value * (1.0f / magnitude);
}

inline Vec3 lerp(const Vec3& a, const Vec3& b, float t) {
  return a + (b - a) * t;
}

inline Vec3 vmin(const Vec3& a, const Vec3& b) {
  return {
      std::min(a.x, b.x),
      std::min(a.y, b.y),
      std::min(a.z, b.z),
  };
}

inline Vec3 vmax(const Vec3& a, const Vec3& b) {
  return {
      std::max(a.x, b.x),
      std::max(a.y, b.y),
      std::max(a.z, b.z),
  };
}

inline bool isFinite(float value) {
  return std::isfinite(value);
}

inline bool isFinite(const Vec3& value) {
  return isFinite(value.x) && isFinite(value.y) && isFinite(value.z);
}

struct Vec4 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 0.0f;

  constexpr Vec4() = default;
  constexpr Vec4(float xValue, float yValue, float zValue, float wValue)
      : x(xValue), y(yValue), z(zValue), w(wValue) {}
  constexpr Vec4(const Vec3& value, float wValue)
      : x(value.x), y(value.y), z(value.z), w(wValue) {}
};

inline Vec4 operator+(const Vec4& a, const Vec4& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

inline Vec4 operator-(const Vec4& a, const Vec4& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

inline Vec4 operator*(const Vec4& value, float scalar) {
  return {value.x * scalar, value.y * scalar, value.z * scalar, value.w * scalar};
}

inline Vec4 operator*(float scalar, const Vec4& value) {
  return value * scalar;
}

inline bool isFinite(const Vec4& value) {
  return isFinite(value.x) && isFinite(value.y) && isFinite(value.z) && isFinite(value.w);
}

struct Mat4 {
  float m[4][4] = {};

  static Mat4 identity() {
    Mat4 result;
    result.m[0][0] = 1.0f;
    result.m[1][1] = 1.0f;
    result.m[2][2] = 1.0f;
    result.m[3][3] = 1.0f;
    return result;
  }

  Vec4 operator*(const Vec4& value) const {
    return {
        m[0][0] * value.x + m[0][1] * value.y + m[0][2] * value.z + m[0][3] * value.w,
        m[1][0] * value.x + m[1][1] * value.y + m[1][2] * value.z + m[1][3] * value.w,
        m[2][0] * value.x + m[2][1] * value.y + m[2][2] * value.z + m[2][3] * value.w,
        m[3][0] * value.x + m[3][1] * value.y + m[3][2] * value.z + m[3][3] * value.w,
    };
  }

  Mat4 operator*(const Mat4& rhs) const {
    Mat4 result;
    for (int row = 0; row < 4; ++row) {
      for (int column = 0; column < 4; ++column) {
        float value = 0.0f;
        for (int k = 0; k < 4; ++k) {
          value += m[row][k] * rhs.m[k][column];
        }
        result.m[row][column] = value;
      }
    }
    return result;
  }

  Vec3 transformPoint(const Vec3& point) const {
    const Vec4 transformed = (*this) * Vec4(point, 1.0f);
    return {transformed.x, transformed.y, transformed.z};
  }

  Vec3 transformVector(const Vec3& vector) const {
    const Vec4 transformed = (*this) * Vec4(vector, 0.0f);
    return {transformed.x, transformed.y, transformed.z};
  }
};

inline Mat4 lookAtRH(const Vec3& eye, const Vec3& center, const Vec3& up) {
  const Vec3 forward = normalize(center - eye);
  Vec3 side = cross(forward, up);
  if (lengthSquared(side) <= 1.0e-16f) {
    side = cross(forward, Vec3(0.0f, 0.0f, 1.0f));
  }
  side = normalize(side);
  const Vec3 cameraUp = normalize(cross(side, forward));

  Mat4 result = Mat4::identity();
  result.m[0][0] = side.x;
  result.m[0][1] = side.y;
  result.m[0][2] = side.z;
  result.m[0][3] = -dot(side, eye);

  result.m[1][0] = cameraUp.x;
  result.m[1][1] = cameraUp.y;
  result.m[1][2] = cameraUp.z;
  result.m[1][3] = -dot(cameraUp, eye);

  result.m[2][0] = -forward.x;
  result.m[2][1] = -forward.y;
  result.m[2][2] = -forward.z;
  result.m[2][3] = dot(forward, eye);

  return result;
}

inline Mat4 perspectiveRH(float fovYRadians, float aspect, float nearPlane, float farPlane) {
  Mat4 result;
  const float focal = 1.0f / std::tan(fovYRadians * 0.5f);
  result.m[0][0] = focal / aspect;
  result.m[1][1] = focal;
  result.m[2][2] = (farPlane + nearPlane) / (nearPlane - farPlane);
  result.m[2][3] = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
  result.m[3][2] = -1.0f;
  return result;
}

inline Mat4 orthoRH(float left, float right, float bottom, float top,
                    float nearPlane, float farPlane) {
  Mat4 result = Mat4::identity();
  result.m[0][0] = 2.0f / (right - left);
  result.m[0][3] = -(right + left) / (right - left);
  result.m[1][1] = 2.0f / (top - bottom);
  result.m[1][3] = -(top + bottom) / (top - bottom);
  result.m[2][2] = -2.0f / (farPlane - nearPlane);
  result.m[2][3] = -(farPlane + nearPlane) / (farPlane - nearPlane);
  return result;
}

inline bool isFinite(const Mat4& matrix) {
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      if (!std::isfinite(matrix.m[row][column])) {
        return false;
      }
    }
  }
  return true;
}

inline float determinant(const Mat4& matrix) {
  const float* a = &matrix.m[0][0];
  const float a00 = a[0];
  const float a01 = a[1];
  const float a02 = a[2];
  const float a03 = a[3];
  const float a10 = a[4];
  const float a11 = a[5];
  const float a12 = a[6];
  const float a13 = a[7];
  const float a20 = a[8];
  const float a21 = a[9];
  const float a22 = a[10];
  const float a23 = a[11];
  const float a30 = a[12];
  const float a31 = a[13];
  const float a32 = a[14];
  const float a33 = a[15];

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

  return b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
}

inline uint64_t hashMix(uint64_t value) {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31u;
  return value;
}

inline uint64_t hashCombine(uint64_t seed, uint64_t value) {
  return hashMix(seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6u) + (seed >> 2u)));
}

inline float hashToUnitFloat(uint64_t value) {
  return static_cast<float>(value >> 40u) * (1.0f / 16777216.0f);
}

}  // namespace terrain

