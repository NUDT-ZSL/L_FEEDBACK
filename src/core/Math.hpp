#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace lcr {

constexpr float kPi = 3.14159265358979323846F;

struct Vec2 {
  float x = 0.0F;
  float y = 0.0F;
};

struct Vec3 {
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
};

inline Vec3 operator+(const Vec3 a, const Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(const Vec3 a, const Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(const Vec3 a, const float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(const float s, const Vec3 a) { return a * s; }
inline Vec3 operator/(const Vec3 a, const float s) { return {a.x / s, a.y / s, a.z / s}; }
inline Vec3 operator-(const Vec3 a) { return {-a.x, -a.y, -a.z}; }
inline Vec3& operator+=(Vec3& a, const Vec3 b) {
  a = a + b;
  return a;
}

inline float dot(const Vec3 a, const Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3 a, const Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float lengthSquared(const Vec3 v) { return dot(v, v); }
inline float length(const Vec3 v) { return std::sqrt(lengthSquared(v)); }
inline Vec3 normalize(const Vec3 v) {
  const float len = length(v);
  return len > 1.0e-8F ? v / len : Vec3{0.0F, 1.0F, 0.0F};
}
inline Vec3 lerp(const Vec3 a, const Vec3 b, const float t) { return a + (b - a) * t; }
inline bool isFinite(const float value) { return std::isfinite(value); }
inline bool isFinite(const Vec3 v) { return isFinite(v.x) && isFinite(v.y) && isFinite(v.z); }

struct Quat {
  float w = 1.0F;
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
};

inline Quat operator*(const Quat a, const Quat b) {
  return {
      a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
      a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
  };
}

inline Quat normalize(const Quat q) {
  const float n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (n <= 1.0e-8F) {
    return {};
  }
  return {q.w / n, q.x / n, q.y / n, q.z / n};
}

inline Quat fromAxisAngle(const Vec3 axis, const float radians) {
  const Vec3 a = normalize(axis);
  const float half = radians * 0.5F;
  const float s = std::sin(half);
  return {std::cos(half), a.x * s, a.y * s, a.z * s};
}

inline Quat fromEulerXyz(const float rx, const float ry, const float rz) {
  const Quat qx = fromAxisAngle({1.0F, 0.0F, 0.0F}, rx);
  const Quat qy = fromAxisAngle({0.0F, 1.0F, 0.0F}, ry);
  const Quat qz = fromAxisAngle({0.0F, 0.0F, 1.0F}, rz);
  return normalize(qz * qy * qx);
}

inline Quat slerp(const Quat a_in, const Quat b_in, const float t) {
  Quat a = normalize(a_in);
  Quat b = normalize(b_in);
  float cos_theta = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
  if (cos_theta < 0.0F) {
    b = {-b.w, -b.x, -b.y, -b.z};
    cos_theta = -cos_theta;
  }
  if (cos_theta > 0.9995F) {
    return normalize(
        {a.w + (b.w - a.w) * t,
         a.x + (b.x - a.x) * t,
         a.y + (b.y - a.y) * t,
         a.z + (b.z - a.z) * t});
  }
  const float theta = std::acos(std::fmax(-1.0F, std::fmin(1.0F, cos_theta)));
  const float sin_theta = std::sin(theta);
  const float wa = std::sin((1.0F - t) * theta) / sin_theta;
  const float wb = std::sin(t * theta) / sin_theta;
  return normalize(
      {a.w * wa + b.w * wb,
       a.x * wa + b.x * wb,
       a.y * wa + b.y * wb,
       a.z * wa + b.z * wb});
}

struct Transform {
  Vec3 translation;
  Quat rotation;
  Vec3 scale{1.0F, 1.0F, 1.0F};
};

struct Mat4 {
  // Column-major, matching OpenGL.
  std::array<float, 16> m{};

  static Mat4 identity() {
    Mat4 out;
    out.m = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    return out;
  }

  float& at(const int row, const int col) { return m[static_cast<std::size_t>(col * 4 + row)]; }
  float at(const int row, const int col) const {
    return m[static_cast<std::size_t>(col * 4 + row)];
  }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
  Mat4 out;
  for (int col = 0; col < 4; ++col) {
    for (int row = 0; row < 4; ++row) {
      float sum = 0.0F;
      for (int k = 0; k < 4; ++k) {
        sum += a.at(row, k) * b.at(k, col);
      }
      out.at(row, col) = sum;
    }
  }
  return out;
}

inline Vec3 transformPoint(const Mat4& m, const Vec3 p) {
  const float x = m.at(0, 0) * p.x + m.at(0, 1) * p.y + m.at(0, 2) * p.z + m.at(0, 3);
  const float y = m.at(1, 0) * p.x + m.at(1, 1) * p.y + m.at(1, 2) * p.z + m.at(1, 3);
  const float z = m.at(2, 0) * p.x + m.at(2, 1) * p.y + m.at(2, 2) * p.z + m.at(2, 3);
  const float w = m.at(3, 0) * p.x + m.at(3, 1) * p.y + m.at(3, 2) * p.z + m.at(3, 3);
  return w != 0.0F ? Vec3{x / w, y / w, z / w} : Vec3{x, y, z};
}

inline Vec3 transformVector(const Mat4& m, const Vec3 v) {
  return {
      m.at(0, 0) * v.x + m.at(0, 1) * v.y + m.at(0, 2) * v.z,
      m.at(1, 0) * v.x + m.at(1, 1) * v.y + m.at(1, 2) * v.z,
      m.at(2, 0) * v.x + m.at(2, 1) * v.y + m.at(2, 2) * v.z,
  };
}

inline Mat4 compose(const Transform& t) {
  Mat4 out = Mat4::identity();
  const float xx = t.rotation.x * t.rotation.x;
  const float yy = t.rotation.y * t.rotation.y;
  const float zz = t.rotation.z * t.rotation.z;
  const float xy = t.rotation.x * t.rotation.y;
  const float xz = t.rotation.x * t.rotation.z;
  const float yz = t.rotation.y * t.rotation.z;
  const float wx = t.rotation.w * t.rotation.x;
  const float wy = t.rotation.w * t.rotation.y;
  const float wz = t.rotation.w * t.rotation.z;

  out.at(0, 0) = (1.0F - 2.0F * (yy + zz)) * t.scale.x;
  out.at(0, 1) = (2.0F * (xy - wz)) * t.scale.y;
  out.at(0, 2) = (2.0F * (xz + wy)) * t.scale.z;
  out.at(0, 3) = t.translation.x;
  out.at(1, 0) = (2.0F * (xy + wz)) * t.scale.x;
  out.at(1, 1) = (1.0F - 2.0F * (xx + zz)) * t.scale.y;
  out.at(1, 2) = (2.0F * (yz - wx)) * t.scale.z;
  out.at(1, 3) = t.translation.y;
  out.at(2, 0) = (2.0F * (xz - wy)) * t.scale.x;
  out.at(2, 1) = (2.0F * (yz + wx)) * t.scale.y;
  out.at(2, 2) = (1.0F - 2.0F * (xx + yy)) * t.scale.z;
  out.at(2, 3) = t.translation.z;
  return out;
}

inline Mat4 inverseRigid(const Mat4& m) {
  Mat4 out = Mat4::identity();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      out.at(row, col) = m.at(col, row);
    }
  }
  const Vec3 translation{m.at(0, 3), m.at(1, 3), m.at(2, 3)};
  const Vec3 invTranslation = -(transformVector(out, translation));
  out.at(0, 3) = invTranslation.x;
  out.at(1, 3) = invTranslation.y;
  out.at(2, 3) = invTranslation.z;
  return out;
}

inline bool isFinite(const Quat q) {
  return isFinite(q.w) && isFinite(q.x) && isFinite(q.y) && isFinite(q.z);
}

}  // namespace lcr

