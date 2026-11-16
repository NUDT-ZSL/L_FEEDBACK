#include "core/CapeSorter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/Animation.hpp"
#include "core/Serializer.hpp"

namespace lcr {
namespace {

float triangleDistance(
    const Mesh& mesh,
    const std::vector<Vec3>& positions,
    const std::uint32_t triangle,
    const Vec3 camera) {
  const std::uint32_t i0 = mesh.indices[triangle * 3U];
  const std::uint32_t i1 = mesh.indices[triangle * 3U + 1U];
  const std::uint32_t i2 = mesh.indices[triangle * 3U + 2U];
  const Vec3 center = (positions[i0] + positions[i1] + positions[i2]) / 3.0F;
  return length(center - camera);
}

bool rayTriangle(
    const Vec3 origin,
    const Vec3 direction,
    const Vec3 a,
    const Vec3 b,
    const Vec3 c,
    float& distance) {
  constexpr float kEpsilon = 1.0e-7F;
  const Vec3 edge1 = b - a;
  const Vec3 edge2 = c - a;
  const Vec3 pvec = cross(direction, edge2);
  const float determinant = dot(edge1, pvec);
  if (std::fabs(determinant) < kEpsilon) {
    return false;
  }
  const float inverse = 1.0F / determinant;
  const Vec3 tvec = origin - a;
  const float u = dot(tvec, pvec) * inverse;
  if (u < 0.0F || u > 1.0F) {
    return false;
  }
  const Vec3 qvec = cross(tvec, edge1);
  const float v = dot(direction, qvec) * inverse;
  if (v < 0.0F || u + v > 1.0F) {
    return false;
  }
  distance = dot(edge2, qvec) * inverse;
  return distance > 1.0e-4F;
}

bool capeTriangleIsOccluded(
    const Vec3 centroid,
    const Vec3 camera,
    const Mesh& opaque,
    const std::vector<Vec3>& opaquePositions) {
  const Vec3 ray = centroid - camera;
  const float capeDistance = length(ray);
  if (capeDistance <= 1.0e-5F) {
    return false;
  }
  const Vec3 direction = ray / capeDistance;
  for (std::uint32_t opaqueTriangle = 0;
       opaqueTriangle < opaque.indices.size() / 3U;
       ++opaqueTriangle) {
    const Vec3 a = opaquePositions[opaque.indices[opaqueTriangle * 3U]];
    const Vec3 b = opaquePositions[opaque.indices[opaqueTriangle * 3U + 1U]];
    const Vec3 c = opaquePositions[opaque.indices[opaqueTriangle * 3U + 2U]];
    float distance = 0.0F;
    if (rayTriangle(camera, direction, a, b, c, distance) &&
        distance < capeDistance - 1.0e-4F) {
      return true;
    }
  }
  return false;
}

struct Counts {
  std::uint32_t torso = 0;
  std::uint32_t weapon = 0;
  std::uint32_t either = 0;
};

Counts countCapeOcclusion(
    const Mesh& cape,
    const std::vector<Vec3>& capePositions,
    const Mesh& torso,
    const std::vector<Vec3>& torsoPositions,
    const Mesh& weapon,
    const std::vector<Vec3>& weaponPositions,
    const Vec3 camera) {
  Counts counts;
  for (std::uint32_t triangle = 0; triangle < cape.indices.size() / 3U; ++triangle) {
    const Vec3 c0 = capePositions[cape.indices[triangle * 3U]];
    const Vec3 c1 = capePositions[cape.indices[triangle * 3U + 1U]];
    const Vec3 c2 = capePositions[cape.indices[triangle * 3U + 2U]];
    const Vec3 centroid = (c0 + c1 + c2) / 3.0F;
    const bool byTorso = capeTriangleIsOccluded(
        centroid, camera, torso, torsoPositions);
    const bool byWeapon = capeTriangleIsOccluded(
        centroid, camera, weapon, weaponPositions);
    if (byTorso) {
      ++counts.torso;
    }
    if (byWeapon) {
      ++counts.weapon;
    }
    if (byTorso || byWeapon) {
      ++counts.either;
    }
  }
  return counts;
}

}  // namespace

std::vector<std::uint32_t> sortCapeTrianglesBackToFront(
    const Mesh& cape,
    const std::vector<Vec3>& skinnedPositions,
    const Vec3 cameraPosition) {
  const std::uint32_t triangleCount = static_cast<std::uint32_t>(cape.indices.size() / 3U);
  std::vector<std::uint32_t> order(triangleCount);
  for (std::uint32_t i = 0; i < triangleCount; ++i) {
    order[i] = i;
  }
  std::stable_sort(
      order.begin(),
      order.end(),
      [&](const std::uint32_t a, const std::uint32_t b) {
        const float da = triangleDistance(cape, skinnedPositions, a, cameraPosition);
        const float db = triangleDistance(cape, skinnedPositions, b, cameraPosition);
        if (da == db) {
          return a < b;
        }
        return da > db;
      });
  return order;
}

float capeEdgeFeather(const float u, const float v) {
  const float border = std::fmin(
      std::fmin(u, 1.0F - u),
      std::fmin(v, 1.0F - v));
  const float t = std::fmax(0.0F, std::fmin(1.0F, (border - 0.018F) / 0.105F));
  return t * t * (3.0F - 2.0F * t);
}

std::uint64_t hashTriangleOrder(const std::vector<std::uint32_t>& order) {
  CanonicalWriter writer;
  writer.string("lcr-triangle-order-v1");
  writer.u32(static_cast<std::uint32_t>(order.size()));
  for (const std::uint32_t value : order) {
    writer.u32(value);
  }
  return hashBytes(writer.bytes());
}

CapeSortReport analyzeCapeSort(
    const CharacterAssets& assets,
    const Pose& pose,
    const Vec3 cameraPosition) {
  const MeshPart* cape = assets.findPart("Cape");
  const MeshPart* torso = assets.findPart("Torso");
  const MeshPart* weapon = assets.findPart("Weapon");
  if (cape == nullptr || torso == nullptr || weapon == nullptr) {
    return {};
  }

  const std::vector<Mat4> skin = assets.skeleton.computeSkinMatrices(pose);
  const std::vector<Vec3> capePositions = transformedVertexPositions(cape->mesh, skin);
  const std::vector<Vec3> torsoPositions = transformedVertexPositions(torso->mesh, skin);
  const std::vector<Vec3> weaponPositions = transformedVertexPositions(weapon->mesh, skin);
  const std::vector<std::uint32_t> order =
      sortCapeTrianglesBackToFront(cape->mesh, capePositions, cameraPosition);

  CapeSortReport report;
  report.triangleCount = static_cast<std::uint32_t>(order.size());
  report.orderHash = hashTriangleOrder(order);
  float previousDistance = std::numeric_limits<float>::infinity();
  for (const std::uint32_t triangle : order) {
    const float distance =
        triangleDistance(cape->mesh, capePositions, triangle, cameraPosition);
    if (distance > previousDistance + 1.0e-5F) {
      report.orderValid = false;
      ++report.orderViolations;
    }
    previousDistance = distance;
  }

  const Counts counts = countCapeOcclusion(
      cape->mesh,
      capePositions,
      torso->mesh,
      torsoPositions,
      weapon->mesh,
      weaponPositions,
      cameraPosition);
  report.torsoOccludedTriangles = counts.torso;
  report.weaponOccludedTriangles = counts.weapon;
  report.trianglesBehindOpaque = counts.either;
  report.trianglesInFrontOfOpaque = report.triangleCount - counts.either;

  report.minEdgeFeather = 1.0F;
  report.maxEdgeFeather = 0.0F;
  for (const Vertex& vertex : cape->mesh.vertices) {
    const float feather = capeEdgeFeather(vertex.uv.x, vertex.uv.y);
    report.minEdgeFeather = std::fmin(report.minEdgeFeather, feather);
    report.maxEdgeFeather = std::fmax(report.maxEdgeFeather, feather);
  }
  return report;
}

}  // namespace lcr
