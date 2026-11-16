#pragma once

#include <cstdint>
#include <vector>

#include "core/Mesh.hpp"

namespace lcr {

struct CapeSortReport {
  std::uint32_t triangleCount = 0;
  bool orderValid = true;
  std::uint32_t orderViolations = 0;
  std::uint32_t trianglesInFrontOfOpaque = 0;
  std::uint32_t trianglesBehindOpaque = 0;
  std::uint32_t torsoOccludedTriangles = 0;
  std::uint32_t weaponOccludedTriangles = 0;
  float minEdgeFeather = 0.0F;
  float maxEdgeFeather = 1.0F;
  std::uint64_t orderHash = 0;
};

std::vector<std::uint32_t> sortCapeTrianglesBackToFront(
    const Mesh& cape,
    const std::vector<Vec3>& skinnedPositions,
    const Vec3 cameraPosition);

float capeEdgeFeather(float u, float v);

CapeSortReport analyzeCapeSort(
    const CharacterAssets& assets,
    const Pose& pose,
    const Vec3 cameraPosition);

std::uint64_t hashTriangleOrder(const std::vector<std::uint32_t>& order);

}  // namespace lcr

