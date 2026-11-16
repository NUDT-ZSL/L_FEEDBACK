#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "core/Math.hpp"
#include "core/Skeleton.hpp"

namespace lcr {

enum class MaterialSlot : std::uint32_t {
  Body = 0,
  Weapon = 1,
  Cape = 2,
};

struct Vertex {
  Vec3 position;
  Vec3 normal;
  Vec2 uv;
  std::array<std::int32_t, kMaxBoneInfluences> joints{};
  std::array<float, kMaxBoneInfluences> weights{};
};

struct Mesh {
  std::vector<Vertex> vertices;
  std::vector<std::uint32_t> indices;
};

struct MeshPart {
  std::string name;
  MaterialSlot materialSlot = MaterialSlot::Body;
  bool transparent = false;
  Mesh mesh;
};

struct CharacterAssets {
  Skeleton skeleton;
  std::vector<MeshPart> parts;
  std::uint64_t skeletonHash = 0;
  std::uint64_t meshHash = 0;
  std::uint64_t combinedHash = 0;
  float animationPhaseA = 0.0F;
  float animationPhaseB = 0.0F;
  float animationPhaseC = 0.0F;

  const MeshPart* findPart(const std::string& name) const;
  void computeHashes();
  void validate() const;
};

CharacterAssets buildDeterministicCharacter(std::uint64_t seed);

std::vector<std::uint8_t> meshCanonicalBytes(const std::vector<MeshPart>& parts);
std::uint64_t meshDataHash(const std::vector<MeshPart>& parts);

Vec3 skinPosition(
    const Vertex& vertex,
    const std::vector<Mat4>& skinMatrices);

std::vector<Vec3> transformedVertexPositions(
    const Mesh& mesh,
    const std::vector<Mat4>& skinMatrices);

}  // namespace lcr

