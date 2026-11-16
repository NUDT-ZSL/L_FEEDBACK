#include "core/Mesh.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

#include "core/Error.hpp"
#include "core/Rng.hpp"
#include "core/Serializer.hpp"

namespace lcr {
namespace {

struct WeightSet {
  std::array<std::int32_t, kMaxBoneInfluences> joints{{0, 0, 0, 0}};
  std::array<float, kMaxBoneInfluences> weights{{1.0F, 0.0F, 0.0F, 0.0F}};
};

WeightSet oneBone(const int bone) {
  WeightSet result;
  result.joints = {{bone, bone, bone, bone}};
  result.weights = {{1.0F, 0.0F, 0.0F, 0.0F}};
  return result;
}

WeightSet twoBones(const int a, const int b, const float blend) {
  WeightSet result;
  const float t = std::fmax(0.0F, std::fmin(1.0F, blend));
  result.joints = {{a, b, b, b}};
  result.weights = {{1.0F - t, t, 0.0F, 0.0F}};
  return result;
}

WeightSet threeBones(
    const int a,
    const int b,
    const int c,
    const float wa,
    const float wb,
    const float wc) {
  const float sum = wa + wb + wc;
  if (sum <= 1.0e-8F) {
    return oneBone(a);
  }
  WeightSet result;
  result.joints = {{a, b, c, c}};
  result.weights = {{wa / sum, wb / sum, wc / sum, 0.0F}};
  return result;
}

WeightSet jitterWeights(WeightSet weights, DeterministicRng& rng, const float amount) {
  float sum = 0.0F;
  for (int i = 0; i < kMaxBoneInfluences; ++i) {
    if (weights.weights[i] > 0.0F) {
      weights.weights[i] = std::fmax(
          0.0F,
          weights.weights[i] + rng.symmetric() * amount);
      sum += weights.weights[i];
    }
  }
  if (sum <= 1.0e-8F) {
    weights.weights = {{1.0F, 0.0F, 0.0F, 0.0F}};
    sum = 1.0F;
  }
  for (float& weight : weights.weights) {
    weight /= sum;
  }
  return weights;
}

Vertex makeVertex(
    const Vec3 position,
    const Vec3 normal,
    const Vec2 uv,
    const WeightSet& weights) {
  Vertex vertex;
  vertex.position = position;
  vertex.normal = normal;
  vertex.uv = uv;
  vertex.joints = weights.joints;
  vertex.weights = weights.weights;
  return vertex;
}

void appendQuad(Mesh& mesh, const std::uint32_t a, const std::uint32_t b, const std::uint32_t c, const std::uint32_t d) {
  mesh.indices.push_back(a);
  mesh.indices.push_back(b);
  mesh.indices.push_back(c);
  mesh.indices.push_back(a);
  mesh.indices.push_back(c);
  mesh.indices.push_back(d);
}

void appendTube(
    Mesh& mesh,
    const Vec3 start,
    const Vec3 end,
    const float radiusStart,
    const float radiusEnd,
    const int sides,
    const int rings,
    DeterministicRng& rng,
    const std::function<WeightSet(float, float)>& weightAt) {
  const Vec3 direction = normalize(end - start);
  const Vec3 reference = std::fabs(direction.y) > 0.9F ? Vec3{1.0F, 0.0F, 0.0F} : Vec3{0.0F, 1.0F, 0.0F};
  const Vec3 basisX = normalize(cross(reference, direction));
  const Vec3 basisY = normalize(cross(direction, basisX));
  const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());

  for (int ring = 0; ring <= rings; ++ring) {
    const float v = static_cast<float>(ring) / static_cast<float>(rings);
    const Vec3 center = lerp(start, end, v);
    const float radius = radiusStart + (radiusEnd - radiusStart) * v;
    for (int side = 0; side < sides; ++side) {
      const float u = static_cast<float>(side) / static_cast<float>(sides);
      const float angle = u * kPi * 2.0F;
      const Vec3 radial = basisX * std::cos(angle) + basisY * std::sin(angle);
      const float localRadius = radius * rng.range(0.965F, 1.035F);
      const Vec3 position = center + radial * localRadius + direction * rng.range(-0.002F, 0.002F);
      mesh.vertices.push_back(
          makeVertex(position, normalize(radial), {u, v}, weightAt(v, u)));
    }
  }

  for (int ring = 0; ring < rings; ++ring) {
    for (int side = 0; side < sides; ++side) {
      const int next = (side + 1) % sides;
      const std::uint32_t a = base + static_cast<std::uint32_t>(ring * sides + side);
      const std::uint32_t b = base + static_cast<std::uint32_t>(ring * sides + next);
      const std::uint32_t c = base + static_cast<std::uint32_t>((ring + 1) * sides + next);
      const std::uint32_t d = base + static_cast<std::uint32_t>((ring + 1) * sides + side);
      appendQuad(mesh, a, b, c, d);
    }
  }

  const std::uint32_t capStart = static_cast<std::uint32_t>(mesh.vertices.size());
  const WeightSet startWeights = weightAt(0.0F, 0.0F);
  const WeightSet endWeights = weightAt(1.0F, 0.0F);
  mesh.vertices.push_back(makeVertex(start, -direction, {0.5F, 0.0F}, startWeights));
  mesh.vertices.push_back(makeVertex(end, direction, {0.5F, 1.0F}, endWeights));
  for (int side = 0; side < sides; ++side) {
    const int next = (side + 1) % sides;
    const std::uint32_t a = base + static_cast<std::uint32_t>(side);
    const std::uint32_t b = base + static_cast<std::uint32_t>(next);
    mesh.indices.push_back(capStart);
    mesh.indices.push_back(b);
    mesh.indices.push_back(a);
    const std::uint32_t c = base + static_cast<std::uint32_t>(rings * sides + next);
    const std::uint32_t d = base + static_cast<std::uint32_t>(rings * sides + side);
    mesh.indices.push_back(capStart + 1U);
    mesh.indices.push_back(d);
    mesh.indices.push_back(c);
  }
}

void appendBox(
    Mesh& mesh,
    const Vec3 center,
    const Vec3 halfExtents,
    const WeightSet& weights) {
  const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
  const Vec3 corners[8] = {
      {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
      {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1},
  };
  for (int i = 0; i < 8; ++i) {
    const Vec3 corner = corners[i];
    const Vec3 position{
        center.x + corner.x * halfExtents.x,
        center.y + corner.y * halfExtents.y,
        center.z + corner.z * halfExtents.z};
    const Vec3 normal = normalize(corner);
    const Vec2 uv{
        (corner.x + 1.0F) * 0.5F,
        (corner.y + 1.0F) * 0.5F};
    mesh.vertices.push_back(makeVertex(position, normal, uv, weights));
  }
  const std::uint32_t faces[6][4] = {
      {0, 3, 2, 1}, {4, 5, 6, 7}, {0, 4, 7, 3},
      {1, 2, 6, 5}, {3, 7, 6, 2}, {0, 1, 5, 4},
  };
  for (const auto& face : faces) {
    appendQuad(mesh, base + face[0], base + face[1], base + face[2], base + face[3]);
  }
}

MeshPart makeTorso(const Skeleton& skeleton, DeterministicRng& rng) {
  MeshPart part;
  part.name = "Torso";
  part.materialSlot = MaterialSlot::Body;
  const Vec3 pelvis = skeleton.bindWorldPositions[kBonePelvis];
  const Vec3 chest = skeleton.bindWorldPositions[kBoneChest];
  constexpr int kSides = 18;
  constexpr int kRings = 12;

  for (int ring = 0; ring <= kRings; ++ring) {
    const float v = static_cast<float>(ring) / static_cast<float>(kRings);
    const float lowerBlend = std::fmin(1.0F, v * 1.8F);
    const Vec3 center = lerp(
        pelvis + Vec3{0.0F, -0.05F, 0.0F},
        chest + Vec3{0.0F, 0.11F, 0.0F},
        v);
    const float shoulder = std::fmax(0.0F, (v - 0.62F) / 0.38F);
    const float rx = 0.24F + 0.075F * shoulder - 0.015F * std::sin(v * kPi);
    const float rz = 0.16F + 0.045F * shoulder;
    const WeightSet weights = threeBones(
        kBonePelvis,
        kBoneSpine,
        kBoneChest,
        (1.0F - lowerBlend) * 0.85F,
        0.55F,
        (1.0F - 0.55F) * lowerBlend + 0.15F);
    for (int side = 0; side < kSides; ++side) {
      const float u = static_cast<float>(side) / static_cast<float>(kSides);
      const float angle = u * kPi * 2.0F;
      const Vec3 radial{std::cos(angle), 0.0F, std::sin(angle)};
      const Vec3 position{
          center.x + radial.x * rx * rng.range(0.985F, 1.015F),
          center.y + rng.range(-0.003F, 0.003F),
          center.z + radial.z * rz * rng.range(0.985F, 1.015F)};
      const Vec3 normal = normalize(Vec3{radial.x / rx, 0.05F, radial.z / rz});
      part.mesh.vertices.push_back(
          makeVertex(position, normal, {u, v}, jitterWeights(weights, rng, 0.018F)));
    }
  }
  for (int ring = 0; ring < kRings; ++ring) {
    for (int side = 0; side < kSides; ++side) {
      const int next = (side + 1) % kSides;
      const std::uint32_t a = static_cast<std::uint32_t>(ring * kSides + side);
      const std::uint32_t b = static_cast<std::uint32_t>(ring * kSides + next);
      const std::uint32_t c = static_cast<std::uint32_t>((ring + 1) * kSides + next);
      const std::uint32_t d = static_cast<std::uint32_t>((ring + 1) * kSides + side);
      appendQuad(part.mesh, a, b, c, d);
    }
  }
  return part;
}

MeshPart makeHead(const Skeleton& skeleton, DeterministicRng& rng) {
  MeshPart part;
  part.name = "Head";
  part.materialSlot = MaterialSlot::Body;
  const Vec3 center = skeleton.bindWorldPositions[kBoneHead] + Vec3{0.0F, 0.12F, 0.0F};
  const Vec3 neck = skeleton.bindWorldPositions[kBoneNeck];
  appendTube(
      part.mesh,
      neck,
      center - Vec3{0.0F, 0.08F, 0.0F},
      0.072F,
      0.085F,
      12,
      3,
      rng,
      [&](const float, const float) {
        return twoBones(kBoneNeck, kBoneHead, 0.58F);
      });
  constexpr int kSegments = 20;
  constexpr int kStacks = 12;
  const float radius = 0.16F;
  for (int stack = 0; stack <= kStacks; ++stack) {
    const float v = static_cast<float>(stack) / static_cast<float>(kStacks);
    const float phi = v * kPi;
    for (int segment = 0; segment < kSegments; ++segment) {
      const float u = static_cast<float>(segment) / static_cast<float>(kSegments);
      const float theta = u * kPi * 2.0F;
      const Vec3 normal{
          std::sin(phi) * std::cos(theta),
          std::cos(phi),
          std::sin(phi) * std::sin(theta)};
      const Vec3 position = center + normal * radius * rng.range(0.98F, 1.02F);
      part.mesh.vertices.push_back(
          makeVertex(position, normal, {u, v}, oneBone(kBoneHead)));
    }
  }
  for (int stack = 0; stack < kStacks; ++stack) {
    for (int segment = 0; segment < kSegments; ++segment) {
      const int next = (segment + 1) % kSegments;
      const std::uint32_t a = static_cast<std::uint32_t>(stack * kSegments + segment);
      const std::uint32_t b = static_cast<std::uint32_t>(stack * kSegments + next);
      const std::uint32_t c = static_cast<std::uint32_t>((stack + 1) * kSegments + next);
      const std::uint32_t d = static_cast<std::uint32_t>((stack + 1) * kSegments + segment);
      appendQuad(part.mesh, a, b, c, d);
    }
  }
  return part;
}

MeshPart makeLimb(
    const Skeleton& skeleton,
    DeterministicRng& rng,
    const std::string& name,
    const int jointA,
    const int jointB,
    const int jointC,
    const float radiusA,
    const float radiusB,
    const float radiusC) {
  MeshPart part;
  part.name = name;
  part.materialSlot = MaterialSlot::Body;
  const Vec3 a = skeleton.bindWorldPositions[static_cast<std::size_t>(jointA)];
  const Vec3 b = skeleton.bindWorldPositions[static_cast<std::size_t>(jointB)];
  const Vec3 c = skeleton.bindWorldPositions[static_cast<std::size_t>(jointC)];
  appendTube(
      part.mesh, a, b, radiusA, radiusB, 12, 7, rng,
      [&](const float t, const float) {
        const float blend = std::fmax(0.0F, (t - 0.55F) / 0.45F);
        return jitterWeights(twoBones(jointA, jointB, blend), rng, 0.025F);
      });
  appendTube(
      part.mesh, b, c, radiusB, radiusC, 12, 6, rng,
      [&](const float t, const float) {
        const float blend = std::fmax(0.0F, (t - 0.55F) / 0.45F);
        return jitterWeights(twoBones(jointB, jointC, blend), rng, 0.025F);
      });
  return part;
}

MeshPart makeWeapon(const Skeleton& skeleton) {
  MeshPart part;
  part.name = "Weapon";
  part.materialSlot = MaterialSlot::Weapon;
  const WeightSet weights = oneBone(kBoneWeaponMount);
  const Vec3 mount = skeleton.bindWorldPositions[kBoneWeaponMount];
  appendBox(part.mesh, mount + Vec3{0.0F, 0.22F, 0.0F}, {0.055F, 0.22F, 0.018F}, weights);
  appendBox(part.mesh, mount + Vec3{0.0F, 0.45F, 0.0F}, {0.035F, 0.24F, 0.012F}, weights);
  appendBox(part.mesh, mount + Vec3{0.0F, 0.01F, 0.0F}, {0.16F, 0.025F, 0.04F}, weights);
  appendBox(part.mesh, mount + Vec3{0.0F, -0.09F, 0.0F}, {0.035F, 0.11F, 0.035F}, weights);
  return part;
}

MeshPart makeCape(const Skeleton& skeleton, DeterministicRng& rng) {
  MeshPart part;
  part.name = "Cape";
  part.materialSlot = MaterialSlot::Cape;
  part.transparent = true;
  const Vec3 root = skeleton.bindWorldPositions[kBoneCapeRoot];
  const Vec3 mid = skeleton.bindWorldPositions[kBoneCapeMid];
  const Vec3 tip = skeleton.bindWorldPositions[kBoneCapeTip];
  constexpr int kCols = 24;
  constexpr int kRows = 16;
  for (int row = 0; row <= kRows; ++row) {
    const float v = static_cast<float>(row) / static_cast<float>(kRows);
    for (int col = 0; col <= kCols; ++col) {
      const float u = static_cast<float>(col) / static_cast<float>(kCols);
      const float x = (u - 0.5F) * 1.04F;
      const float y = root.y - v * (root.y - (tip.y - 0.35F));
      const float z = -0.19F + 0.055F * std::sin((u - 0.5F) * kPi) * v -
                      0.035F * v * v;
      // The bottom corners lift slightly so the silhouette is not a rectangle.
      const float cornerLift =
          0.035F * v * v * std::pow(std::fabs(u - 0.5F) * 2.0F, 2.0F);
      const Vec3 position{x, y + cornerLift, z + rng.range(-0.003F, 0.003F)};

      float dRoot = length(position - root);
      float dMid = length(position - mid);
      float dTip = length(position - tip);
      WeightSet weights = threeBones(
          kBoneCapeRoot,
          kBoneCapeMid,
          kBoneCapeTip,
          1.0F / (dRoot + 0.15F),
          1.0F / (dMid + 0.15F),
          1.0F / (dTip + 0.15F));
      weights = jitterWeights(weights, rng, 0.02F);

      const Vec3 tangentU{1.0F, 0.0F, 0.0F};
      const Vec3 tangentV{0.0F, -1.0F, 0.12F};
      const Vec3 normal = normalize(cross(tangentV, tangentU));
      part.mesh.vertices.push_back(makeVertex(position, normal, {u, v}, weights));
    }
  }
  const int stride = kCols + 1;
  for (int row = 0; row < kRows; ++row) {
    for (int col = 0; col < kCols; ++col) {
      const std::uint32_t a = static_cast<std::uint32_t>(row * stride + col);
      const std::uint32_t b = a + 1U;
      const std::uint32_t c = static_cast<std::uint32_t>((row + 1) * stride + col + 1);
      const std::uint32_t d = static_cast<std::uint32_t>((row + 1) * stride + col);
      appendQuad(part.mesh, a, b, c, d);
    }
  }
  return part;
}

void appendVertex(CanonicalWriter& writer, const Vertex& vertex) {
  writer.f32(vertex.position.x);
  writer.f32(vertex.position.y);
  writer.f32(vertex.position.z);
  writer.f32(vertex.normal.x);
  writer.f32(vertex.normal.y);
  writer.f32(vertex.normal.z);
  writer.f32(vertex.uv.x);
  writer.f32(vertex.uv.y);
  for (const std::int32_t joint : vertex.joints) {
    writer.i32(joint);
  }
  for (const float weight : vertex.weights) {
    writer.f32(weight);
  }
}

}  // namespace

const MeshPart* CharacterAssets::findPart(const std::string& name) const {
  for (const MeshPart& part : parts) {
    if (part.name == name) {
      return &part;
    }
  }
  return nullptr;
}

std::vector<std::uint8_t> meshCanonicalBytes(const std::vector<MeshPart>& parts) {
  CanonicalWriter writer;
  writer.string("lcr-mesh-v1");
  writer.u32(static_cast<std::uint32_t>(parts.size()));
  for (const MeshPart& part : parts) {
    writer.string(part.name);
    writer.u32(static_cast<std::uint32_t>(part.materialSlot));
    writer.u8(part.transparent ? 1U : 0U);
    writer.u32(static_cast<std::uint32_t>(part.mesh.vertices.size()));
    for (const Vertex& vertex : part.mesh.vertices) {
      appendVertex(writer, vertex);
    }
    writer.u32(static_cast<std::uint32_t>(part.mesh.indices.size()));
    for (const std::uint32_t index : part.mesh.indices) {
      writer.u32(index);
    }
  }
  return writer.bytes();
}

std::uint64_t meshDataHash(const std::vector<MeshPart>& parts) {
  return hashBytes(meshCanonicalBytes(parts));
}

CharacterAssets buildDeterministicCharacter(const std::uint64_t seed) {
  SkeletonBuildResult built = buildDeterministicSkeleton(seed);
  DeterministicRng rng(seed ^ 0xD1B54A32D192ED03ULL);

  CharacterAssets assets;
  assets.skeleton = std::move(built.skeleton);
  assets.animationPhaseA = built.phaseA;
  assets.animationPhaseB = built.phaseB;
  assets.animationPhaseC = built.phaseC;
  assets.parts.reserve(8);
  assets.parts.push_back(makeTorso(assets.skeleton, rng));
  assets.parts.push_back(makeHead(assets.skeleton, rng));
  assets.parts.push_back(makeLimb(
      assets.skeleton, rng, "LeftArm",
      kBoneLeftUpperArm, kBoneLeftLowerArm, kBoneLeftHand,
      0.092F, 0.074F, 0.048F));
  assets.parts.push_back(makeLimb(
      assets.skeleton, rng, "RightArm",
      kBoneRightUpperArm, kBoneRightLowerArm, kBoneRightHand,
      0.092F, 0.074F, 0.048F));
  assets.parts.push_back(makeLimb(
      assets.skeleton, rng, "LeftLeg",
      kBoneLeftThigh, kBoneLeftShin, kBoneLeftFoot,
      0.135F, 0.095F, 0.060F));
  assets.parts.push_back(makeLimb(
      assets.skeleton, rng, "RightLeg",
      kBoneRightThigh, kBoneRightShin, kBoneRightFoot,
      0.135F, 0.095F, 0.060F));
  assets.parts.push_back(makeWeapon(assets.skeleton));
  assets.parts.push_back(makeCape(assets.skeleton, rng));
  assets.computeHashes();
  assets.validate();
  return assets;
}

void CharacterAssets::computeHashes() {
  skeletonHash = skeleton.computeHash();
  meshHash = meshDataHash(parts);
  CanonicalWriter writer;
  writer.string("lcr-character-v1");
  writer.u64(skeletonHash);
  writer.u64(meshHash);
  combinedHash = hashBytes(writer.bytes());
}

void CharacterAssets::validate() const {
  skeleton.validate();
  if (parts.empty()) {
    fail(ErrorCode::InternalInvariant, "character.validate", "no mesh parts");
  }
  for (const MeshPart& part : parts) {
    if (part.name.empty() || part.mesh.vertices.empty() || part.mesh.indices.empty()) {
      fail(ErrorCode::InternalInvariant, "character.validate", "empty mesh part");
    }
    if (part.mesh.indices.size() % 3U != 0U) {
      fail(ErrorCode::InternalInvariant, "character.validate", "index count is not triangular");
    }
    for (const std::uint32_t index : part.mesh.indices) {
      if (index >= part.mesh.vertices.size()) {
        fail(
            ErrorCode::BoneIndex,
            "character.validate",
            "mesh index points outside vertex buffer in part " + part.name);
      }
    }
    for (const Vertex& vertex : part.mesh.vertices) {
      if (!isFinite(vertex.position) || !isFinite(vertex.normal)) {
        fail(ErrorCode::InternalInvariant, "character.validate", "non-finite vertex in " + part.name);
      }
      float weightSum = 0.0F;
      for (int i = 0; i < kMaxBoneInfluences; ++i) {
        const int bone = vertex.joints[static_cast<std::size_t>(i)];
        if (bone < 0 || bone >= kBoneCount) {
          fail(
              ErrorCode::BoneIndex,
              "character.validate",
              "bone index out of range in part " + part.name);
        }
        if (!isFinite(vertex.weights[static_cast<std::size_t>(i)]) ||
            vertex.weights[static_cast<std::size_t>(i)] < 0.0F) {
          fail(
              ErrorCode::InternalInvariant,
              "character.validate",
              "invalid skin weight in part " + part.name);
        }
        weightSum += vertex.weights[static_cast<std::size_t>(i)];
      }
      if (std::fabs(weightSum - 1.0F) > 1.0e-3F) {
        fail(
            ErrorCode::InternalInvariant,
            "character.validate",
            "skin weights are not normalized in part " + part.name);
      }
    }
  }
}

Vec3 skinPosition(const Vertex& vertex, const std::vector<Mat4>& skinMatrices) {
  Vec3 result{0.0F, 0.0F, 0.0F};
  for (int i = 0; i < kMaxBoneInfluences; ++i) {
    const float weight = vertex.weights[static_cast<std::size_t>(i)];
    if (weight <= 0.0F) {
      continue;
    }
    const int bone = vertex.joints[static_cast<std::size_t>(i)];
    if (bone < 0 || bone >= static_cast<int>(skinMatrices.size())) {
      fail(ErrorCode::BoneIndex, "mesh.skinPosition", "skin matrix index out of range");
    }
    result += transformPoint(skinMatrices[static_cast<std::size_t>(bone)], vertex.position) * weight;
  }
  return result;
}

std::vector<Vec3> transformedVertexPositions(
    const Mesh& mesh,
    const std::vector<Mat4>& skinMatrices) {
  std::vector<Vec3> positions;
  positions.reserve(mesh.vertices.size());
  for (const Vertex& vertex : mesh.vertices) {
    positions.push_back(skinPosition(vertex, skinMatrices));
  }
  return positions;
}

}  // namespace lcr
