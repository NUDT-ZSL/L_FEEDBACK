#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "core/Math.hpp"

namespace lcr {

enum BoneId : int {
  kBoneRoot = 0,
  kBonePelvis,
  kBoneSpine,
  kBoneChest,
  kBoneNeck,
  kBoneHead,
  kBoneLeftUpperArm,
  kBoneLeftLowerArm,
  kBoneLeftHand,
  kBoneRightUpperArm,
  kBoneRightLowerArm,
  kBoneRightHand,
  kBoneLeftThigh,
  kBoneLeftShin,
  kBoneLeftFoot,
  kBoneRightThigh,
  kBoneRightShin,
  kBoneRightFoot,
  kBoneWeaponMount,
  kBoneCapeRoot,
  kBoneCapeMid,
  kBoneCapeTip,
  kBoneCount,
};

constexpr int kMaxBoneInfluences = 4;

struct Bone {
  std::string name;
  int parent = -1;
  Transform bindLocal;
};

struct Pose {
  std::uint64_t tick = 0;
  std::vector<Transform> local;
};

struct Skeleton {
  std::vector<Bone> bones;
  std::vector<Mat4> inverseBind;
  std::vector<Vec3> bindWorldPositions;
  std::uint64_t dataHash = 0;

  std::uint64_t computeHash() const;
  std::vector<std::uint8_t> canonicalBytes() const;
  std::vector<Mat4> computeWorldMatrices(const Pose& pose) const;
  std::vector<Vec3> jointWorldPositions(const Pose& pose) const;
  std::vector<Mat4> computeSkinMatrices(const Pose& pose) const;
  void validate() const;
};

struct SkeletonBuildResult {
  Skeleton skeleton;
  float phaseA = 0.0F;
  float phaseB = 0.0F;
  float phaseC = 0.0F;
};

SkeletonBuildResult buildDeterministicSkeleton(std::uint64_t seed);
Pose makeBindPose(const Skeleton& skeleton);

std::uint64_t poseHash(const Pose& pose);
float maxJointPositionDelta(
    const Skeleton& skeleton,
    const Pose& a,
    const Pose& b);

}  // namespace lcr

