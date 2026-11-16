#include "core/Skeleton.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "core/Error.hpp"
#include "core/Rng.hpp"
#include "core/Serializer.hpp"

namespace lcr {
namespace {

void appendTransform(CanonicalWriter& writer, const Transform& transform) {
  writer.f32(transform.translation.x);
  writer.f32(transform.translation.y);
  writer.f32(transform.translation.z);
  writer.f32(transform.rotation.w);
  writer.f32(transform.rotation.x);
  writer.f32(transform.rotation.y);
  writer.f32(transform.rotation.z);
  writer.f32(transform.scale.x);
  writer.f32(transform.scale.y);
  writer.f32(transform.scale.z);
}

bool validTransform(const Transform& transform) {
  return isFinite(transform.translation) && isFinite(transform.rotation) &&
         isFinite(transform.scale.x) && isFinite(transform.scale.y) &&
         isFinite(transform.scale.z);
}

}  // namespace

std::vector<std::uint8_t> Skeleton::canonicalBytes() const {
  CanonicalWriter writer;
  writer.string("lcr-skeleton-v1");
  writer.u32(static_cast<std::uint32_t>(bones.size()));
  for (const Bone& bone : bones) {
    writer.string(bone.name);
    writer.i32(bone.parent);
    appendTransform(writer, bone.bindLocal);
  }
  for (const Mat4& matrix : inverseBind) {
    for (const float value : matrix.m) {
      writer.f32(value);
    }
  }
  return writer.bytes();
}

std::uint64_t Skeleton::computeHash() const { return lcr::hashBytes(canonicalBytes()); }

std::vector<Mat4> Skeleton::computeWorldMatrices(const Pose& pose) const {
  if (pose.local.size() != bones.size()) {
    fail(
        ErrorCode::InternalInvariant,
        "skeleton.computeWorldMatrices",
        "pose bone count does not match skeleton bone count");
  }
  std::vector<Mat4> world(bones.size(), Mat4::identity());
  for (std::size_t i = 0; i < bones.size(); ++i) {
    const Mat4 local = compose(pose.local[i]);
    if (bones[i].parent >= 0) {
      const int parent = bones[i].parent;
      if (parent >= static_cast<int>(i)) {
        fail(ErrorCode::BoneIndex, "skeleton.parentOrder", "parent bone must precede child bone");
      }
      world[i] = world[static_cast<std::size_t>(parent)] * local;
    } else {
      world[i] = local;
    }
  }
  return world;
}

std::vector<Vec3> Skeleton::jointWorldPositions(const Pose& pose) const {
  const std::vector<Mat4> world = computeWorldMatrices(pose);
  std::vector<Vec3> positions;
  positions.reserve(world.size());
  for (const Mat4& matrix : world) {
    positions.push_back({matrix.at(0, 3), matrix.at(1, 3), matrix.at(2, 3)});
  }
  return positions;
}

std::vector<Mat4> Skeleton::computeSkinMatrices(const Pose& pose) const {
  const std::vector<Mat4> world = computeWorldMatrices(pose);
  std::vector<Mat4> result;
  result.reserve(world.size());
  for (std::size_t i = 0; i < world.size(); ++i) {
    result.push_back(world[i] * inverseBind[i]);
  }
  return result;
}

void Skeleton::validate() const {
  if (bones.size() != static_cast<std::size_t>(kBoneCount)) {
    fail(ErrorCode::BoneIndex, "skeleton.validate", "unexpected bone count");
  }
  if (bones.empty() || bones.front().parent != -1) {
    fail(ErrorCode::BoneIndex, "skeleton.validate", "root bone is missing or has a parent");
  }
  for (std::size_t i = 0; i < bones.size(); ++i) {
    const Bone& bone = bones[i];
    if (bone.parent < -1 || bone.parent >= static_cast<int>(bones.size())) {
      fail(ErrorCode::BoneIndex, "skeleton.validate", "parent bone index is out of range");
    }
    if (i > 0 && bone.parent >= static_cast<int>(i)) {
      fail(ErrorCode::BoneIndex, "skeleton.validate", "bone hierarchy is not topologically ordered");
    }
    if (bone.name.empty() || !validTransform(bone.bindLocal)) {
      fail(ErrorCode::InternalInvariant, "skeleton.validate", "invalid bone data");
    }
  }
  if (inverseBind.size() != bones.size() || bindWorldPositions.size() != bones.size()) {
    fail(ErrorCode::BoneIndex, "skeleton.validate", "bind matrices are incomplete");
  }
  for (const Mat4& matrix : inverseBind) {
    for (const float value : matrix.m) {
      if (!isFinite(value)) {
        fail(ErrorCode::InternalInvariant, "skeleton.validate", "non-finite inverse bind matrix");
      }
    }
  }
}

std::uint64_t poseHash(const Pose& pose) {
  CanonicalWriter writer;
  writer.string("lcr-pose-v1");
  writer.u64(pose.tick);
  writer.u32(static_cast<std::uint32_t>(pose.local.size()));
  for (const Transform& transform : pose.local) {
    appendTransform(writer, transform);
  }
  return hashBytes(writer.bytes());
}

float maxJointPositionDelta(const Skeleton& skeleton, const Pose& a, const Pose& b) {
  const std::vector<Mat4> wa = skeleton.computeWorldMatrices(a);
  const std::vector<Mat4> wb = skeleton.computeWorldMatrices(b);
  float maxDelta = 0.0F;
  for (std::size_t i = 0; i < wa.size(); ++i) {
    const Vec3 pa{wa[i].at(0, 3), wa[i].at(1, 3), wa[i].at(2, 3)};
    const Vec3 pb{wb[i].at(0, 3), wb[i].at(1, 3), wb[i].at(2, 3)};
    maxDelta = std::fmax(maxDelta, length(pa - pb));
  }
  return maxDelta;
}

SkeletonBuildResult buildDeterministicSkeleton(const std::uint64_t seed) {
  DeterministicRng rng(seed ^ 0x51A7E5C3D2B109F1ULL);

  auto addBone = [](std::vector<Bone>& bones,
                    const char* name,
                    const int parent,
                    const Vec3 translation) {
    Bone bone;
    bone.name = name;
    bone.parent = parent;
    bone.bindLocal.translation = translation;
    bones.push_back(std::move(bone));
  };

  std::vector<Bone> bones;
  bones.reserve(kBoneCount);
  addBone(bones, "Root", -1, {0.0F, 0.0F, 0.0F});
  addBone(bones, "Pelvis", kBoneRoot, {0.0F, 0.96F, 0.0F});
  addBone(bones, "Spine", kBonePelvis, {0.0F, 0.18F, 0.0F});
  addBone(bones, "Chest", kBoneSpine, {0.0F, 0.42F, 0.0F});
  addBone(bones, "Neck", kBoneChest, {0.0F, 0.22F, 0.0F});
  addBone(bones, "Head", kBoneNeck, {0.0F, 0.13F, 0.0F});
  addBone(bones, "LeftUpperArm", kBoneChest, {-0.24F, 0.15F, 0.0F});
  addBone(bones, "LeftLowerArm", kBoneLeftUpperArm, {-0.32F, 0.0F, 0.0F});
  addBone(bones, "LeftHand", kBoneLeftLowerArm, {-0.30F, 0.0F, 0.0F});
  addBone(bones, "RightUpperArm", kBoneChest, {0.24F, 0.15F, 0.0F});
  addBone(bones, "RightLowerArm", kBoneRightUpperArm, {0.32F, 0.0F, 0.0F});
  addBone(bones, "RightHand", kBoneRightLowerArm, {0.30F, 0.0F, 0.0F});
  addBone(bones, "LeftThigh", kBonePelvis, {-0.12F, -0.08F, 0.0F});
  addBone(bones, "LeftShin", kBoneLeftThigh, {0.0F, -0.45F, 0.0F});
  addBone(bones, "LeftFoot", kBoneLeftShin, {0.0F, -0.43F, 0.06F});
  addBone(bones, "RightThigh", kBonePelvis, {0.12F, -0.08F, 0.0F});
  addBone(bones, "RightShin", kBoneRightThigh, {0.0F, -0.45F, 0.0F});
  addBone(bones, "RightFoot", kBoneRightShin, {0.0F, -0.43F, 0.06F});
  addBone(bones, "WeaponMount", kBoneRightHand, {-0.14F, 0.02F, 0.03F});
  addBone(bones, "CapeRoot", kBoneChest, {0.0F, 0.02F, -0.16F});
  addBone(bones, "CapeMid", kBoneCapeRoot, {0.0F, -0.44F, -0.035F});
  addBone(bones, "CapeTip", kBoneCapeMid, {0.0F, -0.43F, -0.025F});

  // Seed-dependent anatomy variation is intentionally small but affects every
  // joint and therefore all mesh bind data and animation phases.
  for (std::size_t i = 1; i < bones.size(); ++i) {
    Vec3& t = bones[i].bindLocal.translation;
    t.x += rng.range(-0.012F, 0.012F);
    t.y += rng.range(-0.012F, 0.012F);
    t.z += rng.range(-0.010F, 0.010F);
    const float yaw = rng.range(-0.025F, 0.025F);
    bones[i].bindLocal.rotation = fromAxisAngle({0.0F, 1.0F, 0.0F}, yaw);
  }

  Skeleton skeleton;
  skeleton.bones = std::move(bones);
  const Pose bindPose = makeBindPose(skeleton);
  const std::vector<Mat4> bindWorld = skeleton.computeWorldMatrices(bindPose);
  skeleton.inverseBind.reserve(bindWorld.size());
  skeleton.bindWorldPositions.reserve(bindWorld.size());
  for (const Mat4& matrix : bindWorld) {
    skeleton.inverseBind.push_back(inverseRigid(matrix));
    skeleton.bindWorldPositions.push_back(
        {matrix.at(0, 3), matrix.at(1, 3), matrix.at(2, 3)});
  }
  skeleton.dataHash = skeleton.computeHash();
  skeleton.validate();

  SkeletonBuildResult result;
  result.skeleton = std::move(skeleton);
  result.phaseA = rng.range(0.0F, kPi * 2.0F);
  result.phaseB = rng.range(0.0F, kPi * 2.0F);
  result.phaseC = rng.range(0.0F, kPi * 2.0F);
  return result;
}

Pose makeBindPose(const Skeleton& skeleton) {
  Pose pose;
  pose.tick = 0;
  pose.local.reserve(skeleton.bones.size());
  for (const Bone& bone : skeleton.bones) {
    pose.local.push_back(bone.bindLocal);
  }
  return pose;
}

}  // namespace lcr
