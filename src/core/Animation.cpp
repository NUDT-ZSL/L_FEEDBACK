#include "core/Animation.hpp"

#include <algorithm>
#include <cmath>

#include "core/Skeleton.hpp"

namespace lcr {
namespace {

float smoothWave(const double time, const double frequency, const float phase) {
  return static_cast<float>(std::sin(time * frequency * 6.283185307179586 + phase));
}

void addEuler(Transform& transform, const float x, const float y, const float z) {
  transform.rotation = normalize(transform.rotation * fromEulerXyz(x, y, z));
}

}  // namespace

CharacterAnimator::CharacterAnimator(const CharacterAssets& assets) : assets_(assets) {}

Pose CharacterAnimator::sample(const double seconds) const {
  Pose pose = makeBindPose(assets_.skeleton);
  pose.tick = tickForTime(seconds);
  const float a = assets_.animationPhaseA;
  const float b = assets_.animationPhaseB;
  const float c = assets_.animationPhaseC;

  const float turn = 0.52F * smoothWave(seconds, 0.085, a * 0.11F) +
                     0.10F * smoothWave(seconds, 0.23, b);
  addEuler(pose.local[kBoneRoot], 0.0F, turn, 0.0F);

  pose.local[kBonePelvis].translation.y +=
      0.016F * smoothWave(seconds, 0.37, a);
  addEuler(
      pose.local[kBonePelvis],
      0.045F * smoothWave(seconds, 0.30, b),
      0.09F * smoothWave(seconds, 0.18, c),
      0.04F * smoothWave(seconds, 0.24, a));
  addEuler(
      pose.local[kBoneSpine],
      0.035F + 0.035F * smoothWave(seconds, 0.33, c),
      0.13F * smoothWave(seconds, 0.19, b),
      0.05F * smoothWave(seconds, 0.27, a));
  addEuler(
      pose.local[kBoneChest],
      0.055F * smoothWave(seconds, 0.29, a),
      0.20F * smoothWave(seconds, 0.16, c),
      0.07F * smoothWave(seconds, 0.21, b));
  addEuler(
      pose.local[kBoneNeck],
      -0.025F * smoothWave(seconds, 0.32, b),
      -0.10F * smoothWave(seconds, 0.16, c),
      0.0F);
  addEuler(
      pose.local[kBoneHead],
      -0.035F * smoothWave(seconds, 0.26, c),
      -0.13F * smoothWave(seconds, 0.13, a),
      0.02F * smoothWave(seconds, 0.22, b));

  const float leftArmSwing = 0.34F * smoothWave(seconds, 0.23, a);
  const float rightArmSwing = -0.41F * smoothWave(seconds, 0.23, a + 0.6F);
  addEuler(
      pose.local[kBoneLeftUpperArm],
      0.14F * smoothWave(seconds, 0.41, b),
      leftArmSwing,
      0.34F + 0.16F * smoothWave(seconds, 0.17, c));
  addEuler(
      pose.local[kBoneLeftLowerArm],
      0.0F,
      -0.17F * smoothWave(seconds, 0.29, c),
      -0.18F - 0.12F * smoothWave(seconds, 0.35, a));
  addEuler(
      pose.local[kBoneRightUpperArm],
      -0.11F * smoothWave(seconds, 0.39, c),
      rightArmSwing,
      -0.40F - 0.18F * smoothWave(seconds, 0.19, b));
  addEuler(
      pose.local[kBoneRightLowerArm],
      0.0F,
      0.21F * smoothWave(seconds, 0.31, b),
      0.24F + 0.16F * smoothWave(seconds, 0.25, a));
  addEuler(
      pose.local[kBoneRightHand],
      0.28F * smoothWave(seconds, 0.44, c),
      0.0F,
      0.0F);

  const float stride = 0.37F * smoothWave(seconds, 0.18, b);
  addEuler(pose.local[kBoneLeftThigh], stride, 0.0F, -0.018F);
  addEuler(
      pose.local[kBoneLeftShin],
      std::fmax(0.0F, -stride * 0.82F) + 0.07F,
      0.0F,
      0.0F);
  addEuler(pose.local[kBoneLeftFoot], -stride * 0.35F, 0.0F, 0.0F);
  addEuler(pose.local[kBoneRightThigh], -stride, 0.0F, 0.018F);
  addEuler(
      pose.local[kBoneRightShin],
      std::fmax(0.0F, stride * 0.82F) + 0.07F,
      0.0F,
      0.0F);
  addEuler(pose.local[kBoneRightFoot], stride * 0.35F, 0.0F, 0.0F);

  // Secondary cape motion intentionally lags the torso and can make the cape
  // cross the torso, arms and weapon during a turn.
  addEuler(
      pose.local[kBoneCapeRoot],
      0.05F + 0.12F * smoothWave(seconds, 0.19, c),
      0.17F * smoothWave(seconds, 0.11, a + 0.45F),
      0.05F * smoothWave(seconds, 0.23, b));
  addEuler(
      pose.local[kBoneCapeMid],
      0.06F + 0.16F * smoothWave(seconds, 0.18, a + 0.85F),
      0.15F * smoothWave(seconds, 0.10, b + 0.40F),
      0.04F * smoothWave(seconds, 0.21, c));
  addEuler(
      pose.local[kBoneCapeTip],
      0.09F + 0.19F * smoothWave(seconds, 0.16, b + 1.15F),
      0.19F * smoothWave(seconds, 0.09, c + 0.65F),
      0.03F * smoothWave(seconds, 0.20F, a));

  return pose;
}

Pose CharacterAnimator::sampleTick(const std::uint64_t tick) const {
  Pose pose = sample(static_cast<double>(tick) * kFixedLogicStep);
  pose.tick = tick;
  return pose;
}

std::uint64_t CharacterAnimator::tickForTime(const double seconds) const {
  if (seconds <= 0.0) {
    return 0;
  }
  return static_cast<std::uint64_t>(std::floor(seconds * kFixedLogicHz + 1.0e-9));
}

Pose CharacterAnimator::interpolate(const Pose& a, const Pose& b, const float alpha) const {
  return interpolatePose(a, b, alpha);
}

Pose interpolatePose(const Pose& a, const Pose& b, const float alpha) {
  if (a.local.size() != b.local.size()) {
    return a;
  }
  const float t = std::fmax(0.0F, std::fmin(1.0F, alpha));
  Pose result;
  result.tick = a.tick;
  result.local.resize(a.local.size());
  for (std::size_t i = 0; i < a.local.size(); ++i) {
    result.local[i].translation = lerp(a.local[i].translation, b.local[i].translation, t);
    result.local[i].rotation = slerp(a.local[i].rotation, b.local[i].rotation, t);
    result.local[i].scale = lerp(a.local[i].scale, b.local[i].scale, t);
  }
  return result;
}

}  // namespace lcr
