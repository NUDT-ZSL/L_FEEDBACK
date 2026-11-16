#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/Animation.hpp"

namespace lcr {

struct JointErrorStats {
  float maxPositionError = 0.0F;
  float rmsPositionError = 0.0F;
  std::uint32_t samples = 0;
};

struct InterpolationComparison {
  double renderFps = 60.0;
  double durationSeconds = 3.0;
  std::uint32_t samples = 0;
  JointErrorStats withoutInterpolation;
  JointErrorStats withInterpolation;
  float maxErrorImprovement = 0.0F;
  float rmsErrorImprovement = 0.0F;
};

InterpolationComparison compareInterpolation(
    const CharacterAssets& assets,
    double renderFps,
    double durationSeconds);

float poseMaxJointError(
    const Skeleton& skeleton,
    const Pose& reference,
    const Pose& observed);

struct FrameTimingStats {
  std::uint32_t frameCount = 0;
  double minMs = 0.0;
  double averageMs = 0.0;
  double p95Ms = 0.0;
  double maxMs = 0.0;
};

FrameTimingStats summarizeFrameTimes(const std::vector<double>& frameTimesMs);

}  // namespace lcr

