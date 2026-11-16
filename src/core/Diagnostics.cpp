#include "core/Diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lcr {
namespace {

float rmsJointError(
    const Skeleton& skeleton,
    const Pose& reference,
    const Pose& observed) {
  const std::vector<Mat4> referenceWorld = skeleton.computeWorldMatrices(reference);
  const std::vector<Mat4> observedWorld = skeleton.computeWorldMatrices(observed);
  double sumSquared = 0.0;
  for (std::size_t i = 0; i < referenceWorld.size(); ++i) {
    const Vec3 a{
        referenceWorld[i].at(0, 3),
        referenceWorld[i].at(1, 3),
        referenceWorld[i].at(2, 3)};
    const Vec3 b{
        observedWorld[i].at(0, 3),
        observedWorld[i].at(1, 3),
        observedWorld[i].at(2, 3)};
    const Vec3 delta = a - b;
    sumSquared += static_cast<double>(dot(delta, delta));
  }
  const std::size_t count = std::max<std::size_t>(1U, referenceWorld.size());
  return static_cast<float>(std::sqrt(sumSquared / static_cast<double>(count)));
}

}  // namespace

float poseMaxJointError(
    const Skeleton& skeleton,
    const Pose& reference,
    const Pose& observed) {
  const std::vector<Mat4> referenceWorld = skeleton.computeWorldMatrices(reference);
  const std::vector<Mat4> observedWorld = skeleton.computeWorldMatrices(observed);
  float maxError = 0.0F;
  for (std::size_t i = 0; i < referenceWorld.size(); ++i) {
    const Vec3 a{
        referenceWorld[i].at(0, 3),
        referenceWorld[i].at(1, 3),
        referenceWorld[i].at(2, 3)};
    const Vec3 b{
        observedWorld[i].at(0, 3),
        observedWorld[i].at(1, 3),
        observedWorld[i].at(2, 3)};
    maxError = std::fmax(maxError, length(a - b));
  }
  return maxError;
}

InterpolationComparison compareInterpolation(
    const CharacterAssets& assets,
    const double renderFps,
    const double durationSeconds) {
  InterpolationComparison comparison;
  comparison.renderFps = renderFps;
  comparison.durationSeconds = durationSeconds;
  if (renderFps <= 0.0 || durationSeconds <= 0.0) {
    return comparison;
  }

  CharacterAnimator animator(assets);
  const std::uint32_t frameCount =
      static_cast<std::uint32_t>(std::floor(renderFps * durationSeconds));
  double maxErrorWithout = 0.0;
  double maxErrorWith = 0.0;
  double squaredWithout = 0.0;
  double squaredWith = 0.0;
  for (std::uint32_t frame = 1; frame <= frameCount; ++frame) {
    const double time = static_cast<double>(frame) / renderFps;
    const Pose reference = animator.sample(time);
    const std::uint64_t tick = static_cast<std::uint64_t>(std::floor(time * kFixedLogicHz));
    const Pose previous = animator.sampleTick(tick);
    const Pose next = animator.sampleTick(tick + 1U);
    const float alpha = static_cast<float>(time * kFixedLogicHz - static_cast<double>(tick));
    const Pose held = previous;
    const Pose interpolated = animator.interpolate(previous, next, alpha);
    const float errorWithout = poseMaxJointError(assets.skeleton, reference, held);
    const float errorWith = poseMaxJointError(assets.skeleton, reference, interpolated);
    const float rmsWithout = rmsJointError(assets.skeleton, reference, held);
    const float rmsWith = rmsJointError(assets.skeleton, reference, interpolated);
    maxErrorWithout = std::fmax(maxErrorWithout, errorWithout);
    maxErrorWith = std::fmax(maxErrorWith, errorWith);
    squaredWithout += static_cast<double>(rmsWithout) * rmsWithout;
    squaredWith += static_cast<double>(rmsWith) * rmsWith;
  }
  comparison.samples = frameCount;
  comparison.withoutInterpolation.maxPositionError = static_cast<float>(maxErrorWithout);
  comparison.withInterpolation.maxPositionError = static_cast<float>(maxErrorWith);
  comparison.withoutInterpolation.rmsPositionError =
      static_cast<float>(std::sqrt(squaredWithout / std::max(1U, frameCount)));
  comparison.withInterpolation.rmsPositionError =
      static_cast<float>(std::sqrt(squaredWith / std::max(1U, frameCount)));
  comparison.withoutInterpolation.samples = frameCount;
  comparison.withInterpolation.samples = frameCount;
  if (maxErrorWithout > 0.0) {
    comparison.maxErrorImprovement = static_cast<float>(maxErrorWith / maxErrorWithout);
  }
  if (comparison.withoutInterpolation.rmsPositionError > 0.0F) {
    comparison.rmsErrorImprovement =
        comparison.withInterpolation.rmsPositionError /
        comparison.withoutInterpolation.rmsPositionError;
  }
  return comparison;
}

FrameTimingStats summarizeFrameTimes(const std::vector<double>& frameTimesMs) {
  FrameTimingStats stats;
  stats.frameCount = static_cast<std::uint32_t>(frameTimesMs.size());
  if (frameTimesMs.empty()) {
    return stats;
  }
  std::vector<double> sorted = frameTimesMs;
  std::sort(sorted.begin(), sorted.end());
  double sum = 0.0;
  for (const double value : sorted) {
    sum += value;
  }
  stats.minMs = sorted.front();
  stats.averageMs = sum / static_cast<double>(sorted.size());
  stats.maxMs = sorted.back();
  const std::size_t p95Index = static_cast<std::size_t>(
      std::ceil(0.95 * static_cast<double>(sorted.size() - 1U)));
  stats.p95Ms = sorted[std::min(p95Index, sorted.size() - 1U)];
  return stats;
}

}  // namespace lcr
