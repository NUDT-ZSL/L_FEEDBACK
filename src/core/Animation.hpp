#pragma once

#include <cstdint>

#include "core/Mesh.hpp"

namespace lcr {

constexpr double kFixedLogicHz = 60.0;
constexpr double kFixedLogicStep = 1.0 / kFixedLogicHz;

class CharacterAnimator {
 public:
  explicit CharacterAnimator(const CharacterAssets& assets);

  // Continuous reference sample. Runtime logic only calls sampleTick(), while
  // deterministic validation uses this function as the analytic reference.
  Pose sample(double seconds) const;
  Pose sampleTick(std::uint64_t tick) const;

  Pose interpolate(const Pose& a, const Pose& b, float alpha) const;
  std::uint64_t tickForTime(double seconds) const;

 private:
  const CharacterAssets& assets_;
};

Pose interpolatePose(const Pose& a, const Pose& b, float alpha);

}  // namespace lcr

