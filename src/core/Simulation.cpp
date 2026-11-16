#include "core/Simulation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/Error.hpp"
#include "core/Serializer.hpp"

namespace lcr {

Simulation::Simulation(SimulationConfig config)
    : config_(std::move(config)),
      assets_(buildDeterministicCharacter(config_.seed)),
      animator_(assets_) {
  EquipmentBuildOptions options;
  options.seed = config_.seed;
  options.textureSize = config_.textureSize;
  options.forceTextureFailure = config_.forceTextureFailure;
  equipment_.build(options);
  if (!equipment_.switchById(config_.initialEquipment)) {
    fail(
        ErrorCode::InvalidArgument,
        "simulation.equipment",
        "unknown initial equipment id: " + config_.initialEquipment);
  }
  std::sort(
      config_.scheduledSwitchTimes.begin(),
      config_.scheduledSwitchTimes.end());
}

bool Simulation::advanceFixedStep() {
  ++logicTick_;
  const double currentTime = logicTime();
  fireScheduledSwitches(currentTime);
  return true;
}

void Simulation::advanceBy(double deltaSeconds) {
  if (!std::isfinite(deltaSeconds) || deltaSeconds <= 0.0) {
    return;
  }
  // Avoid unbounded catch-up after a debugger break or window drag.
  const double capped = std::fmin(deltaSeconds, 0.25);
  accumulator_ += capped;
  while (accumulator_ + 1.0e-12 >= kFixedLogicStep) {
    accumulator_ -= kFixedLogicStep;
    advanceFixedStep();
  }
  if (accumulator_ < 0.0) {
    accumulator_ = 0.0;
  }
}

void Simulation::advanceTo(const double absoluteTime) {
  while (renderTime() + 1.0e-12 < absoluteTime) {
    advanceBy(std::min(0.25, absoluteTime - renderTime()));
  }
}

void Simulation::fireScheduledSwitches(const double currentTime) {
  while (nextScheduledSwitch_ < config_.scheduledSwitchTimes.size()) {
    const double scheduled = config_.scheduledSwitchTimes[nextScheduledSwitch_];
    if (scheduled > currentTime + 1.0e-9) {
      break;
    }
    ++nextScheduledSwitch_;
    switchEquipmentNow(currentTime);
  }
}

Pose Simulation::heldPose(const double renderTime) const {
  const std::uint64_t tick =
      static_cast<std::uint64_t>(std::floor(std::fmax(0.0, renderTime) * kFixedLogicHz + 1.0e-9));
  return animator_.sampleTick(tick);
}

Pose Simulation::interpolatedPose(const double renderTime) const {
  const double clampedTime = std::fmax(0.0, renderTime);
  const std::uint64_t tick =
      static_cast<std::uint64_t>(std::floor(clampedTime * kFixedLogicHz + 1.0e-9));
  const double fractional = clampedTime * kFixedLogicHz - static_cast<double>(tick);
  const Pose a = animator_.sampleTick(tick);
  const Pose b = animator_.sampleTick(tick + 1U);
  return animator_.interpolate(a, b, static_cast<float>(fractional));
}

SwitchEvent Simulation::switchEquipmentNow(const double atRenderTime) {
  const Pose before = interpolatedPose(atRenderTime);
  SwitchEvent event;
  event.tick = logicTick_;
  event.renderTime = atRenderTime;
  event.fromEquipment = equipment_.active().id;
  event.poseHashBefore = poseHash(before);

  if (!equipment_.toggle()) {
    fail(
        ErrorCode::InternalInvariant,
        "simulation.switchEquipment",
        "equipment catalog needs at least two sets");
  }

  const Pose after = interpolatedPose(atRenderTime);
  event.toEquipment = equipment_.active().id;
  event.poseHashAfter = poseHash(after);
  event.maxJointPositionDelta = maxJointPositionDelta(assets_.skeleton, before, after);
  event.triggered = true;
  switchEvents_.push_back(event);
  return event;
}

std::uint64_t Simulation::stateHash(const double renderTime) const {
  const Pose pose = interpolatedPose(renderTime);
  CanonicalWriter writer;
  writer.string("lcr-simulation-state-v1");
  writer.u64(config_.seed);
  writer.u64(logicTick_);
  writer.f32(static_cast<float>(renderTime));
  writer.u64(assets_.combinedHash);
  writer.u64(equipment_.catalogHash());
  writer.string(equipment_.active().id);
  writer.u64(poseHash(pose));
  writer.u32(static_cast<std::uint32_t>(switchEvents_.size()));
  for (const SwitchEvent& event : switchEvents_) {
    writer.u64(event.tick);
    writer.string(event.fromEquipment);
    writer.string(event.toEquipment);
    writer.u64(event.poseHashBefore);
    writer.u64(event.poseHashAfter);
    writer.f32(event.maxJointPositionDelta);
  }
  return hashBytes(writer.bytes());
}

}  // namespace lcr
