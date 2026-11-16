#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/Animation.hpp"
#include "core/Equipment.hpp"

namespace lcr {

struct SwitchEvent {
  std::uint64_t tick = 0;
  double renderTime = 0.0;
  std::string fromEquipment;
  std::string toEquipment;
  std::uint64_t poseHashBefore = 0;
  std::uint64_t poseHashAfter = 0;
  float maxJointPositionDelta = 0.0F;
  bool triggered = false;
};

struct SimulationConfig {
  std::uint64_t seed = 1337;
  std::uint32_t textureSize = 128;
  std::string initialEquipment = "iron";
  std::vector<double> scheduledSwitchTimes;
  bool forceTextureFailure = false;
};

class Simulation {
 public:
  explicit Simulation(SimulationConfig config);

  void advanceBy(double deltaSeconds);
  void advanceTo(double absoluteTime);
  bool advanceFixedStep();

  Pose heldPose(double renderTime) const;
  Pose interpolatedPose(double renderTime) const;
  Pose currentRenderPose() const { return interpolatedPose(renderTime()); }

  SwitchEvent switchEquipmentNow(double atRenderTime);

  const CharacterAssets& assets() const { return assets_; }
  CharacterAssets& assets() { return assets_; }
  const EquipmentCatalog& equipment() const { return equipment_; }
  EquipmentCatalog& equipment() { return equipment_; }
  const CharacterAnimator& animator() const { return animator_; }
  const std::vector<SwitchEvent>& switchEvents() const { return switchEvents_; }
  const SimulationConfig& config() const { return config_; }

  std::uint64_t logicTick() const { return logicTick_; }
  double logicTime() const { return static_cast<double>(logicTick_) * kFixedLogicStep; }
  double renderTime() const { return logicTime() + accumulator_; }
  double accumulator() const { return accumulator_; }
  float interpolationAlpha() const {
    return static_cast<float>(accumulator_ / kFixedLogicStep);
  }
  std::uint64_t stateHash(double renderTime) const;

 private:
  void fireScheduledSwitches(double currentTime);

  SimulationConfig config_;
  CharacterAssets assets_;
  EquipmentCatalog equipment_;
  CharacterAnimator animator_;
  std::uint64_t logicTick_ = 0;
  double accumulator_ = 0.0;
  std::size_t nextScheduledSwitch_ = 0;
  std::vector<SwitchEvent> switchEvents_;
};

}  // namespace lcr

