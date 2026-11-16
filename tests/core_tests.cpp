#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/Animation.hpp"
#include "core/CapeSorter.hpp"
#include "core/Diagnostics.hpp"
#include "core/Equipment.hpp"
#include "core/Error.hpp"
#include "core/Mesh.hpp"
#include "core/Serializer.hpp"
#include "core/Simulation.hpp"

namespace {

using namespace lcr;

void check(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error("CHECK FAILED: " + message);
  }
}

void testDeterministicAssets() {
  const CharacterAssets first = buildDeterministicCharacter(424242);
  const CharacterAssets second = buildDeterministicCharacter(424242);
  const CharacterAssets different = buildDeterministicCharacter(424243);

  check(first.skeleton.canonicalBytes() == second.skeleton.canonicalBytes(), "skeleton bytes");
  check(meshCanonicalBytes(first.parts) == meshCanonicalBytes(second.parts), "mesh bytes");
  check(first.skeletonHash == second.skeletonHash, "skeleton hash");
  check(first.meshHash == second.meshHash, "mesh hash");
  check(first.combinedHash == second.combinedHash, "character hash");
  check(first.skeletonHash != different.skeletonHash, "seed should affect skeleton");
  check(first.meshHash != different.meshHash, "seed should affect mesh");

  const char* requiredParts[] = {
      "Torso", "Head", "LeftArm", "RightArm", "LeftLeg", "RightLeg", "Weapon", "Cape"};
  for (const char* name : requiredParts) {
    check(first.findPart(name) != nullptr, std::string("missing part ") + name);
  }
  check(first.findPart("Cape")->transparent, "cape must be transparent");
  check(!first.findPart("Torso")->transparent, "torso must be opaque");
  first.validate();
  std::cout << "determinism skeleton=" << hexU64(first.skeletonHash)
            << " mesh=" << hexU64(first.meshHash) << '\n';
}

void testBoneIndexFailure() {
  CharacterAssets assets = buildDeterministicCharacter(7);
  assets.parts.front().mesh.vertices.front().joints[0] = kBoneCount;
  bool caught = false;
  try {
    assets.validate();
  } catch (const AppError& error) {
    caught = error.code() == ErrorCode::BoneIndex;
  }
  check(caught, "invalid bone index must raise E_BONE_INDEX");
}

void testTextureFailureAndDeterminism() {
  EquipmentBuildOptions options;
  options.seed = 88;
  options.textureSize = 64;
  EquipmentCatalog a;
  EquipmentCatalog b;
  a.build(options);
  b.build(options);
  check(a.sets().size() >= 2U, "need at least two equipment sets");
  check(a.catalogHash() == b.catalogHash(), "equipment catalog hash");
  check(
      a.sets()[0].bodyTexture.rgba == b.sets()[0].bodyTexture.rgba,
      "texture bytes");
  check(
      a.sets()[0].bodyTexture.hash != a.sets()[1].bodyTexture.hash,
      "equipment sets need different textures");

  bool caught = false;
  try {
    EquipmentBuildOptions failing = options;
    failing.forceTextureFailure = true;
    EquipmentCatalog failed;
    failed.build(failing);
  } catch (const AppError& error) {
    caught = error.code() == ErrorCode::TextureGeneration;
  }
  check(caught, "texture generation failure must raise E_TEXTURE_GENERATION");
  std::cout << "texture catalog hash=" << hexU64(a.catalogHash()) << '\n';
}

void testSwitchPoseContinuity() {
  SimulationConfig config;
  config.seed = 2026;
  config.textureSize = 32;
  config.initialEquipment = "iron";
  config.scheduledSwitchTimes = {0.5, 1.5};
  Simulation simulation(config);
  simulation.advanceTo(2.0);
  check(simulation.switchEvents().size() == 2U, "two scheduled switches");
  for (const SwitchEvent& event : simulation.switchEvents()) {
    check(event.triggered, "switch should be triggered");
    check(event.poseHashBefore == event.poseHashAfter, "pose hash must not change");
    check(event.maxJointPositionDelta == 0.0F, "joint position delta must be zero");
  }
  check(simulation.equipment().active().id == "iron", "two toggles return to iron");
  std::cout << "switch deltas=0,0\n";
}

void testInterpolationImprovement() {
  const CharacterAssets assets = buildDeterministicCharacter(314159);
  for (const double fps : {45.0, 144.0}) {
    const InterpolationComparison comparison =
        compareInterpolation(assets, fps, 4.0);
    check(comparison.samples > 0U, "interpolation samples");
    check(
        comparison.withInterpolation.maxPositionError <
            comparison.withoutInterpolation.maxPositionError * 0.25F,
        "interpolation should reduce max joint error by at least 4x");
    check(
        comparison.withInterpolation.rmsPositionError <
            comparison.withoutInterpolation.rmsPositionError * 0.25F,
        "interpolation should reduce RMS joint error by at least 4x");
    std::cout << "interpolation fps=" << fps
              << " max_hold=" << comparison.withoutInterpolation.maxPositionError
              << " max_interp=" << comparison.withInterpolation.maxPositionError
              << " ratio=" << comparison.maxErrorImprovement << '\n';
  }
}

void testVariableRateDeterminism() {
  auto build = []() {
    SimulationConfig config;
    config.seed = 9001;
    config.textureSize = 32;
    config.initialEquipment = "ember";
    config.scheduledSwitchTimes = {0.75, 1.8};
    return Simulation(config);
  };
  Simulation fast = build();
  Simulation slow = build();
  for (int i = 0; i < 144 * 4; ++i) {
    fast.advanceBy(1.0 / 144.0);
  }
  for (int i = 0; i < 45 * 4; ++i) {
    slow.advanceBy(1.0 / 45.0);
  }
  fast.advanceTo(4.0);
  slow.advanceTo(4.0);
  check(fast.logicTick() == slow.logicTick(), "logic tick independent of render rate");
  check(
      fast.stateHash(4.0) == slow.stateHash(4.0),
      "state hash independent of render rate");
  std::cout << "variable-rate state=" << hexU64(fast.stateHash(4.0)) << '\n';
}

CapeSortReport findCrossingSample(const CharacterAssets& assets) {
  const Vec3 cameras[] = {
      {3.4F, 1.9F, -4.3F},
      {-3.4F, 1.9F, -4.3F},
      {4.5F, 1.8F, -1.2F},
      {-4.5F, 1.8F, -1.2F},
      {4.2F, 2.0F, 2.2F},
      {0.1F, 2.2F, -5.2F},
  };
  const CharacterAnimator animator(assets);
  CapeSortReport best;
  for (const Vec3 camera : cameras) {
    for (double time = 0.0; time <= 6.0; time += 1.0 / 30.0) {
      const CapeSortReport report = analyzeCapeSort(assets, animator.sample(time), camera);
      check(report.orderValid, "cape must sort far-to-near");
      if (report.torsoOccludedTriangles > 0U &&
          report.weaponOccludedTriangles > 0U) {
        std::cout << "cape crossing camera=" << camera.x << "," << camera.y << ","
                  << camera.z << " time=" << time << '\n';
        return report;
      }
      if (report.torsoOccludedTriangles + report.weaponOccludedTriangles >
          best.torsoOccludedTriangles + best.weaponOccludedTriangles) {
        best = report;
      }
    }
  }
  return best;
}

void testCapeSortingAndFeather() {
  check(capeEdgeFeather(0.0F, 0.5F) == 0.0F, "border feather must be zero");
  check(capeEdgeFeather(0.5F, 0.5F) == 1.0F, "interior feather must be one");
  check(capeEdgeFeather(0.5F, 0.0F) == 0.0F, "bottom feather must be zero");
  const CharacterAssets assets = buildDeterministicCharacter(1337);
  const CapeSortReport crossing = findCrossingSample(assets);
  std::cout << "cape torso_occluded=" << crossing.torsoOccludedTriangles
            << " weapon_occluded=" << crossing.weaponOccludedTriangles
            << " order=" << (crossing.orderValid ? "valid" : "invalid") << '\n';
  check(crossing.torsoOccludedTriangles > 0U, "sample must include cape behind torso");
  check(crossing.weaponOccludedTriangles > 0U, "sample must include cape behind weapon");
  check(crossing.trianglesInFrontOfOpaque > 0U, "sample must include cape in front");
  check(crossing.orderViolations == 0U, "no back-to-front ordering violations");
}

}  // namespace

int main() {
  try {
    testDeterministicAssets();
    testBoneIndexFailure();
    testTextureFailureAndDeterminism();
    testSwitchPoseContinuity();
    testInterpolationImprovement();
    testVariableRateDeterminism();
    testCapeSortingAndFeather();
    std::cout << "ALL CORE TESTS PASSED\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "TEST FAILURE: " << error.what() << '\n';
    return 1;
  }
}
