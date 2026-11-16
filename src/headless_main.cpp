#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "core/CapeSorter.hpp"
#include "core/Cli.hpp"
#include "core/Diagnostics.hpp"
#include "core/Error.hpp"
#include "core/Report.hpp"
#include "core/Serializer.hpp"
#include "core/Simulation.hpp"

namespace {

using namespace lcr;

void injectBoneIndexFailure(Simulation& simulation) {
  CharacterAssets& assets = simulation.assets();
  if (assets.parts.empty() || assets.parts.front().mesh.vertices.empty()) {
    fail(ErrorCode::InternalInvariant, "fault.bone", "no vertex available for fault injection");
  }
  assets.parts.front().mesh.vertices.front().joints[0] = kBoneCount;
  assets.validate();
}

RunReport runHeadless(const CliOptions& options) {
  SimulationConfig config;
  config.seed = options.seed;
  config.textureSize = options.textureSize;
  config.initialEquipment = options.equipment;
  config.scheduledSwitchTimes = options.switchTimes;
  config.forceTextureFailure = options.fault == FaultKind::Texture;

  Simulation simulation(config);
  if (options.fault == FaultKind::Bone) {
    injectBoneIndexFailure(simulation);
  }

  const double duration =
      static_cast<double>(options.logicFrames) / kFixedLogicHz;
  const std::uint64_t scheduledFrames =
      static_cast<std::uint64_t>(std::ceil(duration * options.renderFps));
  const std::uint64_t frameLimit =
      options.maxRenderFrames == 0U
          ? scheduledFrames
          : std::min<std::uint64_t>(scheduledFrames, options.maxRenderFrames);

  std::vector<double> frameTimes;
  frameTimes.reserve(static_cast<std::size_t>(frameLimit));
  for (std::uint64_t frame = 0; frame < frameLimit; ++frame) {
    const double target = std::min(
        duration,
        static_cast<double>(frame) / options.renderFps);
    simulation.advanceTo(target);
    frameTimes.push_back(1000.0 / options.renderFps);
  }
  simulation.advanceTo(duration);
  while (simulation.logicTick() < options.logicFrames) {
    simulation.advanceFixedStep();
  }

  for (const SwitchEvent& event : simulation.switchEvents()) {
    if (event.maxJointPositionDelta > 1.0e-7F) {
      fail(
          ErrorCode::InternalInvariant,
          "validation.switchContinuity",
          "equipment switch changed the animation pose");
    }
  }

  RunReport report;
  report.timingMode = "deterministic_synthetic";
  report.config = config;
  report.logicSteps = simulation.logicTick();
  report.simulatedSeconds = duration;
  report.renderFrames = static_cast<std::uint32_t>(frameTimes.size());
  report.frameStats = summarizeFrameTimes(frameTimes);
  report.skeletonHash = simulation.assets().skeletonHash;
  report.meshHash = simulation.assets().meshHash;
  report.characterHash = simulation.assets().combinedHash;
  report.equipmentCatalogHash = simulation.equipment().catalogHash();
  report.finalStateHash = simulation.stateHash(duration);
  report.finalEquipment = simulation.equipment().active().id;
  report.equipmentSets = simulation.equipment().sets();
  report.switchEvents = simulation.switchEvents();
  report.pipeline.initialized = false;

  report.interpolation.push_back(compareInterpolation(simulation.assets(), 45.0, duration));
  report.interpolation.push_back(compareInterpolation(simulation.assets(), 144.0, duration));

  const Vec3 camera{4.5F, 1.8F, -1.2F};
  std::vector<double> capeTimes;
  for (double time = 0.35; time <= duration + 1.0e-9; time += 0.25) {
    capeTimes.push_back(time);
  }
  for (const double time : capeTimes) {
    CapeSampleReport sample;
    sample.renderTime = time;
    sample.equipment = simulation.config().initialEquipment;
    for (const SwitchEvent& event : simulation.switchEvents()) {
      if (event.renderTime <= time + 1.0e-9) {
        sample.equipment = event.toEquipment;
      }
    }
    sample.sort = analyzeCapeSort(
        simulation.assets(),
        simulation.interpolatedPose(time),
        camera);
    report.capeSamples.push_back(sample);
  }
  writeStateSnapshot(
      options.snapshotPath,
      simulation,
      duration,
      report.pipeline);
  return report;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions options = parseCommandLine(argc, argv);
    if (options.help) {
      printUsage(argv[0]);
      return 0;
    }
    if (options.fault == FaultKind::Shader) {
      fail(
          ErrorCode::ShaderCompile,
          "shader.compile",
          "forced shader compile failure (headless deterministic fault injection)");
    }

    const RunReport report = runHeadless(options);
    writeValidationReport(options.reportPath, report);

    std::cout << "{\"status\":\"ok\""
              << ",\"seed\":" << options.seed
              << ",\"logic_steps\":" << report.logicSteps
              << ",\"render_frames\":" << report.renderFrames
              << ",\"state_hash\":\"" << hexU64(report.finalStateHash)
              << "\",\"report\":\"" << options.reportPath
              << "\",\"snapshot\":\"" << options.snapshotPath
              << "\"}\n";
    return 0;
  } catch (const lcr::AppError& error) {
    lcr::emitErrorToStderr(error);
    return lcr::errorExitCode(error.code());
  } catch (const std::exception& error) {
    lcr::emitErrorToStderr(lcr::AppError(
        lcr::ErrorCode::InternalInvariant,
        "main.unhandled",
        error.what()));
    return lcr::errorExitCode(lcr::ErrorCode::InternalInvariant);
  }
}
