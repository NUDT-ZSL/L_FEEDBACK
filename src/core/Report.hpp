#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/CapeSorter.hpp"
#include "core/Diagnostics.hpp"
#include "core/Simulation.hpp"

namespace lcr {

struct CapeSampleReport {
  double renderTime = 0.0;
  std::string equipment;
  CapeSortReport sort;
};

struct PipelineReport {
  bool initialized = false;
  std::uint32_t shaderProgramBuilds = 0;
  std::uint32_t vaoBuilds = 0;
  std::uint32_t textureUploads = 0;
  std::uint32_t equipmentPipelineRebuilds = 0;
  std::string openGlVersion;
  std::string rendererName;
};

struct RunReport {
  std::string timingMode = "synthetic";
  SimulationConfig config;
  std::uint64_t logicSteps = 0;
  double simulatedSeconds = 0.0;
  std::uint32_t renderFrames = 0;
  FrameTimingStats frameStats;
  std::uint64_t skeletonHash = 0;
  std::uint64_t meshHash = 0;
  std::uint64_t characterHash = 0;
  std::uint64_t equipmentCatalogHash = 0;
  std::uint64_t finalStateHash = 0;
  std::string finalEquipment;
  std::vector<EquipmentSet> equipmentSets;
  std::vector<InterpolationComparison> interpolation;
  std::vector<CapeSampleReport> capeSamples;
  std::vector<SwitchEvent> switchEvents;
  PipelineReport pipeline;
};

void writeValidationReport(const std::string& path, const RunReport& report);
void writeStateSnapshot(
    const std::string& path,
    const Simulation& simulation,
    double renderTime,
    const PipelineReport& pipeline);

}  // namespace lcr

