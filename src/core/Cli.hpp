#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lcr {

enum class FaultKind {
  None,
  Texture,
  Shader,
  Bone,
};

struct CliOptions {
  bool headless = false;
  bool help = false;
  std::uint64_t seed = 1337;
  std::uint32_t logicFrames = 240;
  bool logicFramesSpecified = false;
  std::uint32_t maxRenderFrames = 0;
  double renderFps = 120.0;
  std::uint32_t textureSize = 128;
  std::string equipment = "iron";
  std::vector<double> switchTimes;
  std::string reportPath = "reports/validation-report.json";
  std::string snapshotPath = "reports/state-snapshot.json";
  std::string faultName = "none";
  FaultKind fault = FaultKind::None;
  int windowWidth = 1280;
  int windowHeight = 720;
};

CliOptions parseCommandLine(int argc, char** argv);
void printUsage(const char* executableName);

}  // namespace lcr

