#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "errors.hpp"

namespace orv {

struct Image8 {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> rgb;
};

struct RenderStats {
  int width = 0;
  int height = 0;
  double renderTimeMs = 0.0;
  std::uint64_t visiblePixels = 0;
  double averageLuminance = 0.0;
  double maximumLuminance = 0.0;
  bool averageLuminanceClamped = false;
  bool maximumLuminanceClamped = false;
  std::string shadowMethod;
  int lightCount = 0;
  std::string pass;
};

std::filesystem::path makePpmPath(const std::filesystem::path& prefix,
                                  const std::string& passName);
std::filesystem::path makeStatsPath(const std::filesystem::path& prefix);

std::string makeStatsJson(const RenderStats& stats);
std::string makeFinalStateJson(bool ok, const std::string& state, const std::string& code,
                               int exitCode, const std::string& pass,
                               const std::filesystem::path& outputPath,
                               const std::filesystem::path& statsPath);
std::string makeRecoveryJson(std::size_t removedFiles);

void emitErrorJson(const AppError& error);

bool writePpmAtomically(const std::filesystem::path& path, const Image8& image,
                        AppError& error);
bool writeStatsAtomically(const std::filesystem::path& path, const RenderStats& stats,
                          AppError& error);

}  // namespace orv
