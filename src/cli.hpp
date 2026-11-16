#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "errors.hpp"

namespace orv {

enum class RenderPass { Geometry, Lighting, Full };

struct CliOptions {
  std::filesystem::path scenePath;
  std::filesystem::path outputPrefix;
  int width = 640;
  int height = 360;
  int shadowSize = 1024;
  float shadowBias = 0.002f;
  RenderPass pass = RenderPass::Full;
  bool showHelp = false;
};

std::string renderPassName(RenderPass pass);
std::string usageText();

bool parseCli(int argc, char** argv, CliOptions& options, AppError& error);
bool validateShadowSize(int size);

}  // namespace orv
