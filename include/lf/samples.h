#pragma once

#include "lf/config.h"

#include <filesystem>
#include <string>
#include <vector>

namespace lf {

constexpr int kSampleTextureSize = 256;
constexpr std::uint64_t kSampleSeed = 0x5A17C0DE9E3779B9ULL;

struct SampleGenerationResult {
  bool texturesGenerated = false;
  bool configGenerated = false;
  std::vector<std::filesystem::path> textureFiles;
};

[[nodiscard]] std::vector<std::string> expected_texture_files();
[[nodiscard]] LightingConfig make_default_config();

SampleGenerationResult generate_samples(const std::filesystem::path& assetRoot,
                                        const std::filesystem::path& configFile,
                                        bool overwrite);

SampleGenerationResult ensure_samples(const std::filesystem::path& assetRoot,
                                      const std::filesystem::path& configFile);

}  // namespace lf

