#pragma once

#include "common.hpp"

#include <vector>

namespace cr {

inline constexpr std::uint64_t kSampleSeed = 0x51A7C0DE9E3779B9ULL;
inline constexpr int kMaterialTextureSize = 512;
inline constexpr int kDefaultViewportWidth = 640;
inline constexpr int kDefaultViewportHeight = 480;

void ensureGeneratedSamples(const fs::path& dataDirectory, bool forceRegenerate);
Camera loadCamera(const fs::path& dataDirectory);
std::vector<Light> loadDefaultLights(const fs::path& dataDirectory);

}  // namespace cr

