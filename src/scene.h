#pragma once

#include "math_types.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace subdiv {

struct SceneSettings {
    std::uint64_t seed = 20251029ULL;
    Vec3 diffuse{0.20f, 0.52f, 0.82f};
    float roughness = 0.36f;
    std::vector<std::pair<int, float>> creaseOverrides;
    int maxSubdivision = 4;
    int initialSubdivision = 1;
    CameraSettings camera;
};

struct SceneLoadResult {
    SceneSettings settings;
    std::vector<std::string> warnings;
};

SceneLoadResult loadSceneFile(const std::string& path);

}  // namespace subdiv

