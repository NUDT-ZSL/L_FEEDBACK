#pragma once

#include "math.h"
#include "scene.h"

#include <array>

namespace pbr {

struct EnvironmentSH {
    std::array<Vec3, 9> radiance{};
    std::array<Vec3, 9> irradiance{};
};

EnvironmentSH computeEnvironmentSH(const Environment& environment, int sampleCount = 8192);
Vec3 evaluateEnvironment(const Environment& environment, const Vec3& direction);

}  // namespace pbr
