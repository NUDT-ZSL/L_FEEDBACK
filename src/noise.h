#pragma once

#include "math.h"

#include <cstdint>

namespace df {

uint32_t hash3D(int x, int y, int z, uint32_t seed = 0x9e3779b9u);
float valueNoise3D(const Vec3& p, uint32_t seed);
float fbm3D(const Vec3& p, int octaves, uint32_t seed);
float densityAt(const Vec3& world, float densityScale, float noiseScale);

} // namespace df

