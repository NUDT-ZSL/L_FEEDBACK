#pragma once

#include "terrain/terrain.hpp"

#include <cstdint>

namespace terrain {

uint64_t computeTerrainVersion(const TerrainParameters& parameters);
uint64_t fnv1aBytes(const void* data, std::size_t size, uint64_t seed = 1469598103934665603ull);

}  // namespace terrain

