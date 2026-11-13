#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dungeon {

struct ShadowCacheEntry {
    uint32_t lightId = 0;
    int32_t slot = -1;
    int32_t geometryVersion = 0;
    uint64_t lightHash = 0;
    uint64_t sceneHash = 0;
    std::vector<double> depth;
};

struct ShadowCache {
    int columns = 0;
    int rows = 0;
    int tileSize = 0;
    std::vector<ShadowCacheEntry> entries;
};

bool loadShadowCache(
    const std::string& path,
    int expectedColumns,
    int expectedRows,
    int expectedTileSize,
    ShadowCache& cache);

bool saveShadowCache(const std::string& path, const ShadowCache& cache);

}  // namespace dungeon
