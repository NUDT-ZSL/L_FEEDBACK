#pragma once

#include "scene.hpp"
#include "shadow_cache.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dungeon {

struct RenderSettings {
    int width = 160;
    int height = 112;
    Camera camera;
    int geometryVersion = 0;
    double shadowBias = 0.015;
    int atlasColumns = 2;
    int atlasRows = 2;
    int shadowTileSize = 64;
    std::string cachePath;
};

struct BusinessError {
    std::string code;
    std::string message;
    int lightId = -1;
};

struct TileReport {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    std::vector<uint32_t> lightIds;
    uint64_t luminanceChecksum = 0;
    double averageLuminance = 0.0;
};

struct LightReport {
    uint32_t id = 0;
    std::string type;
    bool castsShadow = false;
    int tileIndex = -1;
    int atlasX = -1;
    int atlasY = -1;
    std::string action = "none";
    uint64_t transformHash = 0;
    uint64_t shadowTapAttempts = 0;
    uint64_t shadowTapSuccesses = 0;
    uint64_t shadowTapShadowed = 0;
    double sampleSuccessRate = 0.0;
    double shadowedRate = 0.0;
    uint64_t litPixels = 0;
};

struct RenderReport {
    int width = 0;
    int height = 0;
    int tileColumns = 0;
    int tileRows = 0;
    int atlasColumns = 0;
    int atlasRows = 0;
    int shadowTileSize = 0;
    int atlasAllocated = 0;
    int tileHits = 0;
    int tileRebuilds = 0;
    int degradedLights = 0;
    int ambientOnlyTiles = 0;
    bool cacheRequested = false;
    bool cacheLoaded = false;
    bool cacheSaved = false;
    uint64_t sceneHash = 0;
    uint64_t pixelLuminanceChecksum = 0;
    uint64_t shadowComparisons = 0;
    uint64_t gbufferCoveredPixels = 0;
    std::vector<uint32_t> materialPixelCounts;
    std::vector<TileReport> tiles;
    std::vector<LightReport> lights;
    std::vector<BusinessError> businessErrors;
};

RenderReport renderDeferred(const Scene& scene, const RenderSettings& settings);

}  // namespace dungeon
