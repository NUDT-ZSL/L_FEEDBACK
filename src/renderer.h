#pragma once

#include "error.h"
#include "scene.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace softraster {

struct RenderStats {
    int width = 0;
    int height = 0;
    int viewportX = 0;
    int viewportY = 0;
    int viewportWidth = 0;
    int viewportHeight = 0;
    int shadowMapSize = 0;
    bool shadowEnabled = false;
    std::uint64_t rasterizedTriangles = 0;
    std::uint64_t clippedTriangles = 0;
    std::uint64_t shadowSampleCount = 0;
    std::uint64_t transparentSortSwaps = 0;
    std::uint64_t peakMemoryEstimatedBytes = 0;
    std::uint64_t opaqueFragments = 0;
    std::uint64_t transparentFragments = 0;
    std::uint64_t alphaBlendedFragments = 0;
    std::uint64_t transparentDepthWrites = 0;
    std::uint64_t transparentTriangles = 0;
    std::uint64_t ppmBytes = 0;
};

struct RenderedFrame {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> ppm;
    RenderStats stats;
};

struct RenderResult {
    std::optional<RenderedFrame> frame;
    std::optional<RenderError> error;
};

[[nodiscard]] RenderResult renderScene(const Scene& scene);

}  // namespace softraster
