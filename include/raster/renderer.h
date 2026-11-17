#pragma once

#include "raster/scene.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace raster {

struct RenderStats {
    std::uint64_t totalTriangles = 0;
    std::uint64_t culledMeshes = 0;
    std::uint64_t rasterizedTriangles = 0;
    std::uint64_t totalFragments = 0;
    std::uint64_t depthTestFailures = 0;
};

struct RenderImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> bgrTopDown;
};

using WarningCallback = std::function<void(const std::string&)>;

RenderImage renderScene(const Scene& scene, int width, int height,
                        const WarningCallback& warning,
                        RenderStats& stats);

}  // namespace raster

