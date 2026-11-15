#pragma once

#include "image.h"
#include "scene.h"

#include <array>
#include <string>
#include <vector>

namespace pbr {

constexpr int kHighlightHistogramBins = 16;

struct MaterialStats {
    int id = -1;
    std::string name;
    long long pixelCount = 0;
    Vec3 meanColor{};
    std::array<float, kHighlightHistogramBins> highlightHistogram{};
    Vec3 specularDirection{0.0f, 1.0f, 0.0f};
};

struct RenderStatistics {
    std::vector<MaterialStats> materials;
};

RenderStatistics computeStatistics(const Scene& scene, const Image& image);

}  // namespace pbr
