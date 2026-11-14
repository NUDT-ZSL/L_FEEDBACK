#pragma once

#include "config.h"
#include "lights.h"
#include "scene.h"
#include "sorting.h"

#include <cstdint>
#include <string>
#include <vector>

namespace df {

struct GBuffer {
    int width = 0;
    int height = 0;
    std::vector<Vec4> gPosition;
    std::vector<Vec4> gNormal;
    std::vector<Vec4> gAlbedo;
    std::vector<Vec4> gMaterial;
};

struct FrameStats {
    std::string backend;
    int fogSteps = 0;
    int activeLights = 0;
    int transparentDrawn = 0;
    int transparentDepthWrites = 0;
    std::uint64_t lightVersion = 0;
    std::uint64_t lightingUploads = 0;
    std::uint64_t fogUploads = 0;
    double cpuMilliseconds = 0.0;
    std::string gpuRenderer;
    std::string gpuVersion;
};

struct FrameOutput {
    int width = 0;
    int height = 0;
    Camera camera;
    std::vector<std::uint8_t> colorRgb;
    std::vector<float> linearDepth;
    GBuffer gbuffer;
    SortResult transparent;
    FrameStats stats;
};

class CpuRenderer {
public:
    FrameOutput render(const Scene& scene, const RenderConfig& config, bool offline,
                       bool collectGBuffer = true);

    const LightSnapshot& lightSnapshot() const { return lightSnapshot_; }
    const LightUploadStats& uploadStats() const { return uploadStats_; }

private:
    LightSnapshot lightSnapshot_;
    LightUploadStats uploadStats_;
};

} // namespace df

