#pragma once

#include "config.h"
#include "renderer_cpu.h"

#include <cstdint>
#include <string>

namespace df {

struct OfflineArtifacts {
    std::string outputDirectory;
    std::string colorPng;
    std::string depthPfm;
    std::string transparentJson;
    std::string metadataJson;
    std::uint64_t colorHash = 0;
    std::uint64_t depthHash = 0;
    FrameOutput frame;
};

OfflineArtifacts renderOffline(const RenderConfig& config,
                               const std::string& outputDirectory);
OfflineArtifacts writeFrameArtifacts(FrameOutput frame, const Scene& scene,
                                     const RenderConfig& config,
                                     const std::string& outputDirectory);

} // namespace df

