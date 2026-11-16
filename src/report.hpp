#pragma once

#include "error.hpp"
#include "offline_renderer.hpp"
#include "scene.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace skel {

struct FrameRecord {
    std::uint32_t frame = 0;
    FrameStats stats;
    std::string depthFile;
    std::string colorFile;
    double elapsedMs = 0.0;
};

struct RenderReport {
    std::string assetFile;
    std::string mode = "offline";
    std::uint32_t assetFrameCount = 0;
    float framesPerSecond = 0.0F;
    CompositionOrder compositionOrder = CompositionOrder::TRS;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    Camera camera;
    std::size_t vertexCount = 0;
    std::size_t triangleCount = 0;
    std::uint32_t normalizedVertexCount = 0;
    std::vector<FrameRecord> frames;
    double elapsedMs = 0.0;

    std::uint64_t totalNanPositions() const;
    std::uint64_t totalNanNormals() const;
};

// Serialized with a stable, ordered schema. Timing values vary by run and are
// deliberately isolated from all deterministic fields.
std::string statisticsJson(const RenderReport& report);
bool writeStatisticsJson(const std::filesystem::path& path,
                         const RenderReport& report, Error& error);
bool parseStatisticsJson(const std::string& text, RenderReport& report,
                         Error& error);

} // namespace skel

