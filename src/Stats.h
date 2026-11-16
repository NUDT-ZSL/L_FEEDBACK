#pragma once

#include "Common.h"
#include "Renderer.h"

struct FrameStats {
    uint32_t frame = 0;
    double animationTimeSeconds = 0.0;
    uint32_t drawCalls = 0;
    std::array<uint32_t, 3> shadowMapSizes{};
    std::array<double, 3> cascadeSplitRatios{};
    std::array<double, 4> cascadeSplitDistances{};
    std::array<CascadeData, 3> cascades{};
    double slopeOffset = 0.0;
    double constOffset = 0.0;
    uint64_t deviceLossRecoveries = 0;
};

struct AggregateStats {
    uint32_t frameCount = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    std::array<double, 3> cascadeSplitRatios{};
    std::array<double, 4> cascadeSplitDistances{};
    bool hasDeviceLossInjection = false;
    uint32_t deviceLossInjectionFrame = 0;
    uint64_t deviceLossRecoveries = 0;
    double slopeOffset = 0.0;
    double constOffset = 0.0;
    std::vector<FrameStats> frames;
};

std::string FrameBaseName(uint32_t frame, uint32_t totalFrames);
std::string BuildFrameJson(const FrameStats& stats);
std::string BuildAggregateJson(const AggregateStats& stats);
void WriteTextFile(const std::filesystem::path& path, const std::string& contents);

