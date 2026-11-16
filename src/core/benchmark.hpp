#pragma once

#include "core/adaptive.hpp"
#include "core/animation.hpp"
#include "core/command_recorder.hpp"
#include "core/culling.hpp"
#include "core/scene.hpp"
#include "core/shadows.hpp"

#include <cstdint>
#include <filesystem>

namespace uid166 {

struct DataPaths {
    std::filesystem::path scene;
    std::filesystem::path animation;
};

struct PreparedData {
    SceneDesc scene;
    CharacterAsset character;
    DataPaths paths;
    uint64_t sceneHash = 0;
    uint64_t animationHash = 0;
};

struct DataOptions {
    std::filesystem::path dataDir = ".";
    uint32_t objectCount = 5000;
    uint64_t seed = 166;
    bool reuseExisting = false;
};

struct BenchmarkOptions {
    DataOptions data{};
    std::filesystem::path outputPath = "artifacts/uid166_benchmark.json";
    uint32_t frames = 120;
    float frameSeconds = 1.0f / 60.0f;
    RecordingConfig recording{};
    CascadeConfig cascades{};
    LocalShadowConfig localShadows{};
    AdaptiveConfig adaptive{};
    CullOptions culling{};
    bool simulateGpuLoad = true;
    uint32_t simulatedLoadStart = 20;
    uint32_t simulatedLoadEnd = 46;
    bool writeCullingSets = true;
};

struct BenchmarkSummary {
    uint64_t frameStatsHash = kFnvOffset;
    uint64_t visibleSum = 0;
    uint64_t occludedSum = 0;
    uint64_t shadowUpdates = 0;
    uint64_t localShadowUpdates = 0;
    uint64_t csmUpdates = 0;
    uint64_t recordingSumUs = 0;
    uint64_t recordingMaxUs = 0;
    uint32_t budgetHitFrames = 0;
    uint32_t minVisible = 0xffffffffu;
    uint32_t maxVisible = 0;
    uint32_t minQualityLevel = 0;
    uint32_t maxQualityLevel = 0;
    uint32_t longestStableMatrixRun = 0;
    uint32_t stableWindowStart = 0;
    uint32_t stableWindowEnd = 0;
    bool defaultQualityRestored = true;
    bool validationPassed = true;
};

[[nodiscard]] PreparedData prepareSceneData(const DataOptions& options);
[[nodiscard]] PreparedData loadSceneData(const DataOptions& options);
[[nodiscard]] BenchmarkSummary runBenchmark(const BenchmarkOptions& options,
                                            PreparedData data);

} // namespace uid166

