#pragma once

#include "model.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace rigskin {

enum class RunStatus {
    Complete,
    Interrupted,
    Failed,
};

struct RunOptions {
    PipelineConfig config;
    int threads = 1;
    std::filesystem::path snapshotDirectory;
    int commitDelayMilliseconds = 0;
};

struct RunResult {
    RunStatus status = RunStatus::Failed;
    int exitCode = 2;
    std::uint32_t framesSuccess = 0;
    std::uint32_t framesReused = 0;
    std::uint32_t framesRecomputed = 0;
    std::vector<std::string> errorCodes;
    std::string message;
};

void requestInterrupt() noexcept;
bool interruptionRequested() noexcept;

RunResult runPipeline(const RunOptions& options);

}  // namespace rigskin

