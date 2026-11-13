#pragma once

#include "skinbake/snapshot.hpp"
#include "skinbake/state_machine.hpp"

#include <cstdint>
#include <functional>
#include <string>

namespace skinbake {

struct BakeOptions {
    std::string outputPath;
    std::uint32_t version = 1;
    std::uint32_t totalFrames = 180;
    std::uint32_t requestedWorkers = 1;
    std::uint32_t blockFrames = 8;
    bool resume = true;
    std::string eventLogPath;
    std::optional<std::uint32_t> crashAfterCommitFrame;
    std::function<void(const FrameRecord&)> onFrameCommitted;
};

struct BakeReport {
    std::uint32_t version = 0;
    std::string clip;
    std::uint32_t frames = 0;
    std::uint32_t fps = 0;
    std::uint64_t fingerprint = 0;
    bool resumed = false;
    bool discardedProgress = false;
    std::string discardReason;
    std::uint32_t resumeFrame = 0;
    std::uint32_t firstNewFrame = 0;
    std::uint32_t newFramesCommitted = 0;
    std::uint32_t requestedWorkers = 1;
    std::uint32_t usedWorkers = 0;
    std::uint32_t logicalCores = 1;
    std::string outputPath;
};

std::string progress_path_for(const std::string& outputPath);

BakeReport bake(const TimelineEvaluator& timeline, const BakeOptions& options);

Snapshot snapshot_from_progress(const Snapshot& progress, bool complete);

}

