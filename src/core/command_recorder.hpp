#pragma once

#include "core/math.hpp"
#include "core/scene.hpp"

#include <cstdint>
#include <vector>

namespace uid166 {

struct RenderCommand {
    uint32_t objectId = 0;
    uint32_t material = 0;
    Mat4 world = Mat4::identity();
};

struct RenderBatch {
    uint32_t material = 0;
    std::vector<RenderCommand> commands;
};

struct RecordingConfig {
    uint32_t budgetMicros = 1800;
    uint32_t workerCount = 4;
};

struct WorkerLoad {
    uint32_t workerId = 0;
    uint32_t commands = 0;
    uint64_t elapsedMicros = 0;
};

struct RecordingResult {
    std::vector<RenderCommand> commands;
    std::vector<RenderBatch> batches;
    std::vector<WorkerLoad> workers;
    uint64_t elapsedMicros = 0;
    uint64_t parallelStageMicros = 0;
    uint32_t fallbackSerialCommands = 0;
    bool budgetHit = false;
    bool budgetExceeded = false;
    uint32_t mergedDrawCalls = 0;
};

class CommandRecorder {
public:
    [[nodiscard]] RecordingResult record(const std::vector<uint32_t>& visibleObjectIds,
                                         const std::vector<SceneObject>& objects,
                                         const RecordingConfig& config) const;
};

} // namespace uid166

