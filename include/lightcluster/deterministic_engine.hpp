#pragma once

#include "lightcluster/animation.hpp"
#include "lightcluster/buffer_sync.hpp"
#include "lightcluster/scene.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lightcluster {

struct RunConfig {
    std::uint32_t frames = 120;
    SyncMode syncMode = SyncMode::Immediate;
    std::uint32_t delayFrames = 0;
    std::size_t animationPoolCapacity = 8;
    std::uint64_t seed = 20261115ULL;
    std::optional<std::uint32_t> saveSnapshotFrame;
    std::optional<std::uint32_t> restoreSnapshotFrame;
    bool debug = false;
    bool injectDebugFault = false;
    bool headless = false;
};

enum class RunEventKind {
    LightAdded,
    LightRemoved,
    AnimationQueued,
    AnimationRejected,
    SnapshotSaved,
    SnapshotRestored,
};

struct RunEvent {
    RunEventKind kind = RunEventKind::LightAdded;
    std::uint32_t frame = 0;
    std::uint32_t lightId = 0;
    LightType lightType = LightType::Point;
    std::uint64_t taskId = 0;
    std::uint32_t durationFrames = 0;
    std::uint32_t startFrame = 0;
    std::uint32_t endFrame = 0;
    bool animatePosition = false;
    bool animateIntensity = false;
    Vec3 targetPosition{};
    float targetIntensity = 0.0F;
    std::uint32_t activeLights = 0;
    std::uint32_t activeTasks = 0;
    std::uint32_t poolCapacity = 0;
    std::uint32_t rejectedTotal = 0;
    std::string businessCode;
    std::uint64_t versionBefore = 0;
    std::uint64_t versionAfter = 0;
    std::uint64_t restoredSnapshotVersion = 0;
};

struct FrameTick {
    std::uint32_t frame = 0;
    SceneSnapshot frontend{};
    SyncOutcome sync{};
    AnimationFrameResult animations{};
    std::uint32_t animationActiveAfterQueue = 0;
    std::uint32_t animationRejectedThisFrame = 0;
    std::uint32_t animationRejectedTotal = 0;
    std::vector<RunEvent> events;
};

class DeterministicEngine {
public:
    explicit DeterministicEngine(RunConfig config);

    [[nodiscard]] FrameTick tick(std::uint32_t frame);
    [[nodiscard]] const RunConfig& config() const noexcept;
    [[nodiscard]] const SyncOutcome& initialSync() const noexcept;
    [[nodiscard]] const SceneManager& scene() const noexcept;
    [[nodiscard]] const LightBufferSynchronizer& synchronizer() const noexcept;

private:
    class SplitMix64 {
    public:
        explicit SplitMix64(std::uint64_t seed);
        [[nodiscard]] std::uint64_t next() noexcept;
        [[nodiscard]] std::uint32_t bounded(std::uint32_t upperExclusive);
        [[nodiscard]] bool chancePercent(std::uint32_t percent);
        [[nodiscard]] float unitFloat();
        [[nodiscard]] float range(float minimum, float maximum);

    private:
        std::uint64_t state_;
    };

    [[nodiscard]] static std::vector<Light> makeInitialLights();
    [[nodiscard]] Light makeRandomLight();
    void queueRandomAnimation(
        std::uint32_t frame,
        const SceneSnapshot& snapshot,
        FrameTick& tick);

    RunConfig config_;
    SceneManager scene_;
    AnimationPool animations_;
    LightBufferSynchronizer synchronizer_;
    SplitMix64 random_;
    std::optional<SceneSnapshot> savedSnapshot_;
    SyncOutcome initialSync_{};
    std::uint32_t rejectedTotal_ = 0;
    std::uint32_t mutationEndFrame_ = 0;
};

}  // namespace lightcluster
