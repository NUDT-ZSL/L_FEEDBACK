#include "lightcluster/deterministic_engine.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace lightcluster {

namespace {

constexpr std::uint32_t kMinimumActiveLights = 64;
constexpr std::uint32_t kMinimumPointLights = 64;
constexpr std::uint32_t kMinimumDirectionalLights = 4;
constexpr std::uint32_t kMaximumActiveLights = 160;
constexpr std::uint32_t kInitialPointLights = 64;
constexpr std::uint32_t kInitialDirectionalLights = 8;

float colorChannel(std::uint32_t value, std::uint32_t phase) {
    const float t = static_cast<float>((value * 37U + phase * 19U) % 100U) /
                    100.0F;
    return 0.22F + 0.78F * t;
}

}  // namespace

DeterministicEngine::SplitMix64::SplitMix64(std::uint64_t seed)
    : state_(seed) {}

std::uint64_t DeterministicEngine::SplitMix64::next() noexcept {
    std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

std::uint32_t DeterministicEngine::SplitMix64::bounded(
    std::uint32_t upperExclusive) {
    if (upperExclusive == 0) {
        return 0;
    }
    return static_cast<std::uint32_t>(next() % upperExclusive);
}

bool DeterministicEngine::SplitMix64::chancePercent(std::uint32_t percent) {
    return bounded(100) < percent;
}

float DeterministicEngine::SplitMix64::unitFloat() {
    return static_cast<float>(next() >> 40U) / 16777216.0F;
}

float DeterministicEngine::SplitMix64::range(
    float minimum,
    float maximum) {
    return minimum + (maximum - minimum) * unitFloat();
}

DeterministicEngine::DeterministicEngine(RunConfig config)
    : config_(std::move(config)),
      scene_(makeInitialLights()),
      animations_(config_.animationPoolCapacity),
      synchronizer_(config_.syncMode, config_.delayFrames),
      random_(config_.seed) {
    if (config_.frames == 0) {
        throw std::invalid_argument("frames must be positive");
    }
    if (config_.delayFrames > 4096) {
        throw std::invalid_argument("delay must not exceed 4096 frames");
    }
    if (config_.restoreSnapshotFrame.has_value() &&
        !config_.saveSnapshotFrame.has_value()) {
        throw std::invalid_argument(
            "restore-snapshot-frame requires save-snapshot-frame");
    }
    if (config_.saveSnapshotFrame.has_value() &&
        *config_.saveSnapshotFrame >= config_.frames) {
        throw std::invalid_argument(
            "save-snapshot-frame must be less than frames");
    }
    if (config_.restoreSnapshotFrame.has_value() &&
        *config_.restoreSnapshotFrame >= config_.frames) {
        throw std::invalid_argument(
            "restore-snapshot-frame must be less than frames");
    }
    if (config_.saveSnapshotFrame.has_value() &&
        config_.restoreSnapshotFrame.has_value() &&
        *config_.restoreSnapshotFrame <= *config_.saveSnapshotFrame) {
        throw std::invalid_argument(
            "restore-snapshot-frame must be greater than save-snapshot-frame");
    }

    const std::uint32_t syncDelay =
        config_.syncMode == SyncMode::Deferred ? config_.delayFrames : 0;
    const std::uint32_t quietTail =
        std::max(std::uint32_t{16}, std::uint32_t{14} + syncDelay + 2U);
    mutationEndFrame_ =
        config_.frames > quietTail + 1U ? config_.frames - quietTail
                                        : config_.frames;
    initialSync_ = synchronizer_.initialize(scene_.snapshot(), 0);
}

FrameTick DeterministicEngine::tick(std::uint32_t frame) {
    FrameTick tick;
    tick.frame = frame;

    if (config_.restoreSnapshotFrame.has_value() &&
        frame == *config_.restoreSnapshotFrame && savedSnapshot_.has_value()) {
        RunEvent event;
        event.kind = RunEventKind::SnapshotRestored;
        event.frame = frame;
        event.versionBefore = scene_.version();
        event.restoredSnapshotVersion = savedSnapshot_->version;
        event.activeLights =
            static_cast<std::uint32_t>(savedSnapshot_->lights.size());
        event.versionAfter = scene_.restore(*savedSnapshot_);
        tick.events.push_back(event);
    }

    tick.animations = animations_.applyFrame(scene_, frame);

    if (frame > 0 && frame < mutationEndFrame_) {
        std::uint64_t count = scene_.lightCount();
        const bool capacitySpike = frame >= 20 && frame < 30;

        const std::uint32_t addAttempts = capacitySpike ? 8U : 1U;
        for (std::uint32_t attempt = 0; attempt < addAttempts &&
             count < kMaximumActiveLights;
             ++attempt) {
            if (!capacitySpike && !random_.chancePercent(34)) {
                break;
            }
            const Light added = makeRandomLight();
            const std::uint32_t id = scene_.addLight(added);
            ++count;
            RunEvent event;
            event.kind = RunEventKind::LightAdded;
            event.frame = frame;
            event.lightId = id;
            event.lightType = added.type;
            event.activeLights =
                static_cast<std::uint32_t>(scene_.lightCount());
            event.versionAfter = scene_.version();
            event.versionBefore = event.versionAfter - 1;
            tick.events.push_back(event);
        }

        const SceneSnapshot deletionSnapshot = scene_.snapshot();
        const LightCounts deletionCounts = deletionSnapshot.counts();
        const bool mayDeletePoint =
            deletionCounts.point > kMinimumPointLights;
        const bool mayDeleteDirectional =
            deletionCounts.directional > kMinimumDirectionalLights;
        if (!capacitySpike && count > kMinimumActiveLights &&
            (mayDeletePoint || mayDeleteDirectional) &&
            random_.chancePercent(30)) {
            std::vector<std::uint32_t> candidates;
            candidates.reserve(deletionSnapshot.lights.size());
            for (const Light& light : deletionSnapshot.lights) {
                const bool typeMayDelete =
                    light.type == LightType::Point ? mayDeletePoint
                                                   : mayDeleteDirectional;
                if (!typeMayDelete) {
                    continue;
                }
                if (!animations_.isLightBusy(light.id)) {
                    candidates.push_back(light.id);
                }
            }
            if (!candidates.empty()) {
                const std::uint32_t id = candidates[
                    random_.bounded(
                        static_cast<std::uint32_t>(candidates.size()))];
                if (scene_.removeLight(id)) {
                    RunEvent event;
                    event.kind = RunEventKind::LightRemoved;
                    event.frame = frame;
                    event.lightId = id;
                    event.activeLights =
                        static_cast<std::uint32_t>(scene_.lightCount());
                    event.versionAfter = scene_.version();
                    event.versionBefore = event.versionAfter - 1;
                    tick.events.push_back(event);
                }
            }
        }
    }

    if (frame < mutationEndFrame_) {
        const SceneSnapshot beforeQueue = scene_.snapshot();
        queueRandomAnimation(frame, beforeQueue, tick);
    }

    if (config_.saveSnapshotFrame.has_value() &&
        frame == *config_.saveSnapshotFrame) {
        savedSnapshot_ = scene_.saveSnapshot();
        RunEvent event;
        event.kind = RunEventKind::SnapshotSaved;
        event.frame = frame;
        event.versionAfter = savedSnapshot_->version;
        event.restoredSnapshotVersion = savedSnapshot_->version;
        event.activeLights =
            static_cast<std::uint32_t>(savedSnapshot_->lights.size());
        tick.events.push_back(event);
    }

    tick.frontend = scene_.snapshot();
    tick.sync = synchronizer_.sync(tick.frontend, frame);
    tick.animationActiveAfterQueue =
        static_cast<std::uint32_t>(animations_.activeCount());
    tick.animationRejectedThisFrame = static_cast<std::uint32_t>(
        std::count_if(
            tick.events.begin(), tick.events.end(), [](const RunEvent& event) {
                return event.kind == RunEventKind::AnimationRejected;
            }));
    tick.animationRejectedTotal = rejectedTotal_;
    return tick;
}

const RunConfig& DeterministicEngine::config() const noexcept {
    return config_;
}

const SyncOutcome& DeterministicEngine::initialSync() const noexcept {
    return initialSync_;
}

const SceneManager& DeterministicEngine::scene() const noexcept {
    return scene_;
}

const LightBufferSynchronizer& DeterministicEngine::synchronizer() const noexcept {
    return synchronizer_;
}

std::vector<Light> DeterministicEngine::makeInitialLights() {
    std::vector<Light> lights;
    lights.reserve(kInitialPointLights + kInitialDirectionalLights);

    for (std::uint32_t i = 0; i < kInitialPointLights; ++i) {
        const std::uint32_t x = i % 8U;
        const std::uint32_t z = i / 8U;
        Light light;
        light.type = LightType::Point;
        light.position = {
            (static_cast<float>(x) - 3.5F) * 0.9F,
            0.65F + static_cast<float>((i % 4U) * 0.4F),
            (static_cast<float>(z) - 3.5F) * 0.9F,
        };
        light.color = {
            colorChannel(i, 0),
            colorChannel(i, 1),
            colorChannel(i, 2),
        };
        light.intensity = 1.0F + static_cast<float>(i % 5U) * 0.16F;
        light.range = 4.2F + static_cast<float>(i % 3U) * 0.7F;
        lights.push_back(light);
    }

    for (std::uint32_t i = 0; i < kInitialDirectionalLights; ++i) {
        const float angle =
            6.28318530718F * static_cast<float>(i) /
            static_cast<float>(kInitialDirectionalLights);
        Light light;
        light.type = LightType::Directional;
        light.direction = normalizeOr(
            {
                std::cos(angle) * 0.55F,
                -0.75F,
                std::sin(angle) * 0.55F,
            },
            {0.0F, -1.0F, 0.0F});
        light.color = {
            colorChannel(i, 4),
            colorChannel(i, 5),
            colorChannel(i, 6),
        };
        light.intensity = 0.12F + static_cast<float>(i) * 0.018F;
        lights.push_back(light);
    }
    return lights;
}

Light DeterministicEngine::makeRandomLight() {
    Light light;
    light.type =
        random_.chancePercent(82) ? LightType::Point : LightType::Directional;
    if (light.type == LightType::Point) {
        light.position = {
            random_.range(-4.1F, 4.1F),
            random_.range(0.25F, 2.4F),
            random_.range(-4.1F, 4.1F),
        };
        light.range = random_.range(3.0F, 7.0F);
    } else {
        light.direction = normalizeOr(
            {
                random_.range(-1.0F, 1.0F),
                random_.range(-1.2F, -0.35F),
                random_.range(-1.0F, 1.0F),
            },
            {0.0F, -1.0F, 0.0F});
    }
    light.color = {
        random_.range(0.2F, 1.0F),
        random_.range(0.2F, 1.0F),
        random_.range(0.2F, 1.0F),
    };
    light.intensity =
        light.type == LightType::Point ? random_.range(0.6F, 3.2F)
                                       : random_.range(0.05F, 0.28F);
    return light;
}

void DeterministicEngine::queueRandomAnimation(
    std::uint32_t frame,
    const SceneSnapshot& snapshot,
    FrameTick& tick) {
    std::vector<std::uint32_t> candidates;
    candidates.reserve(snapshot.lights.size());
    for (const Light& light : snapshot.lights) {
        if (!animations_.isLightBusy(light.id)) {
            candidates.push_back(light.id);
        }
    }
    if (candidates.empty()) {
        return;
    }

    const std::uint32_t lightId = candidates[
        random_.bounded(static_cast<std::uint32_t>(candidates.size()))];
    const auto selected = scene_.findLight(lightId);
    if (!selected.has_value()) {
        return;
    }

    const std::uint32_t mode = random_.bounded(3);
    AnimationRequest request;
    request.lightId = lightId;
    request.durationFrames = 10U + random_.bounded(5);
    request.animatePosition =
        selected->type == LightType::Point && mode != 1;
    request.animateIntensity = selected->type == LightType::Directional ||
                               mode != 0;
    request.targetPosition = {
        random_.range(-3.8F, 3.8F),
        random_.range(0.2F, 2.2F),
        random_.range(-3.8F, 3.8F),
    };
    request.targetIntensity =
        selected->type == LightType::Point ? random_.range(0.5F, 3.8F)
                                           : random_.range(0.06F, 0.32F);

    const AnimationQueueResult queued =
        animations_.queue(scene_, request, frame);
    RunEvent event;
    event.frame = frame;
    event.lightId = lightId;
    event.poolCapacity = static_cast<std::uint32_t>(animations_.capacity());
    event.activeTasks =
        static_cast<std::uint32_t>(animations_.activeCount());
    event.animatePosition = request.animatePosition;
    event.animateIntensity = request.animateIntensity;
    event.targetPosition = request.targetPosition;
    event.targetIntensity = request.targetIntensity;
    event.versionAfter = scene_.version();
    event.versionBefore = event.versionAfter;

    if (queued.accepted) {
        event.kind = RunEventKind::AnimationQueued;
        event.taskId = queued.taskId;
        event.durationFrames = request.durationFrames;
        event.startFrame = queued.startFrame;
        event.endFrame = queued.endFrame;
    } else {
        ++rejectedTotal_;
        event.kind = RunEventKind::AnimationRejected;
        event.businessCode = std::string(toString(queued.code));
        event.rejectedTotal = rejectedTotal_;
    }
    tick.events.push_back(std::move(event));
}

}  // namespace lightcluster
