#pragma once

#include "lightcluster/scene.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace lightcluster {

enum class AnimationRejectCode {
    None,
    PoolFull,
    LightNotFound,
    InvalidRequest,
};

[[nodiscard]] std::string_view toString(AnimationRejectCode code) noexcept;

struct AnimationRequest {
    std::uint32_t lightId = 0;
    std::uint32_t durationFrames = 1;
    Vec3 targetPosition{};
    float targetIntensity = 0.0F;
    bool animatePosition = true;
    bool animateIntensity = true;
};

struct AnimationQueueResult {
    bool accepted = false;
    AnimationRejectCode code = AnimationRejectCode::None;
    std::uint64_t taskId = 0;
    std::uint32_t startFrame = 0;
    std::uint32_t endFrame = 0;
};

struct AnimationFrameResult {
    std::uint32_t applied = 0;
    std::uint32_t completed = 0;
    std::uint32_t cancelled = 0;
    std::uint32_t active = 0;
};

class AnimationPool {
public:
    explicit AnimationPool(std::size_t capacity);

    [[nodiscard]] AnimationQueueResult queue(
        SceneManager& scene,
        const AnimationRequest& request,
        std::uint32_t currentFrame);

    [[nodiscard]] AnimationFrameResult applyFrame(
        SceneManager& scene,
        std::uint32_t frame);

    [[nodiscard]] bool isLightBusy(std::uint32_t lightId) const;
    [[nodiscard]] std::size_t activeCount() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;

private:
    struct Slot {
        bool active = false;
        std::uint64_t taskId = 0;
        std::uint32_t lightId = 0;
        std::uint32_t startFrame = 0;
        std::uint32_t durationFrames = 0;
        Vec3 startPosition{};
        Vec3 targetPosition{};
        float startIntensity = 0.0F;
        float targetIntensity = 0.0F;
        bool animatePosition = false;
        bool animateIntensity = false;
    };

    std::vector<Slot> slots_;
    std::vector<std::size_t> freeSlots_;
    std::uint64_t nextTaskId_ = 1;
};

}  // namespace lightcluster
