#include "lightcluster/animation.hpp"

#include <algorithm>
#include <stdexcept>

namespace lightcluster {

namespace {

float lerp(float from, float to, float amount) noexcept {
    return from + (to - from) * amount;
}

Vec3 lerp(Vec3 from, Vec3 to, float amount) noexcept {
    return {
        lerp(from.x, to.x, amount),
        lerp(from.y, to.y, amount),
        lerp(from.z, to.z, amount),
    };
}

}  // namespace

std::string_view toString(AnimationRejectCode code) noexcept {
    switch (code) {
        case AnimationRejectCode::None:
            return "NONE";
        case AnimationRejectCode::PoolFull:
            return "ANIM_POOL_FULL";
        case AnimationRejectCode::LightNotFound:
            return "ANIM_LIGHT_NOT_FOUND";
        case AnimationRejectCode::InvalidRequest:
            return "ANIM_INVALID_REQUEST";
    }
    return "ANIM_UNKNOWN";
}

AnimationPool::AnimationPool(std::size_t capacity) : slots_(capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("animation pool capacity must be positive");
    }
    freeSlots_.reserve(capacity);
    for (std::size_t i = capacity; i > 0; --i) {
        freeSlots_.push_back(i - 1);
    }
}

AnimationQueueResult AnimationPool::queue(
    SceneManager& scene,
    const AnimationRequest& request,
    std::uint32_t currentFrame) {
    if (freeSlots_.empty()) {
        return {false, AnimationRejectCode::PoolFull, 0, 0, 0};
    }
    if (request.durationFrames == 0 ||
        (!request.animatePosition && !request.animateIntensity)) {
        return {false, AnimationRejectCode::InvalidRequest, 0, 0, 0};
    }

    const auto light = scene.findLight(request.lightId);
    if (!light.has_value()) {
        return {false, AnimationRejectCode::LightNotFound, 0, 0, 0};
    }

    const std::size_t slotIndex = freeSlots_.back();
    freeSlots_.pop_back();

    Slot& slot = slots_[slotIndex];
    slot.active = true;
    slot.taskId = nextTaskId_++;
    slot.lightId = request.lightId;
    slot.startFrame = currentFrame;
    slot.durationFrames = request.durationFrames;
    slot.startPosition = light->position;
    slot.targetPosition = request.targetPosition;
    slot.startIntensity = light->intensity;
    slot.targetIntensity = request.targetIntensity;
    slot.animatePosition = request.animatePosition;
    slot.animateIntensity = request.animateIntensity;

    return {
        true,
        AnimationRejectCode::None,
        slot.taskId,
        slot.startFrame,
        slot.startFrame + slot.durationFrames,
    };
}

AnimationFrameResult AnimationPool::applyFrame(
    SceneManager& scene,
    std::uint32_t frame) {
    AnimationFrameResult result;

    for (std::size_t i = 0; i < slots_.size(); ++i) {
        Slot& slot = slots_[i];
        if (!slot.active || frame <= slot.startFrame) {
            continue;
        }

        const std::uint32_t elapsed = frame - slot.startFrame;
        const float amount = std::min(
            1.0F,
            static_cast<float>(elapsed) /
                static_cast<float>(slot.durationFrames));
        const Vec3 position =
            lerp(slot.startPosition, slot.targetPosition, amount);
        const float intensity =
            lerp(slot.startIntensity, slot.targetIntensity, amount);

        const bool updated = scene.updateLight(
            slot.lightId, [&](Light& light) {
                if (slot.animatePosition) {
                    light.position = position;
                }
                if (slot.animateIntensity) {
                    light.intensity = intensity;
                }
            });

        if (!updated) {
            ++result.cancelled;
        } else {
            ++result.applied;
        }

        if (!updated || elapsed >= slot.durationFrames) {
            slot.active = false;
            freeSlots_.push_back(i);
            ++result.completed;
        }
    }

    result.active = static_cast<std::uint32_t>(
        slots_.size() - freeSlots_.size());
    return result;
}

bool AnimationPool::isLightBusy(std::uint32_t lightId) const {
    return std::any_of(
        slots_.begin(), slots_.end(), [lightId](const Slot& slot) {
            return slot.active && slot.lightId == lightId;
        });
}

std::size_t AnimationPool::activeCount() const noexcept {
    return slots_.size() - freeSlots_.size();
}

std::size_t AnimationPool::capacity() const noexcept {
    return slots_.size();
}

}  // namespace lightcluster
