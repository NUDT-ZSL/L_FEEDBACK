#pragma once

#include "lightcluster/types.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string_view>
#include <vector>

namespace lightcluster {

enum class SyncMode {
    Immediate,
    Deferred,
};

[[nodiscard]] std::string_view toString(SyncMode mode) noexcept;

constexpr std::size_t kGpuHeaderBytes = 32;
constexpr std::size_t kGpuLightBytes = 48;
constexpr std::uint32_t kMinimumBufferCapacity = 64;

struct GpuLightRecord {
    float positionType[4]{};
    float directionRange[4]{};
    float colorIntensity[4]{};
};

static_assert(sizeof(GpuLightRecord) == kGpuLightBytes);

struct GpuBufferImage {
    std::uint32_t capacityLights = 0;
    std::uint32_t usedBytes = 0;
    std::uint32_t lightCount = 0;
    std::uint32_t pointCount = 0;
    std::uint32_t directionalCount = 0;
    std::uint64_t version = 0;
    std::vector<std::uint8_t> bytes;
};

struct SyncOutcome {
    bool uploaded = false;
    bool grew = false;
    std::string_view reason = "not_initialized";
    std::uint64_t frontendVersion = 0;
    std::uint64_t gpuVersion = 0;
    std::uint32_t gpuLightCount = 0;
    std::uint32_t gpuPointCount = 0;
    std::uint32_t gpuDirectionalCount = 0;
    std::uint32_t capacityLights = 0;
    std::uint32_t usedBytes = 0;
    std::uint32_t pendingSnapshots = 0;
    std::uint32_t sourceFrame = 0;
    std::uint32_t snapshotAgeFrames = 0;
};

struct BufferValidation {
    bool ok = true;
    std::string_view code = "OK";
    std::uint32_t expectedLights = 0;
    std::uint32_t actualLights = 0;
    std::uint32_t expectedPoint = 0;
    std::uint32_t actualPoint = 0;
    std::uint32_t expectedDirectional = 0;
    std::uint32_t actualDirectional = 0;
    std::uint64_t expectedVersion = 0;
    std::uint64_t actualVersion = 0;
};

class LightBufferSynchronizer {
public:
    LightBufferSynchronizer(SyncMode mode, std::uint32_t maxDelayFrames);

    [[nodiscard]] SyncOutcome initialize(
        const SceneSnapshot& snapshot,
        std::uint32_t sourceFrame);

    [[nodiscard]] SyncOutcome sync(
        const SceneSnapshot& snapshot,
        std::uint32_t currentFrame);

    [[nodiscard]] const GpuBufferImage& residentImage() const noexcept;
    [[nodiscard]] const SceneSnapshot& residentSnapshot() const noexcept;
    [[nodiscard]] std::uint64_t residentVersion() const noexcept;
    [[nodiscard]] std::size_t pendingCount() const noexcept;
    [[nodiscard]] SyncMode mode() const noexcept;
    [[nodiscard]] std::uint32_t maxDelayFrames() const noexcept;

private:
    struct PendingSnapshot {
        SceneSnapshot snapshot;
        std::uint32_t sourceFrame = 0;
        std::uint64_t readyFrame = 0;
    };

    [[nodiscard]] SyncOutcome applySnapshot(
        const SceneSnapshot& snapshot,
        std::uint32_t sourceFrame,
        std::uint32_t currentFrame,
        std::string_view reason);

    void rebuildImage(
        const SceneSnapshot& snapshot,
        std::uint32_t capacityLights);

    [[nodiscard]] SyncOutcome makeOutcome(
        bool uploaded,
        bool grew,
        std::string_view reason,
        std::uint64_t frontendVersion,
        std::uint32_t currentFrame) const;

    SyncMode mode_;
    std::uint32_t maxDelayFrames_;
    bool initialized_ = false;
    SceneSnapshot residentSnapshot_{};
    std::uint32_t residentSourceFrame_ = 0;
    GpuBufferImage image_{};
    std::deque<PendingSnapshot> pending_;
};

[[nodiscard]] BufferValidation validateReadback(
    const std::vector<std::uint8_t>& actual,
    const GpuBufferImage& expected);

}  // namespace lightcluster
