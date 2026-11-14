#include "lightcluster/buffer_sync.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace lightcluster {

namespace {

std::uint32_t roundCapacity(std::uint32_t required) {
    std::uint32_t capacity = kMinimumBufferCapacity;
    while (capacity < required) {
        if (capacity > std::numeric_limits<std::uint32_t>::max() / 2) {
            return required;
        }
        capacity *= 2;
    }
    return capacity;
}

void writeU32(
    std::vector<std::uint8_t>& bytes,
    std::size_t offset,
    std::uint32_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

std::uint32_t readU32(
    const std::vector<std::uint8_t>& bytes,
    std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

GpuLightRecord packLight(const Light& light) {
    GpuLightRecord record{};
    record.positionType[0] = light.position.x;
    record.positionType[1] = light.position.y;
    record.positionType[2] = light.position.z;
    record.positionType[3] =
        light.type == LightType::Directional ? 1.0F : 0.0F;

    const Vec3 direction = normalizeOr(light.direction, {0.0F, -1.0F, 0.0F});
    record.directionRange[0] = direction.x;
    record.directionRange[1] = direction.y;
    record.directionRange[2] = direction.z;
    record.directionRange[3] =
        light.type == LightType::Point ? light.range : 0.0F;

    record.colorIntensity[0] = light.color.x;
    record.colorIntensity[1] = light.color.y;
    record.colorIntensity[2] = light.color.z;
    record.colorIntensity[3] = light.intensity;
    return record;
}

}  // namespace

std::string_view toString(SyncMode mode) noexcept {
    return mode == SyncMode::Immediate ? "immediate" : "deferred";
}

LightBufferSynchronizer::LightBufferSynchronizer(
    SyncMode mode,
    std::uint32_t maxDelayFrames)
    : mode_(mode), maxDelayFrames_(maxDelayFrames) {}

SyncOutcome LightBufferSynchronizer::initialize(
    const SceneSnapshot& snapshot,
    std::uint32_t sourceFrame) {
    if (initialized_) {
        throw std::logic_error("light buffer synchronizer already initialized");
    }
    const std::uint32_t required =
        static_cast<std::uint32_t>(snapshot.lights.size());
    const std::uint32_t capacity = roundCapacity(required);
    rebuildImage(snapshot, capacity);
    residentSnapshot_ = snapshot;
    residentSourceFrame_ = sourceFrame;
    initialized_ = true;
    return makeOutcome(true, true, "initial", snapshot.version, sourceFrame);
}

SyncOutcome LightBufferSynchronizer::sync(
    const SceneSnapshot& snapshot,
    std::uint32_t currentFrame) {
    if (!initialized_) {
        throw std::logic_error("light buffer synchronizer must be initialized");
    }

    // A repeated version has byte-identical contents. Method its effective
    // source frame forward so the reported age becomes zero after convergence.
    if (snapshot.version == residentSnapshot_.version) {
        residentSourceFrame_ = currentFrame;
    }

    if (mode_ == SyncMode::Immediate || maxDelayFrames_ == 0) {
        if (snapshot.version == residentSnapshot_.version) {
            return makeOutcome(
                false, false, "version_unchanged", snapshot.version,
                currentFrame);
        }
        return applySnapshot(
            snapshot, currentFrame, currentFrame, "immediate_version_change");
    }

    if (snapshot.version != residentSnapshot_.version &&
        (pending_.empty() ||
         pending_.back().snapshot.version != snapshot.version)) {
        pending_.push_back(PendingSnapshot{
            snapshot,
            currentFrame,
            static_cast<std::uint64_t>(currentFrame) + maxDelayFrames_,
        });
    }

    std::size_t dueIndex = pending_.size();
    for (std::size_t i = 0; i < pending_.size(); ++i) {
        if (pending_[i].readyFrame <= currentFrame) {
            dueIndex = i;
        } else {
            break;
        }
    }

    SyncOutcome result;
    if (dueIndex < pending_.size()) {
        const PendingSnapshot due = pending_[dueIndex];
        pending_.erase(
            pending_.begin(),
            pending_.begin() + static_cast<std::ptrdiff_t>(dueIndex + 1));
        if (due.snapshot.version == residentSnapshot_.version) {
            result = makeOutcome(
                false, false, "version_unchanged", snapshot.version,
                currentFrame);
        } else {
            result = applySnapshot(
                due.snapshot, due.sourceFrame, currentFrame, "deferred_due");
        }
    } else {
        result = makeOutcome(
            false,
            false,
            pending_.empty() ? "version_unchanged" : "deferred_wait",
            snapshot.version,
            currentFrame);
    }
    result.pendingSnapshots =
        static_cast<std::uint32_t>(pending_.size());
    return result;
}

const GpuBufferImage& LightBufferSynchronizer::residentImage() const noexcept {
    return image_;
}

const SceneSnapshot& LightBufferSynchronizer::residentSnapshot() const noexcept {
    return residentSnapshot_;
}

std::uint64_t LightBufferSynchronizer::residentVersion() const noexcept {
    return residentSnapshot_.version;
}

std::size_t LightBufferSynchronizer::pendingCount() const noexcept {
    return pending_.size();
}

SyncMode LightBufferSynchronizer::mode() const noexcept {
    return mode_;
}

std::uint32_t LightBufferSynchronizer::maxDelayFrames() const noexcept {
    return maxDelayFrames_;
}

SyncOutcome LightBufferSynchronizer::applySnapshot(
    const SceneSnapshot& snapshot,
    std::uint32_t sourceFrame,
    std::uint32_t currentFrame,
    std::string_view reason) {
    const std::uint32_t required =
        static_cast<std::uint32_t>(snapshot.lights.size());
    bool grew = false;
    std::uint32_t capacity = image_.capacityLights;
    if (required > capacity) {
        capacity = roundCapacity(required);
        grew = true;
    }
    rebuildImage(snapshot, capacity);
    residentSnapshot_ = snapshot;
    residentSourceFrame_ = sourceFrame;
    return makeOutcome(true, grew, reason, snapshot.version, currentFrame);
}

void LightBufferSynchronizer::rebuildImage(
    const SceneSnapshot& snapshot,
    std::uint32_t capacityLights) {
    const LightCounts counts = snapshot.counts();
    const std::size_t usedBytes =
        kGpuHeaderBytes + snapshot.lights.size() * kGpuLightBytes;

    image_.capacityLights = capacityLights;
    image_.usedBytes = static_cast<std::uint32_t>(usedBytes);
    image_.lightCount = counts.total;
    image_.pointCount = counts.point;
    image_.directionalCount = counts.directional;
    image_.version = snapshot.version;
    image_.bytes.assign(
        kGpuHeaderBytes + snapshot.lights.size() * kGpuLightBytes, 0);

    writeU32(image_.bytes, 0, counts.total);
    writeU32(image_.bytes, 4, counts.point);
    writeU32(image_.bytes, 8, counts.directional);
    writeU32(image_.bytes, 12, 0);
    writeU32(
        image_.bytes, 16,
        static_cast<std::uint32_t>(snapshot.version & 0xffffffffULL));
    writeU32(
        image_.bytes, 20,
        static_cast<std::uint32_t>(snapshot.version >> 32U));
    writeU32(image_.bytes, 24, 0);
    writeU32(image_.bytes, 28, 0);

    for (std::size_t i = 0; i < snapshot.lights.size(); ++i) {
        const GpuLightRecord record = packLight(snapshot.lights[i]);
        std::memcpy(
            image_.bytes.data() + kGpuHeaderBytes + i * kGpuLightBytes,
            &record, sizeof(record));
    }
}

SyncOutcome LightBufferSynchronizer::makeOutcome(
    bool uploaded,
    bool grew,
    std::string_view reason,
    std::uint64_t frontendVersion,
    std::uint32_t currentFrame) const {
    return {
        uploaded,
        grew,
        reason,
        frontendVersion,
        residentSnapshot_.version,
        image_.lightCount,
        image_.pointCount,
        image_.directionalCount,
        image_.capacityLights,
        image_.usedBytes,
        static_cast<std::uint32_t>(pending_.size()),
        residentSourceFrame_,
        currentFrame >= residentSourceFrame_
            ? currentFrame - residentSourceFrame_
            : 0U,
    };
}

BufferValidation validateReadback(
    const std::vector<std::uint8_t>& actual,
    const GpuBufferImage& expected) {
    BufferValidation result;
    result.expectedLights = expected.lightCount;
    result.expectedPoint = expected.pointCount;
    result.expectedDirectional = expected.directionalCount;
    result.expectedVersion = expected.version;

    if (actual.size() < kGpuHeaderBytes) {
        result.ok = false;
        result.code = "GPU_READBACK_TOO_SMALL";
        return result;
    }

    result.actualLights = readU32(actual, 0);
    result.actualPoint = readU32(actual, 4);
    result.actualDirectional = readU32(actual, 8);
    result.actualVersion =
        static_cast<std::uint64_t>(readU32(actual, 16)) |
        (static_cast<std::uint64_t>(readU32(actual, 20)) << 32U);

    if (result.actualVersion != expected.version) {
        result.ok = false;
        result.code = "GPU_VERSION_MISMATCH";
        return result;
    }
    if (result.actualLights != expected.lightCount) {
        result.ok = false;
        result.code = "GPU_COUNT_MISMATCH";
        return result;
    }
    if (result.actualPoint != expected.pointCount ||
        result.actualDirectional != expected.directionalCount) {
        result.ok = false;
        result.code = "GPU_TYPE_COUNT_MISMATCH";
        return result;
    }

    const std::size_t required =
        kGpuHeaderBytes + expected.lightCount * kGpuLightBytes;
    if (actual.size() < required) {
        result.ok = false;
        result.code = "GPU_READBACK_TOO_SMALL";
        return result;
    }
    if (std::memcmp(
            actual.data(),
            expected.bytes.data(),
            required) != 0) {
        result.ok = false;
        result.code = "GPU_LIGHT_DATA_MISMATCH";
    }
    return result;
}

}  // namespace lightcluster
