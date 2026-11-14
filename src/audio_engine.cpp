#include "spatial_audio/audio_engine.h"

#include "wav_io.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace spatial_audio {
namespace {

constexpr std::size_t kCommandCapacity = 128;
constexpr float kEpsilon = 1.0e-6f;
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kClearCutoffHz = 20000.0f;

static_assert(std::atomic<float>::is_always_lock_free,
              "audio callback requires lock-free float atomics");
static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
              "audio callback requires lock-free 32-bit atomics");
static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
              "audio callback requires lock-free 64-bit atomics");

Vec3 subtract(Vec3 a, Vec3 b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 cross(Vec3 a, Vec3 b) noexcept {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

float dot(Vec3 a, Vec3 b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float length(Vec3 v) noexcept {
    return std::sqrt(dot(v, v));
}

bool normalize(Vec3& v) noexcept {
    const float magnitude = length(v);
    if (!(magnitude > kEpsilon) || !std::isfinite(magnitude)) {
        return false;
    }
    v.x /= magnitude;
    v.y /= magnitude;
    v.z /= magnitude;
    return true;
}

float clampUnit(float value) noexcept {
    return std::max(-1.0f, std::min(1.0f, value));
}

float lowPassAlpha(float cutoffHz) noexcept {
    const float bounded = std::max(20.0f, std::min(kClearCutoffHz, cutoffHz));
    return 1.0f - std::exp(-kTwoPi * bounded / static_cast<float>(kSampleRate));
}

bool finiteVec(Vec3 v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

float softLimit(float value) noexcept {
    constexpr float knee = 0.90f;
    constexpr float ceiling = 0.98f;
    constexpr float range = ceiling - knee;

    if (!std::isfinite(value)) {
        return 0.0f;
    }
    const float magnitude = std::fabs(value);
    if (magnitude <= knee) {
        return value;
    }
    const float limited = knee + range * std::tanh((magnitude - knee) / range);
    return std::copysign(limited, value);
}

std::int16_t quantize(float value) noexcept {
    const float limited = std::max(-0.999f, std::min(0.999f, value));
    const long rounded = std::lrintf(limited * 32767.0f);
    return static_cast<std::int16_t>(
        std::max(-32768L, std::min(32767L, rounded)));
}

bool segmentIntersectsBox(Vec3 start,
                          Vec3 end,
                          Vec3 center,
                          Vec3 halfExtents) noexcept {
    if (!(halfExtents.x > 0.0f && halfExtents.y > 0.0f &&
          halfExtents.z > 0.0f)) {
        return false;
    }

    const Vec3 minimum{
        center.x - halfExtents.x,
        center.y - halfExtents.y,
        center.z - halfExtents.z,
    };
    const Vec3 maximum{
        center.x + halfExtents.x,
        center.y + halfExtents.y,
        center.z + halfExtents.z,
    };
    const Vec3 direction = subtract(end, start);
    const std::array<float, 3> origin{start.x, start.y, start.z};
    const std::array<float, 3> delta{direction.x, direction.y, direction.z};
    const std::array<float, 3> lower{minimum.x, minimum.y, minimum.z};
    const std::array<float, 3> upper{maximum.x, maximum.y, maximum.z};

    float entry = 0.0f;
    float exit = 1.0f;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (std::fabs(delta[axis]) < kEpsilon) {
            if (origin[axis] < lower[axis] || origin[axis] > upper[axis]) {
                return false;
            }
            continue;
        }
        const float inverse = 1.0f / delta[axis];
        float first = (lower[axis] - origin[axis]) * inverse;
        float second = (upper[axis] - origin[axis]) * inverse;
        if (first > second) {
            std::swap(first, second);
        }
        entry = std::max(entry, first);
        exit = std::min(exit, second);
        if (entry > exit) {
            return false;
        }
    }
    return true;
}

}  // namespace

struct AudioEngine::Impl {
    struct Sample {
        std::vector<float> data;
    };

    enum class SlotState : std::uint8_t {
        Free,
        Initializing,
        PendingAdd,
        Active,
        PendingRemove,
        Removed,
    };

    struct RuntimeSource {
        const Sample* sample = nullptr;
        bool loop = true;

        // Audio-thread-only state.
        std::size_t cursor = 0;
        float currentGain = 0.0f;
        float currentLateral = 0.0f;
        float filterState = 0.0f;
        float filterAlpha = 0.0f;
        float occlusionMix = 0.0f;
        bool fadingOut = false;

        // Control/audio-thread atomics.
        std::atomic<SlotState> state{SlotState::Free};
        std::atomic<std::uint32_t> publicId{kInvalidSourceId};
        std::atomic<float> positionX{0.0f};
        std::atomic<float> positionY{0.0f};
        std::atomic<float> positionZ{0.0f};
        std::atomic<float> baseGain{1.0f};
        std::atomic<float> maxDistance{20.0f};
        std::atomic<float> occludedCutoff{1200.0f};

        // Accumulators read by the control thread.
        std::atomic<std::uint64_t> statBlocks{0};
        std::atomic<float> statDistanceGain{0.0f};
        std::atomic<float> statPan{0.0f};
        std::atomic<float> statLeftGain{0.0f};
        std::atomic<float> statRightGain{0.0f};
        std::atomic<std::uint64_t> statOccludedBlocks{0};
        std::atomic<float> statCutoff{0.0f};
    };

    struct Command {
        enum class Type : std::uint8_t { Add, Remove };
        Type type = Type::Add;
        std::uint32_t slot = 0;
    };

    struct AtomicOccluder {
        std::atomic<bool> enabled{false};
        std::atomic<float> centerX{0.0f};
        std::atomic<float> centerY{0.0f};
        std::atomic<float> centerZ{0.0f};
        std::atomic<float> halfX{0.0f};
        std::atomic<float> halfY{0.0f};
        std::atomic<float> halfZ{0.0f};
    };

    Impl() {
        for (auto& slot : slots) {
            slot = std::make_unique<RuntimeSource>();
        }
        listenerForwardX.store(0.0f, std::memory_order_relaxed);
        listenerForwardY.store(0.0f, std::memory_order_relaxed);
        listenerForwardZ.store(1.0f, std::memory_order_relaxed);
        listenerUpX.store(0.0f, std::memory_order_relaxed);
        listenerUpY.store(1.0f, std::memory_order_relaxed);
        listenerUpZ.store(0.0f, std::memory_order_relaxed);
    }

    bool pushCommand(Command command) noexcept {
        const std::uint32_t head = commandHead.load(std::memory_order_relaxed);
        const std::uint32_t tail = commandTail.load(std::memory_order_acquire);
        if (head - tail >= kCommandCapacity) {
            return false;
        }
        commandQueue[head % kCommandCapacity] = command;
        commandHead.store(head + 1, std::memory_order_release);
        return true;
    }

    bool popCommand(Command& command) noexcept {
        const std::uint32_t tail = commandTail.load(std::memory_order_relaxed);
        const std::uint32_t head = commandHead.load(std::memory_order_acquire);
        if (tail == head) {
            return false;
        }
        command = commandQueue[tail % kCommandCapacity];
        commandTail.store(tail + 1, std::memory_order_release);
        return true;
    }

    void drainCommands() noexcept {
        Command command{};
        while (popCommand(command)) {
            RuntimeSource& source = *slots[command.slot];
            if (command.type == Command::Type::Add) {
                const SlotState observed =
                    source.state.load(std::memory_order_acquire);
                if (observed != SlotState::PendingAdd &&
                    observed != SlotState::PendingRemove) {
                    continue;
                }
                source.cursor = 0;
                source.currentGain = 0.0f;
                source.currentLateral = 0.0f;
                source.filterState = 0.0f;
                source.filterAlpha = lowPassAlpha(kClearCutoffHz);
                source.occlusionMix = 0.0f;
                source.fadingOut = false;

                source.statBlocks.store(0, std::memory_order_relaxed);
                source.statDistanceGain.store(0.0f, std::memory_order_relaxed);
                source.statPan.store(0.0f, std::memory_order_relaxed);
                source.statLeftGain.store(0.0f, std::memory_order_relaxed);
                source.statRightGain.store(0.0f, std::memory_order_relaxed);
                source.statOccludedBlocks.store(0, std::memory_order_relaxed);
                source.statCutoff.store(0.0f, std::memory_order_relaxed);

                if (activeCount < active.size()) {
                    active[activeCount++] = command.slot;
                    if (observed == SlotState::PendingRemove) {
                        source.fadingOut = true;
                    } else {
                        SlotState expected = SlotState::PendingAdd;
                        if (!source.state.compare_exchange_strong(
                                expected, SlotState::Active,
                                std::memory_order_acq_rel,
                                std::memory_order_acquire)) {
                            source.fadingOut = true;
                        }
                    }
                }
            } else {
                if (source.state.load(std::memory_order_acquire) ==
                    SlotState::PendingRemove) {
                    source.fadingOut = true;
                }
            }
        }
    }

    bool isOccluded(Vec3 listener, Vec3 source) const noexcept {
        for (const AtomicOccluder& occluder : occluders) {
            if (!occluder.enabled.load(std::memory_order_acquire)) {
                continue;
            }
            const Vec3 center{
                occluder.centerX.load(std::memory_order_relaxed),
                occluder.centerY.load(std::memory_order_relaxed),
                occluder.centerZ.load(std::memory_order_relaxed),
            };
            const Vec3 halfExtents{
                occluder.halfX.load(std::memory_order_relaxed),
                occluder.halfY.load(std::memory_order_relaxed),
                occluder.halfZ.load(std::memory_order_relaxed),
            };
            if (segmentIntersectsBox(listener, source, center, halfExtents)) {
                return true;
            }
        }
        return false;
    }

    RuntimeSource* findSource(SourceId id) noexcept {
        if (id == kInvalidSourceId) {
            return nullptr;
        }
        const std::uint32_t slotIndex = id & 0xffffu;
        if (slotIndex >= slots.size()) {
            return nullptr;
        }
        RuntimeSource& source = *slots[slotIndex];
        if (source.publicId.load(std::memory_order_acquire) != id) {
            return nullptr;
        }
        return &source;
    }

    const RuntimeSource* findSource(SourceId id) const noexcept {
        return const_cast<Impl*>(this)->findSource(id);
    }

    static bool sourceAcceptsControl(SlotState state) noexcept {
        return state == SlotState::PendingAdd || state == SlotState::Active ||
               state == SlotState::PendingRemove;
    }

    std::array<std::unique_ptr<RuntimeSource>, kMaxSources> slots;
    std::vector<std::unique_ptr<Sample>> samples;
    std::array<std::uint32_t, kMaxSources> generation{};

    std::array<Command, kCommandCapacity> commandQueue{};
    std::atomic<std::uint32_t> commandHead{0};
    std::atomic<std::uint32_t> commandTail{0};

    std::array<std::uint32_t, kMaxSources> active{};
    std::size_t activeCount = 0;

    std::atomic<float> listenerPositionX{0.0f};
    std::atomic<float> listenerPositionY{0.0f};
    std::atomic<float> listenerPositionZ{0.0f};
    std::atomic<float> listenerForwardX{0.0f};
    std::atomic<float> listenerForwardY{0.0f};
    std::atomic<float> listenerForwardZ{1.0f};
    std::atomic<float> listenerUpX{0.0f};
    std::atomic<float> listenerUpY{1.0f};
    std::atomic<float> listenerUpZ{0.0f};

    std::array<AtomicOccluder, kMaxOccluders> occluders;
    std::array<float, kBlockFrames * kChannels> mixBuffer{};
};

AudioEngine::AudioEngine() : impl_(std::make_unique<Impl>()) {}

AudioEngine::~AudioEngine() = default;

const char* errorCodeName(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::Ok:
            return "ok";
        case ErrorCode::InvalidArgument:
            return "invalid argument";
        case ErrorCode::WavOpenFailed:
            return "WAV open failed";
        case ErrorCode::WavUnsupportedFormat:
            return "unsupported WAV format";
        case ErrorCode::WavReadFailed:
            return "WAV read failed";
        case ErrorCode::NoFreeSourceSlot:
            return "no free source slot";
        case ErrorCode::CommandQueueFull:
            return "source command queue full";
        case ErrorCode::UnknownSample:
            return "unknown sample";
        case ErrorCode::PortAudioInitFailed:
            return "PortAudio initialization failed";
        case ErrorCode::PortAudioStreamOpenFailed:
            return "PortAudio stream open failed";
        case ErrorCode::PortAudioStreamStartFailed:
            return "PortAudio stream start failed";
    }
    return "unknown error";
}

ErrorCode AudioEngine::loadWav(const std::string& path, SampleId& sampleId) {
    std::vector<float> decoded;
    std::string error;
    if (!detail::readMonoWav16(path, decoded, error)) {
        std::fprintf(stderr, "spatial_audio: failed to load '%s': %s\n",
                     path.c_str(), error.c_str());
        if (error.rfind("cannot open", 0) == 0) {
            return ErrorCode::WavOpenFailed;
        }
        if (error.rfind("unsupported", 0) == 0) {
            return ErrorCode::WavUnsupportedFormat;
        }
        return ErrorCode::WavReadFailed;
    }

    try {
        auto sample = std::make_unique<Impl::Sample>();
        sample->data = std::move(decoded);
        sampleId = static_cast<SampleId>(impl_->samples.size());
        impl_->samples.push_back(std::move(sample));
        return ErrorCode::Ok;
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr,
                     "spatial_audio: failed to load '%s': out of memory\n",
                     path.c_str());
        return ErrorCode::WavReadFailed;
    }
}

ErrorCode AudioEngine::addSource(const SourceConfig& config,
                                  SourceId& sourceId) {
    sourceId = kInvalidSourceId;
    if (config.sample >= impl_->samples.size()) {
        std::fprintf(stderr,
                     "spatial_audio: cannot add source: sample id %u is not "
                     "loaded\n",
                     config.sample);
        return ErrorCode::UnknownSample;
    }
    if (!finiteVec(config.initialPosition) ||
        !std::isfinite(config.baseGain) || config.baseGain < 0.0f ||
        !std::isfinite(config.maxAudibleDistance) ||
        config.maxAudibleDistance <= 0.0f ||
        !std::isfinite(config.occludedCutoffHz) ||
        config.occludedCutoffHz <= 0.0f ||
        config.occludedCutoffHz >
            static_cast<float>(kSampleRate) * 0.5f) {
        std::fprintf(stderr,
                     "spatial_audio: cannot add source: invalid position, gain, "
                     "distance, or cutoff\n");
        return ErrorCode::InvalidArgument;
    }

    for (std::uint32_t index = 0; index < impl_->slots.size(); ++index) {
        auto& slot = *impl_->slots[index];
        auto previous = slot.state.load(std::memory_order_acquire);
        if (previous != Impl::SlotState::Free &&
            previous != Impl::SlotState::Removed) {
            continue;
        }
        if (!slot.state.compare_exchange_strong(
                previous, Impl::SlotState::Initializing,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            continue;
        }

        std::uint32_t nextGeneration = (impl_->generation[index] + 1u) & 0xffffu;
        if (nextGeneration == 0) {
            nextGeneration = 1;
        }
        impl_->generation[index] = nextGeneration;
        const SourceId id = (nextGeneration << 16u) | index;

        slot.sample = impl_->samples[config.sample].get();
        slot.loop = config.loop;
        slot.publicId.store(id, std::memory_order_relaxed);
        slot.positionX.store(config.initialPosition.x, std::memory_order_relaxed);
        slot.positionY.store(config.initialPosition.y, std::memory_order_relaxed);
        slot.positionZ.store(config.initialPosition.z, std::memory_order_relaxed);
        slot.baseGain.store(config.baseGain, std::memory_order_relaxed);
        slot.maxDistance.store(config.maxAudibleDistance,
                               std::memory_order_relaxed);
        slot.occludedCutoff.store(config.occludedCutoffHz,
                                  std::memory_order_relaxed);

        slot.state.store(Impl::SlotState::PendingAdd, std::memory_order_release);
        const Impl::Command command{Impl::Command::Type::Add, index};
        if (!impl_->pushCommand(command)) {
            slot.publicId.store(kInvalidSourceId, std::memory_order_release);
            slot.state.store(previous, std::memory_order_release);
            std::fprintf(stderr,
                         "spatial_audio: cannot add source: command queue "
                         "full\n");
            return ErrorCode::CommandQueueFull;
        }
        sourceId = id;
        return ErrorCode::Ok;
    }

    std::fprintf(stderr, "spatial_audio: cannot add source: all %zu slots busy\n",
                 kMaxSources);
    return ErrorCode::NoFreeSourceSlot;
}

bool AudioEngine::removeSource(SourceId sourceId) noexcept {
    Impl::RuntimeSource* source = impl_->findSource(sourceId);
    if (source == nullptr) {
        return false;
    }
    auto expected = source->state.load(std::memory_order_acquire);
    if (expected != Impl::SlotState::Active &&
        expected != Impl::SlotState::PendingAdd) {
        return false;
    }
    if (!source->state.compare_exchange_strong(
            expected, Impl::SlotState::PendingRemove,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return false;
    }
    const Impl::Command command{
        Impl::Command::Type::Remove,
        static_cast<std::uint32_t>(sourceId & 0xffffu),
    };
    if (!impl_->pushCommand(command)) {
        source->state.store(Impl::SlotState::Active, std::memory_order_release);
        return false;
    }
    return true;
}

bool AudioEngine::setSourcePosition(SourceId sourceId,
                                    Vec3 position) noexcept {
    if (!finiteVec(position)) {
        return false;
    }
    Impl::RuntimeSource* source = impl_->findSource(sourceId);
    if (source == nullptr ||
        !Impl::sourceAcceptsControl(
            source->state.load(std::memory_order_acquire))) {
        return false;
    }
    source->positionX.store(position.x, std::memory_order_relaxed);
    source->positionY.store(position.y, std::memory_order_relaxed);
    source->positionZ.store(position.z, std::memory_order_relaxed);
    return true;
}

bool AudioEngine::setSourceBaseGain(SourceId sourceId,
                                    float gain) noexcept {
    if (!std::isfinite(gain) || gain < 0.0f) {
        return false;
    }
    Impl::RuntimeSource* source = impl_->findSource(sourceId);
    if (source == nullptr ||
        !Impl::sourceAcceptsControl(
            source->state.load(std::memory_order_acquire))) {
        return false;
    }
    source->baseGain.store(gain, std::memory_order_relaxed);
    return true;
}

bool AudioEngine::setSourceMaxDistance(SourceId sourceId,
                                       float distance) noexcept {
    if (!std::isfinite(distance) || distance <= 0.0f) {
        return false;
    }
    Impl::RuntimeSource* source = impl_->findSource(sourceId);
    if (source == nullptr ||
        !Impl::sourceAcceptsControl(
            source->state.load(std::memory_order_acquire))) {
        return false;
    }
    source->maxDistance.store(distance, std::memory_order_relaxed);
    return true;
}

bool AudioEngine::setSourceOccludedCutoff(SourceId sourceId,
                                          float cutoffHz) noexcept {
    if (!std::isfinite(cutoffHz) || cutoffHz <= 0.0f ||
        cutoffHz > static_cast<float>(kSampleRate) * 0.5f) {
        return false;
    }
    Impl::RuntimeSource* source = impl_->findSource(sourceId);
    if (source == nullptr ||
        !Impl::sourceAcceptsControl(
            source->state.load(std::memory_order_acquire))) {
        return false;
    }
    source->occludedCutoff.store(cutoffHz, std::memory_order_relaxed);
    return true;
}

bool AudioEngine::setListener(const ListenerState& listener) noexcept {
    if (!finiteVec(listener.position) || !finiteVec(listener.forward) ||
        !finiteVec(listener.up)) {
        return false;
    }
    Vec3 forward = listener.forward;
    Vec3 up = listener.up;
    if (!normalize(forward) || !normalize(up)) {
        return false;
    }
    Vec3 right = cross(up, forward);
    if (!normalize(right)) {
        return false;
    }

    impl_->listenerPositionX.store(listener.position.x,
                                   std::memory_order_relaxed);
    impl_->listenerPositionY.store(listener.position.y,
                                   std::memory_order_relaxed);
    impl_->listenerPositionZ.store(listener.position.z,
                                   std::memory_order_relaxed);
    impl_->listenerForwardX.store(forward.x, std::memory_order_relaxed);
    impl_->listenerForwardY.store(forward.y, std::memory_order_relaxed);
    impl_->listenerForwardZ.store(forward.z, std::memory_order_relaxed);
    impl_->listenerUpX.store(up.x, std::memory_order_relaxed);
    impl_->listenerUpY.store(up.y, std::memory_order_relaxed);
    impl_->listenerUpZ.store(up.z, std::memory_order_relaxed);
    return true;
}

bool AudioEngine::setOccluders(
    const std::vector<Occluder>& occluders) noexcept {
    if (occluders.size() > kMaxOccluders) {
        return false;
    }
    for (const Occluder& occluder : occluders) {
        if (!finiteVec(occluder.center) || !finiteVec(occluder.halfExtents) ||
            occluder.halfExtents.x <= 0.0f ||
            occluder.halfExtents.y <= 0.0f ||
            occluder.halfExtents.z <= 0.0f) {
            return false;
        }
    }

    for (auto& target : impl_->occluders) {
        target.enabled.store(false, std::memory_order_relaxed);
    }
    for (std::size_t i = 0; i < occluders.size(); ++i) {
        auto& target = impl_->occluders[i];
        const Occluder& source = occluders[i];
        target.centerX.store(source.center.x, std::memory_order_relaxed);
        target.centerY.store(source.center.y, std::memory_order_relaxed);
        target.centerZ.store(source.center.z, std::memory_order_relaxed);
        target.halfX.store(source.halfExtents.x, std::memory_order_relaxed);
        target.halfY.store(source.halfExtents.y, std::memory_order_relaxed);
        target.halfZ.store(source.halfExtents.z, std::memory_order_relaxed);
        target.enabled.store(source.enabled, std::memory_order_release);
    }
    return true;
}

void AudioEngine::processBlock(std::int16_t* interleavedStereo,
                               std::size_t frames) noexcept {
    if (interleavedStereo == nullptr) {
        return;
    }
    if (frames > kBlockFrames) {
        std::fill(interleavedStereo, interleavedStereo + frames * kChannels,
                  static_cast<std::int16_t>(0));
        return;
    }
    if (frames == 0) {
        return;
    }

    impl_->drainCommands();

    const std::size_t sampleCount = frames * kChannels;
    std::fill_n(impl_->mixBuffer.begin(), sampleCount, 0.0f);

    const Vec3 listenerPosition{
        impl_->listenerPositionX.load(std::memory_order_relaxed),
        impl_->listenerPositionY.load(std::memory_order_relaxed),
        impl_->listenerPositionZ.load(std::memory_order_relaxed),
    };
    Vec3 forward{
        impl_->listenerForwardX.load(std::memory_order_relaxed),
        impl_->listenerForwardY.load(std::memory_order_relaxed),
        impl_->listenerForwardZ.load(std::memory_order_relaxed),
    };
    Vec3 up{
        impl_->listenerUpX.load(std::memory_order_relaxed),
        impl_->listenerUpY.load(std::memory_order_relaxed),
        impl_->listenerUpZ.load(std::memory_order_relaxed),
    };
    if (!normalize(forward)) {
        forward = {0.0f, 0.0f, 1.0f};
    }
    if (!normalize(up)) {
        up = {0.0f, 1.0f, 0.0f};
    }
    Vec3 right = cross(up, forward);
    if (!normalize(right)) {
        right = {1.0f, 0.0f, 0.0f};
    }

    for (std::size_t activeIndex = 0; activeIndex < impl_->activeCount;) {
        const std::uint32_t slotIndex = impl_->active[activeIndex];
        Impl::RuntimeSource& source = *impl_->slots[slotIndex];
        const Impl::SlotState state =
            source.state.load(std::memory_order_acquire);
        if (state != Impl::SlotState::Active &&
            state != Impl::SlotState::PendingRemove) {
            impl_->active[activeIndex] = impl_->active[--impl_->activeCount];
            continue;
        }

        const Vec3 sourcePosition{
            source.positionX.load(std::memory_order_relaxed),
            source.positionY.load(std::memory_order_relaxed),
            source.positionZ.load(std::memory_order_relaxed),
        };
        const float baseGain =
            source.baseGain.load(std::memory_order_relaxed);
        const float maxDistance =
            source.maxDistance.load(std::memory_order_relaxed);
        const float occludedCutoff =
            source.occludedCutoff.load(std::memory_order_relaxed);

        const Vec3 offset = subtract(sourcePosition, listenerPosition);
        const float distance = length(offset);
        float distanceGain = 0.0f;
        if (distance <= kEpsilon) {
            distanceGain = 1.0f;
        } else if (distance < maxDistance && maxDistance > kEpsilon) {
            distanceGain = 1.0f - distance / maxDistance;
        }
        distanceGain = std::max(0.0f, std::min(1.0f, distanceGain));

        float lateral = 0.0f;
        if (distance > kEpsilon) {
            lateral = clampUnit(dot(offset, right) / distance);
        }
        const float leftPan = std::sqrt(0.5f * (1.0f - lateral));
        const float rightPan = std::sqrt(0.5f * (1.0f + lateral));

        const bool occluded = impl_->isOccluded(listenerPosition, sourcePosition);
        const float outputGain = distanceGain * baseGain;
        const float targetGain = source.fadingOut ? 0.0f : outputGain;
        const float targetLateral = lateral;

        const float inverseFrames = 1.0f / static_cast<float>(frames);
        const float gainStep =
            (targetGain - source.currentGain) * inverseFrames;
        const float lateralStep =
            (targetLateral - source.currentLateral) * inverseFrames;

        const float targetCutoff = occluded ? occludedCutoff : kClearCutoffHz;
        const float targetAlpha = lowPassAlpha(targetCutoff);
        const float alphaStep =
            (targetAlpha - source.filterAlpha) * inverseFrames;
        const float targetOcclusionMix = occluded ? 1.0f : 0.0f;
        const float occlusionStep =
            (targetOcclusionMix - source.occlusionMix) * inverseFrames;

        const std::vector<float>* sampleData =
            source.sample != nullptr ? &source.sample->data : nullptr;
        const std::size_t sampleFrames =
            sampleData != nullptr ? sampleData->size() : 0;

        for (std::size_t frame = 0; frame < frames; ++frame) {
            float input = 0.0f;
            if (sampleFrames > 0) {
                if (source.cursor < sampleFrames) {
                    input = (*sampleData)[source.cursor++];
                }
                if (source.cursor >= sampleFrames) {
                    if (source.loop) {
                        source.cursor = 0;
                    }
                }
            }

            source.filterAlpha += alphaStep;
            source.filterState +=
                source.filterAlpha * (input - source.filterState);
            source.occlusionMix += occlusionStep;
            const float mixed =
                input + source.occlusionMix * (source.filterState - input);

            source.currentGain += gainStep;
            source.currentLateral += lateralStep;
            const float frameLateral = clampUnit(source.currentLateral);
            const float frameGain = std::max(0.0f, source.currentGain);
            const float frameLeftPan =
                std::sqrt(0.5f * (1.0f - frameLateral));
            const float frameRightPan =
                std::sqrt(0.5f * (1.0f + frameLateral));
            impl_->mixBuffer[frame * kChannels] +=
                mixed * frameGain * frameLeftPan;
            impl_->mixBuffer[frame * kChannels + 1] +=
                mixed * frameGain * frameRightPan;
        }

        source.currentGain = targetGain;
        source.currentLateral = targetLateral;
        source.filterAlpha = targetAlpha;
        source.occlusionMix = targetOcclusionMix;

        const std::uint64_t previousBlocks =
            source.statBlocks.load(std::memory_order_relaxed);
        source.statBlocks.store(previousBlocks + 1, std::memory_order_relaxed);
        source.statDistanceGain.store(
            source.statDistanceGain.load(std::memory_order_relaxed) +
                distanceGain,
            std::memory_order_relaxed);
        source.statPan.store(
            source.statPan.load(std::memory_order_relaxed) + lateral,
            std::memory_order_relaxed);
        source.statLeftGain.store(
            source.statLeftGain.load(std::memory_order_relaxed) +
                outputGain * leftPan,
            std::memory_order_relaxed);
        source.statRightGain.store(
            source.statRightGain.load(std::memory_order_relaxed) +
                outputGain * rightPan,
            std::memory_order_relaxed);
        if (occluded) {
            source.statOccludedBlocks.fetch_add(1, std::memory_order_relaxed);
        }
        source.statCutoff.store(
            source.statCutoff.load(std::memory_order_relaxed) + targetCutoff,
            std::memory_order_relaxed);

        if (source.fadingOut) {
            source.state.store(Impl::SlotState::Removed,
                               std::memory_order_release);
            impl_->active[activeIndex] = impl_->active[--impl_->activeCount];
            continue;
        }
        ++activeIndex;
    }

    for (std::size_t frame = 0; frame < frames; ++frame) {
        const float left = softLimit(impl_->mixBuffer[frame * kChannels]);
        const float right =
            softLimit(impl_->mixBuffer[frame * kChannels + 1]);
        interleavedStereo[frame * kChannels] = quantize(left);
        interleavedStereo[frame * kChannels + 1] = quantize(right);
    }
}

bool AudioEngine::sourceStats(SourceId sourceId,
                              SourceStats& stats) const noexcept {
    const Impl::RuntimeSource* source = impl_->findSource(sourceId);
    if (source == nullptr) {
        return false;
    }
    const std::uint64_t blocks =
        source->statBlocks.load(std::memory_order_acquire);
    if (blocks == 0) {
        stats = {};
        return true;
    }
    const float divisor = static_cast<float>(blocks);
    stats.processedBlocks = blocks;
    stats.averageDistanceGain =
        source->statDistanceGain.load(std::memory_order_relaxed) / divisor;
    stats.averagePan =
        source->statPan.load(std::memory_order_relaxed) / divisor;
    stats.averageLeftGain =
        source->statLeftGain.load(std::memory_order_relaxed) / divisor;
    stats.averageRightGain =
        source->statRightGain.load(std::memory_order_relaxed) / divisor;
    stats.occlusionFraction =
        static_cast<float>(
            source->statOccludedBlocks.load(std::memory_order_relaxed)) /
        divisor;
    stats.averageCutoffHz =
        source->statCutoff.load(std::memory_order_relaxed) / divisor;
    return true;
}

std::size_t AudioEngine::activeSourceCount() const noexcept {
    std::size_t count = 0;
    for (const auto& slot : impl_->slots) {
        const auto state = slot->state.load(std::memory_order_acquire);
        if (state == Impl::SlotState::Active ||
            state == Impl::SlotState::PendingRemove) {
            ++count;
        }
    }
    return count;
}

std::size_t AudioEngine::loadedSampleCount() const noexcept {
    return impl_->samples.size();
}

}  // namespace spatial_audio
