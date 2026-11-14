#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace spatial_audio {

constexpr int kSampleRate = 44100;
constexpr std::size_t kBlockFrames = 256;
constexpr std::size_t kChannels = 2;
constexpr std::size_t kMaxSources = 32;
constexpr std::size_t kMaxOccluders = 16;
constexpr std::uint32_t kInvalidSourceId = 0;

using SampleId = std::uint32_t;
using SourceId = std::uint32_t;

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct ListenerState {
    Vec3 position{};
    Vec3 forward{0.0f, 0.0f, 1.0f};
    Vec3 up{0.0f, 1.0f, 0.0f};
};

struct SourceConfig {
    SampleId sample = 0;
    bool loop = true;
    Vec3 initialPosition{};
    float baseGain = 1.0f;
    float maxAudibleDistance = 20.0f;
    float occludedCutoffHz = 1200.0f;
};

struct Occluder {
    Vec3 center{};
    Vec3 halfExtents{};
    bool enabled = true;
};

struct SourceStats {
    std::uint64_t processedBlocks = 0;
    float averageDistanceGain = 0.0f;
    float averagePan = 0.0f;
    float averageLeftGain = 0.0f;
    float averageRightGain = 0.0f;
    float occlusionFraction = 0.0f;
    float averageCutoffHz = 0.0f;
};

enum class ErrorCode : int {
    Ok = 0,
    InvalidArgument = 1,
    WavOpenFailed = 2,
    WavUnsupportedFormat = 3,
    WavReadFailed = 4,
    NoFreeSourceSlot = 5,
    CommandQueueFull = 6,
    UnknownSample = 7,
    PortAudioInitFailed = 20,
    PortAudioStreamOpenFailed = 21,
    PortAudioStreamStartFailed = 22
};

const char* errorCodeName(ErrorCode code) noexcept;

// The control methods are intended for one main/control thread. processBlock()
// is intended for exactly one audio thread. Position and listener updates cross
// the thread boundary through lock-free atomics; source add/remove operations
// cross it through a bounded SPSC command queue.
class AudioEngine {
public:
    AudioEngine();
    ~AudioEngine();

    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;
    AudioEngine(AudioEngine&&) = delete;
    AudioEngine& operator=(AudioEngine&&) = delete;

    // Loads a mono, 44.1 kHz, 16-bit PCM WAV. On failure, prints a precise
    // diagnostic to stderr and returns a non-Ok code.
    ErrorCode loadWav(const std::string& path, SampleId& sampleId);

    ErrorCode addSource(const SourceConfig& config, SourceId& sourceId);
    bool removeSource(SourceId sourceId) noexcept;

    bool setSourcePosition(SourceId sourceId, Vec3 position) noexcept;
    bool setSourceBaseGain(SourceId sourceId, float gain) noexcept;
    bool setSourceMaxDistance(SourceId sourceId, float distance) noexcept;
    bool setSourceOccludedCutoff(SourceId sourceId, float cutoffHz) noexcept;

    bool setListener(const ListenerState& listener) noexcept;
    bool setOccluders(const std::vector<Occluder>& occluders) noexcept;

    // Must be called only from the audio thread (or the deterministic driver
    // when no stream is running). frames <= kBlockFrames. The function performs
    // no allocation, locking, file I/O, or logging.
    void processBlock(std::int16_t* interleavedStereo, std::size_t frames) noexcept;

    bool sourceStats(SourceId sourceId, SourceStats& stats) const noexcept;
    std::size_t activeSourceCount() const noexcept;
    std::size_t loadedSampleCount() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace spatial_audio

