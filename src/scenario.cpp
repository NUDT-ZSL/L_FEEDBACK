#include "scenario.h"

#include "wav_io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <system_error>

namespace spatial_audio {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;

Vec3 makePosition(float x, float y, float z) noexcept {
    return {x, y, z};
}

}  // namespace

MovingScenario::MovingScenario(AudioEngine& engine, std::uint64_t seed)
    : engine_(engine), seed_(seed) {}

bool MovingScenario::generateSample(const std::string& path,
                                    int toneHz,
                                    std::string& error) {
    constexpr std::size_t frames = static_cast<std::size_t>(kSampleRate);
    std::vector<std::int16_t> pcm(frames);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        const float time = static_cast<float>(frame) /
                           static_cast<float>(kSampleRate);
        const float fundamental = std::sin(kTwoPi * toneHz * time);
        const float second = std::sin(kTwoPi * 2.0f * toneHz * time + 0.3f);
        const float envelope = 0.88f + 0.12f * std::sin(kTwoPi * 2.0f * time);
        const float value = envelope * (0.31f * fundamental + 0.10f * second);
        pcm[frame] = static_cast<std::int16_t>(std::lrintf(value * 32767.0f));
    }
    std::error_code filesystemError;
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path(), filesystemError);
    if (filesystemError) {
        error = "cannot create sample directory: " +
                filesystemError.message();
        return false;
    }
    return detail::writeMonoWav16(path, pcm, error);
}

bool MovingScenario::addSource(int toneHz,
                               float radius,
                               float phase,
                               float angularSpeed,
                               ScenarioSource& output) {
    const std::filesystem::path path =
        std::filesystem::path(sampleDirectory_) /
        ("tone_" + std::to_string(toneHz) + "hz.wav");
    std::string error;
    if (!generateSample(path.string(), toneHz, error)) {
        std::fprintf(stderr, "spatial_audio: sample generation failed: %s\n",
                     error.c_str());
        return false;
    }

    SampleId sampleId = 0;
    if (engine_.loadWav(path.string(), sampleId) != ErrorCode::Ok) {
        return false;
    }

    output.toneHz = toneHz;
    output.radius = radius;
    output.phase = phase;
    output.angularSpeed = angularSpeed;
    SourceConfig config{};
    config.sample = sampleId;
    config.loop = true;
    config.initialPosition = sourcePosition(output, 0.0);
    config.baseGain = 0.16f;
    config.maxAudibleDistance = 16.0f;
    config.occludedCutoffHz = 950.0f;

    const ErrorCode result = engine_.addSource(config, output.id);
    if (result != ErrorCode::Ok) {
        std::fprintf(stderr,
                     "spatial_audio: scenario failed to add source: %s\n",
                     errorCodeName(result));
        return false;
    }
    return true;
}

bool MovingScenario::prepare(const std::string& sampleDirectory,
                             std::string& error) {
    sampleDirectory_ = sampleDirectory;

    static constexpr int kTones[] = {220, 247, 262, 294, 330, 349, 392, 440};
    std::mt19937_64 random(seed_);
    std::uniform_real_distribution<float> phaseDistribution(0.0f, kTwoPi);
    std::uniform_real_distribution<float> radiusDistribution(3.2f, 7.8f);
    std::uniform_real_distribution<float> speedDistribution(0.10f, 0.24f);

    permanent_.clear();
    permanent_.reserve(8);
    for (int tone : kTones) {
        ScenarioSource source{};
        if (!addSource(tone, radiusDistribution(random),
                       phaseDistribution(random), speedDistribution(random),
                       source)) {
            return false;
        }
        permanent_.push_back(source);
    }

    const std::filesystem::path transientPath =
        std::filesystem::path(sampleDirectory_) / "tone_494hz.wav";
    if (!generateSample(transientPath.string(), 494, error)) {
        return false;
    }
    if (engine_.loadWav(transientPath.string(), transientSampleId_) !=
        ErrorCode::Ok) {
        error = "failed to load generated transient sample";
        return false;
    }
    transient_.toneHz = 494;
    transient_.radius = 5.6f;
    transient_.phase = phaseDistribution(random);
    transient_.angularSpeed = 0.19f;

    ListenerState listener{};
    listener.position = makePosition(0.0f, 0.0f, 0.0f);
    listener.forward = makePosition(0.0f, 0.0f, 1.0f);
    listener.up = makePosition(0.0f, 1.0f, 0.0f);
    if (!engine_.setListener(listener)) {
        std::fprintf(stderr, "spatial_audio: failed to initialize listener\n");
        return false;
    }

    std::vector<Occluder> occluders(1);
    occluders[0].center = makePosition(0.0f, 0.0f, 4.5f);
    occluders[0].halfExtents = makePosition(2.2f, 3.0f, 0.12f);
    occluders[0].enabled = true;
    if (!engine_.setOccluders(occluders)) {
        std::fprintf(stderr, "spatial_audio: failed to configure occluder\n");
        return false;
    }
    return true;
}

bool MovingScenario::addTransientSource(double timeSeconds) {
    SourceConfig config{};
    config.sample = transientSampleId_;
    config.loop = true;
    config.initialPosition = sourcePosition(transient_, timeSeconds);
    config.baseGain = 0.16f;
    config.maxAudibleDistance = 16.0f;
    config.occludedCutoffHz = 950.0f;
    const ErrorCode result = engine_.addSource(config, transient_.id);
    if (result != ErrorCode::Ok) {
        std::fprintf(stderr,
                     "spatial_audio: failed to add transient source: %s\n",
                     errorCodeName(result));
        return false;
    }
    transientAdded_ = true;
    return true;
}

Vec3 MovingScenario::sourcePosition(const ScenarioSource& source,
                                    double timeSeconds) const {
    const double angle =
        source.phase + source.angularSpeed * timeSeconds;
    const double radius =
        source.radius + 0.45 * std::sin(0.31 * timeSeconds + source.phase);
    const double x = radius * std::cos(angle) +
                     0.6 * std::sin(0.43 * timeSeconds + source.phase);
    const double y = 0.55 * std::sin(0.27 * timeSeconds + 1.1 * source.phase);
    const double z = 6.0 + 1.8 * std::sin(angle + 0.8) +
                     0.35 * std::sin(0.19 * timeSeconds);
    return makePosition(static_cast<float>(x), static_cast<float>(y),
                        static_cast<float>(z));
}

void MovingScenario::update(double timeSeconds) {
    const double time = std::max(0.0, timeSeconds);
    const float yaw = static_cast<float>(0.70 * std::sin(0.43 * time));
    ListenerState listener{};
    listener.position = makePosition(
        static_cast<float>(1.5 * std::sin(0.37 * time)),
        static_cast<float>(0.2 * std::sin(0.21 * time)),
        static_cast<float>(0.9 * std::sin(0.29 * time)));
    listener.forward = makePosition(std::sin(yaw), 0.0f, std::cos(yaw));
    listener.up = makePosition(0.0f, 1.0f, 0.0f);
    engine_.setListener(listener);

    for (const ScenarioSource& source : permanent_) {
        engine_.setSourcePosition(source.id, sourcePosition(source, time));
    }

    if (!transientAdded_ && !transientRemoved_ &&
        time >= transientAddTime_) {
        addTransientSource(time);
    }
    if (transientAdded_ && !transientRemoved_ &&
        time >= transientRemoveTime_) {
        if (engine_.removeSource(transient_.id)) {
            transientRemoved_ = true;
        }
    }
    if (transientAdded_ && !transientRemoved_) {
        engine_.setSourcePosition(transient_.id,
                                  sourcePosition(transient_, time));
    }
}

}  // namespace spatial_audio
