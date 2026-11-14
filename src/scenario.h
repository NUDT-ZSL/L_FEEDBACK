#pragma once

#include "spatial_audio/audio_engine.h"

#include <cstdint>
#include <string>
#include <vector>

namespace spatial_audio {

struct ScenarioSource {
    SourceId id = kInvalidSourceId;
    int toneHz = 0;
    float radius = 0.0f;
    float phase = 0.0f;
    float angularSpeed = 0.0f;
};

class MovingScenario {
public:
    explicit MovingScenario(AudioEngine& engine, std::uint64_t seed = 0x5eed1234ULL);

    bool prepare(const std::string& sampleDirectory, std::string& error);
    void update(double timeSeconds);

    const std::vector<ScenarioSource>& permanentSources() const noexcept {
        return permanent_;
    }
    const ScenarioSource& transientSource() const noexcept {
        return transient_;
    }
    bool transientWasAdded() const noexcept { return transientAdded_; }
    bool transientWasRemoved() const noexcept { return transientRemoved_; }
    std::uint64_t seed() const noexcept { return seed_; }

private:
    bool generateSample(const std::string& path, int toneHz, std::string& error);
    bool addSource(int toneHz,
                   float radius,
                   float phase,
                   float angularSpeed,
                   ScenarioSource& output);
    bool addTransientSource(double timeSeconds);
    Vec3 sourcePosition(const ScenarioSource& source, double timeSeconds) const;

    AudioEngine& engine_;
    std::uint64_t seed_;
    std::string sampleDirectory_;
    std::vector<ScenarioSource> permanent_;
    ScenarioSource transient_{};
    SampleId transientSampleId_ = 0;
    bool transientAdded_ = false;
    bool transientRemoved_ = false;
    double transientAddTime_ = 1.0;
    double transientRemoveTime_ = 2.5;
};

}  // namespace spatial_audio

