#pragma once

#include "automix/config.h"
#include "automix/drc.h"
#include "automix/limiter.h"
#include "automix/logger.h"
#include "automix/loudness.h"
#include "automix/stereo.h"

#include <cstdint>
#include <string>
#include <vector>

namespace automix {

struct TrackReport {
    std::string name;
    std::string sourcePath;
    std::string sourceFormat;
    int sourceSampleRate = 0;
    int sourceChannels = 0;
    std::uint64_t sourceFrames = 0;
    std::uint64_t outputFrames = 0;
    double configuredGainDb = 0.0;
    double effectiveGainDb = 0.0;
    bool muted = false;
    bool soloSuppressed = false;
    bool drcEnabled = false;
    DrcConfig drc;
    DrcStats drcStats;
    LevelStats postDrcLevels;
    double readMs = 0.0;
    double resampleMs = 0.0;
    double gainPanMs = 0.0;
    double drcMs = 0.0;
};

struct RenderResult {
    StereoBuffer audio;
    std::vector<TrackReport> tracks;
    LevelStats mixLevels;
    bool masterDrcEnabled = false;
    DrcConfig masterDrc;
    DrcStats masterDrcStats;
    LevelStats preLimiterLevels;
    LimiterStats limiterStats;
    LoudnessResult loudness;
    double mixMs = 0.0;
    double masterDrcMs = 0.0;
    double limiterMs = 0.0;
    double loudnessMs = 0.0;
    double totalMs = 0.0;
};

RenderResult renderMix(const AppConfig& config, Logger& logger);
std::string buildReportJson(const AppConfig& config,
                            const RenderResult& result,
                            double writeMs);

}  // namespace automix
