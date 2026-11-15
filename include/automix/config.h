#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace automix {

struct DrcConfig {
    bool enabled = true;
    double thresholdDb = -18.0;
    double ratio = 4.0;
    double attackMs = 10.0;
    double releaseMs = 100.0;
    double kneeDb = 6.0;
};

struct LimiterConfig {
    bool enabled = true;
    double thresholdDbfs = -1.0;
    double lookaheadMs = 10.0;
    double releaseMs = 80.0;
};

struct TrackConfig {
    std::string name;
    std::filesystem::path input;
    double gainDb = 0.0;
    bool mute = false;
    bool solo = false;
    std::optional<DrcConfig> drc;
};

struct AppConfig {
    std::filesystem::path configPath;
    std::filesystem::path output;
    std::filesystem::path report;
    std::filesystem::path logDir;
    std::string logFile = "automix.log";
    double masterGainDb = 0.0;
    std::optional<DrcConfig> masterDrc;
    LimiterConfig limiter;
    std::vector<TrackConfig> tracks;
};

struct ConfigOverrides {
    std::optional<std::filesystem::path> logDir;
    std::optional<std::string> logFile;
};

AppConfig parseConfigFile(const std::filesystem::path& path,
                          const ConfigOverrides& overrides);
std::string configSchemaSummary();

}  // namespace automix
