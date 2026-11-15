#include "automix/config.h"

#include "automix/error.h"
#include "automix/json.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>

namespace automix {

namespace {

[[noreturn]] void missing(const std::string& path) {
    throw AppError(ErrorCode::CfgMissing, "missing required field '" + path + "'");
}

[[noreturn]] void invalid(const std::string& path, const std::string& detail) {
    throw AppError(ErrorCode::CfgValue, "invalid field '" + path + "': " + detail);
}

const JsonValue& requireField(const JsonValue& object, const std::string& key,
                              const std::string& path) {
    if (!object.isObject()) {
        invalid(path, "expected an object");
    }
    const JsonValue* value = object.find(key);
    if (value == nullptr) {
        missing(path.empty() ? key : path + "." + key);
    }
    return *value;
}

const JsonValue* optionalField(const JsonValue& object, const std::string& key) {
    return object.isObject() ? object.find(key) : nullptr;
}

std::string asString(const JsonValue& value, const std::string& path) {
    if (!value.isString() || value.asString().empty()) {
        invalid(path, "expected a non-empty string");
    }
    return value.asString();
}

double asNumber(const JsonValue& value, const std::string& path) {
    if (!value.isNumber() || !std::isfinite(value.asNumber())) {
        invalid(path, "expected a finite number");
    }
    return value.asNumber();
}

bool asBool(const JsonValue& value, const std::string& path) {
    if (!value.isBool()) {
        invalid(path, "expected true or false");
    }
    return value.asBool();
}

double requireNumber(const JsonValue& object, const std::string& key,
                     const std::string& path) {
    return asNumber(requireField(object, key, path), path + "." + key);
}

double optionalNumber(const JsonValue& object, const std::string& key,
                      const std::string& path, double fallback) {
    const JsonValue* value = optionalField(object, key);
    return value == nullptr ? fallback : asNumber(*value, path + "." + key);
}

bool optionalBool(const JsonValue& object, const std::string& key,
                  const std::string& path, bool fallback) {
    const JsonValue* value = optionalField(object, key);
    return value == nullptr ? fallback : asBool(*value, path + "." + key);
}

void requireRange(double value, double low, double high,
                  const std::string& path) {
    if (value < low || value > high) {
        std::ostringstream out;
        out << "value " << value << " is outside [" << low << ", " << high << "]";
        invalid(path, out.str());
    }
}

std::filesystem::path resolvePath(const std::filesystem::path& base,
                                  const std::string& text) {
    std::filesystem::path path = std::filesystem::u8path(text);
    if (path.is_relative()) {
        path = base / path;
    }
    return path.lexically_normal();
}

DrcConfig parseDrc(const JsonValue& value, const std::string& path) {
    if (!value.isObject()) {
        invalid(path, "expected an object");
    }
    DrcConfig config;
    config.enabled = optionalBool(value, "enabled", path, true);
    if (!config.enabled) {
        return config;
    }
    config.thresholdDb = requireNumber(value, "threshold_db", path);
    config.ratio = requireNumber(value, "ratio", path);
    config.attackMs = requireNumber(value, "attack_ms", path);
    config.releaseMs = requireNumber(value, "release_ms", path);
    const JsonValue* knee = optionalField(value, "knee_db");
    const JsonValue* softKnee = optionalField(value, "soft_knee_db");
    if (knee != nullptr && softKnee != nullptr) {
        invalid(path, "specify only one of knee_db and soft_knee_db");
    }
    if (knee != nullptr) {
        config.kneeDb = asNumber(*knee, path + ".knee_db");
    } else if (softKnee != nullptr) {
        config.kneeDb = asNumber(*softKnee, path + ".soft_knee_db");
    } else {
        missing(path + ".knee_db");
    }
    requireRange(config.thresholdDb, -40.0, 0.0, path + ".threshold_db");
    requireRange(config.ratio, 1.0, 20.0, path + ".ratio");
    requireRange(config.attackMs, 5.0, 200.0, path + ".attack_ms");
    requireRange(config.releaseMs, 5.0, 200.0, path + ".release_ms");
    requireRange(config.kneeDb, 0.0, 24.0, path + ".knee_db");
    return config;
}

LimiterConfig parseLimiter(const JsonValue* value) {
    LimiterConfig config;
    if (value == nullptr) {
        return config;
    }
    const std::string path = "limiter";
    if (!value->isObject()) {
        invalid(path, "expected an object");
    }
    config.enabled = optionalBool(*value, "enabled", path, true);
    if (!config.enabled) {
        return config;
    }
    config.thresholdDbfs = requireNumber(*value, "threshold_dbfs", path);
    config.lookaheadMs = requireNumber(*value, "lookahead_ms", path);
    config.releaseMs = optionalNumber(*value, "release_ms", path, 80.0);
    requireRange(config.thresholdDbfs, -3.0, 0.0, path + ".threshold_dbfs");
    requireRange(config.lookaheadMs, 0.1, 50.0, path + ".lookahead_ms");
    requireRange(config.releaseMs, 5.0, 1000.0, path + ".release_ms");
    return config;
}

}  // namespace

AppConfig parseConfigFile(const std::filesystem::path& path,
                          const ConfigOverrides& overrides) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw AppError(ErrorCode::FileMissing,
                       "configuration file does not exist or cannot be opened: " +
                           path.string());
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw AppError(ErrorCode::FileCorrupt,
                       "failed while reading configuration file: " + path.string());
    }

    JsonParseError parseError;
    const JsonValue root = parseJson(contents.str(), parseError);
    if (!parseError.message.empty()) {
        std::ostringstream message;
        message << "JSON syntax error at line " << parseError.line << ", column "
                << parseError.column << ": " << parseError.message;
        throw AppError(ErrorCode::CfgSyntax, message.str());
    }
    if (!root.isObject()) {
        throw AppError(ErrorCode::CfgSyntax, "top-level JSON value must be an object");
    }

    AppConfig config;
    config.configPath = std::filesystem::absolute(path).lexically_normal();
    const std::filesystem::path base = config.configPath.parent_path();

    if (const JsonValue* sampleRate = optionalField(root, "sample_rate")) {
        const double rate = asNumber(*sampleRate, "sample_rate");
        if (rate != 48000.0) {
            invalid("sample_rate", "only 48000 Hz output is supported");
        }
    }

    config.output = resolvePath(base, asString(requireField(root, "output", ""), "output"));
    if (const JsonValue* report = optionalField(root, "report")) {
        config.report = resolvePath(base, asString(*report, "report"));
    } else {
        const auto stem = config.output.stem().u8string();
        config.report = config.output.parent_path() /
                        std::filesystem::u8path(std::string(stem.begin(), stem.end()) +
                                                ".report.json");
    }
    if (config.output == config.report) {
        invalid("report", "output and report paths must be different");
    }

    config.masterGainDb = optionalNumber(root, "master_gain_db", "", 0.0);
    requireRange(config.masterGainDb, -60.0, 12.0, "master_gain_db");

    if (const JsonValue* drc = optionalField(root, "drc")) {
        config.masterDrc = parseDrc(*drc, "drc");
    }
    config.limiter = parseLimiter(optionalField(root, "limiter"));

    if (overrides.logDir) {
        config.logDir = std::filesystem::absolute(*overrides.logDir).lexically_normal();
    } else if (const JsonValue* logDir = optionalField(root, "log_dir")) {
        config.logDir = resolvePath(base, asString(*logDir, "log_dir"));
    } else {
        config.logDir = base / "logs";
    }
    if (overrides.logFile) {
        config.logFile = *overrides.logFile;
    } else if (const JsonValue* logFile = optionalField(root, "log_file")) {
        config.logFile = asString(*logFile, "log_file");
    }
    if (config.logFile.empty() ||
        std::filesystem::u8path(config.logFile).filename().u8string() != config.logFile) {
        invalid("log_file", "must be a plain file name without path components");
    }

    const JsonValue* tracksValue = root.find("tracks");
    if (tracksValue == nullptr) {
        missing("tracks");
    }
    const JsonValue& tracks = *tracksValue;
    if (!tracks.isArray()) {
        invalid("tracks", "expected an array");
    }
    if (tracks.asArray().empty() || tracks.asArray().size() > 8) {
        invalid("tracks", "must contain between 1 and 8 tracks");
    }

    std::set<std::string> names;
    for (std::size_t index = 0; index < tracks.asArray().size(); ++index) {
        const JsonValue& node = tracks.asArray()[index];
        const std::string pathPrefix = "tracks[" + std::to_string(index) + "]";
        if (!node.isObject()) {
            invalid(pathPrefix, "expected an object");
        }
        TrackConfig track;
        track.name = asString(requireField(node, "name", pathPrefix), pathPrefix + ".name");
        if (!names.insert(track.name).second) {
            invalid(pathPrefix + ".name", "track names must be unique");
        }
        track.input = resolvePath(
            base, asString(requireField(node, "input", pathPrefix), pathPrefix + ".input"));
        track.gainDb = requireNumber(node, "gain_db", pathPrefix);
        requireRange(track.gainDb, -60.0, 12.0, pathPrefix + ".gain_db");
        track.mute = optionalBool(node, "mute", pathPrefix, false);
        track.solo = optionalBool(node, "solo", pathPrefix, false);
        if (const JsonValue* drc = optionalField(node, "drc")) {
            track.drc = parseDrc(*drc, pathPrefix + ".drc");
        }
        if (track.input == config.output) {
            invalid(pathPrefix + ".input", "input and output paths must be different");
        }
        config.tracks.push_back(std::move(track));
    }

    return config;
}

std::string configSchemaSummary() {
    return R"({
  "sample_rate": 48000,
  "output": "out/mix.wav",
  "report": "out/report.json",
  "log_dir": "logs",
  "master_gain_db": 0,
  "drc": {
    "enabled": true, "threshold_db": -18, "ratio": 4,
    "attack_ms": 10, "release_ms": 100, "knee_db": 6
  },
  "limiter": {
    "enabled": true, "threshold_dbfs": -1,
    "lookahead_ms": 10, "release_ms": 80
  },
  "tracks": [
    {
      "name": "navigation", "input": "nav.wav", "gain_db": -3,
      "mute": false, "solo": false,
      "drc": {
        "enabled": true, "threshold_db": -20, "ratio": 3,
        "attack_ms": 8, "release_ms": 120, "knee_db": 4
      }
    }
  ]
})";
}

}  // namespace automix
