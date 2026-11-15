#include "pipeline.hpp"

#include "error.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <set>
#include <sstream>

namespace scene {
namespace {

using json = nlohmann::json;

[[noreturn]] void configError(const std::string& label, const std::string& message) {
    throw AppError(ErrorType::Config, label + ": " + message);
}

void rejectUnknownKeys(const json& object,
                       std::initializer_list<const char*> allowed,
                       const std::string& label,
                       const std::string& context) {
    if (!object.is_object()) {
        configError(label, context + " must be a JSON object");
    }
    for (auto it = object.begin(); it != object.end(); ++it) {
        const bool known = std::any_of(
            allowed.begin(), allowed.end(),
            [&](const char* key) { return it.key() == key; });
        if (!known) {
            configError(label, context + " contains unknown key '" + it.key() + "'");
        }
    }
}

bool readBool(const json& object, const char* key, bool fallback,
              const std::string& label, const std::string& context) {
    const auto it = object.find(key);
    if (it == object.end()) {
        return fallback;
    }
    if (!it->is_boolean()) {
        configError(label, context + "." + key + " must be a boolean");
    }
    return it->get<bool>();
}

int readInt(const json& object, const char* key, int fallback, int minimum,
            int maximum, const std::string& label, const std::string& context) {
    const auto it = object.find(key);
    if (it == object.end()) {
        return fallback;
    }
    if (!it->is_number_integer()) {
        configError(label, context + "." + key + " must be an integer");
    }
    const int value = it->get<int>();
    if (value < minimum || value > maximum) {
        configError(label, context + "." + key + " must be in [" +
                               std::to_string(minimum) + ", " +
                               std::to_string(maximum) + "]");
    }
    return value;
}

float readFloat(const json& object, const char* key, float fallback,
                float minimum, float maximum, const std::string& label,
                const std::string& context) {
    const auto it = object.find(key);
    if (it == object.end()) {
        return fallback;
    }
    if (!it->is_number()) {
        configError(label, context + "." + key + " must be a number");
    }
    const float value = it->get<float>();
    if (!std::isfinite(value) || value < minimum || value > maximum) {
        configError(label, context + "." + key + " must be finite and in [" +
                               std::to_string(minimum) + ", " +
                               std::to_string(maximum) + "]");
    }
    return value;
}

Vec3 readVec3(const json& object, const char* key, Vec3 fallback,
              float minimum, float maximum, const std::string& label,
              const std::string& context) {
    const auto it = object.find(key);
    if (it == object.end()) {
        return fallback;
    }
    if (!it->is_array() || it->size() != 3) {
        configError(label, context + "." + key + " must be an array of 3 numbers");
    }
    Vec3 value{};
    float* components[3] = {&value.x, &value.y, &value.z};
    for (size_t i = 0; i < 3; ++i) {
        if (!(*it)[i].is_number()) {
            configError(label, context + "." + key + " must contain only numbers");
        }
        *components[i] = (*it)[i].get<float>();
        if (!std::isfinite(*components[i]) ||
            *components[i] < minimum || *components[i] > maximum) {
            configError(label, context + "." + key + " components must be in [" +
                                   std::to_string(minimum) + ", " +
                                   std::to_string(maximum) + "]");
        }
    }
    return value;
}

json vec3ToJson(Vec3 value) {
    return json::array({value.x, value.y, value.z});
}

EffectConfig parseEffect(const json& value, size_t index, const std::string& label) {
    const std::string context = "stages[postprocess].effects[" +
                                std::to_string(index) + "]";
    if (!value.is_object()) {
        configError(label, context + " must be an object");
    }
    const auto typeIt = value.find("type");
    if (typeIt == value.end() || !typeIt->is_string()) {
        configError(label, context + ".type must be a string");
    }

    EffectConfig effect;
    effect.type = typeIt->get<std::string>();
    effect.enabled = readBool(value, "enabled", true, label, context);

    if (effect.type == "bloom") {
        rejectUnknownKeys(value, {"type", "enabled", "threshold", "intensity",
                                  "radius", "knee"}, label, context);
        effect.params = {
            {"threshold", readFloat(value, "threshold", 0.9f, 0.0f, 100.0f, label, context)},
            {"intensity", readFloat(value, "intensity", 0.65f, 0.0f, 10.0f, label, context)},
            {"radius", readFloat(value, "radius", 1.0f, 0.1f, 8.0f, label, context)},
            {"knee", readFloat(value, "knee", 0.35f, 0.0f, 1.0f, label, context)}
        };
    } else if (effect.type == "tonemap") {
        rejectUnknownKeys(value, {"type", "enabled", "mode", "exposure"},
                          label, context);
        std::string mode = "aces";
        const auto modeIt = value.find("mode");
        if (modeIt != value.end()) {
            if (!modeIt->is_string()) {
                configError(label, context + ".mode must be a string");
            }
            mode = modeIt->get<std::string>();
        }
        if (mode != "aces" && mode != "reinhard") {
            configError(label, context + ".mode must be 'aces' or 'reinhard'");
        }
        effect.params = {
            {"mode", mode},
            {"exposure", readFloat(value, "exposure", 1.0f, 0.0f, 16.0f, label, context)}
        };
    } else if (effect.type == "fxaa") {
        rejectUnknownKeys(value, {"type", "enabled", "edge_threshold"},
                          label, context);
        effect.params = {
            {"edge_threshold", readFloat(value, "edge_threshold", 0.08f,
                                         0.0f, 1.0f, label, context)}
        };
    } else if (effect.type == "color_grade") {
        rejectUnknownKeys(value, {"type", "enabled", "contrast", "saturation",
                                  "tint", "lift"}, label, context);
        effect.params = {
            {"contrast", readFloat(value, "contrast", 1.04f, 0.0f, 4.0f, label, context)},
            {"saturation", readFloat(value, "saturation", 1.06f, 0.0f, 4.0f, label, context)},
            {"tint", vec3ToJson(readVec3(value, "tint", {1.0f, 1.0f, 1.0f},
                                        0.0f, 4.0f, label, context))},
            {"lift", vec3ToJson(readVec3(value, "lift", {0.0f, 0.0f, 0.0f},
                                        -1.0f, 1.0f, label, context))}
        };
    } else {
        throw AppError(ErrorType::UnknownEffect,
                       label + ": " + context + " has unknown effect type '" +
                       effect.type + "'");
    }
    return effect;
}

} // namespace

const char* stageName(StageType type) {
    switch (type) {
    case StageType::Geometry: return "geometry";
    case StageType::Shadow: return "shadow";
    case StageType::Deferred: return "deferred";
    case StageType::PostProcess: return "postprocess";
    }
    return "unknown";
}

StageType parseStageType(const std::string& name) {
    if (name == "geometry") return StageType::Geometry;
    if (name == "shadow") return StageType::Shadow;
    if (name == "deferred") return StageType::Deferred;
    if (name == "postprocess") return StageType::PostProcess;
    throw AppError(ErrorType::UnknownStage, "unknown pipeline stage '" + name + "'");
}

bool PipelineConfig::hasStage(StageType type) const {
    return findStage(type) != nullptr;
}

bool PipelineConfig::hasEnabledStage(StageType type) const {
    return findEnabledStage(type) != nullptr;
}

const StageEntry* PipelineConfig::findStage(StageType type) const {
    for (const auto& stage : stages) {
        if (stage.type == type) {
            return &stage;
        }
    }
    return nullptr;
}

const StageEntry* PipelineConfig::findEnabledStage(StageType type) const {
    for (const auto& stage : stages) {
        if (stage.type == type && stage.enabled) {
            return &stage;
        }
    }
    return nullptr;
}

nlohmann::json PipelineConfig::toJson() const {
    json result = {
        {"version", version},
        {"clear_color", vec3ToJson(clearColor)},
        {"stages", json::array()}
    };

    for (const auto& stage : stages) {
        json entry = {
            {"name", stageName(stage.type)},
            {"enabled", stage.enabled}
        };
        if (stage.type == StageType::Shadow) {
            entry["resolution"] = shadow.resolution;
            entry["bias"] = shadow.bias;
            entry["strength"] = shadow.strength;
        } else if (stage.type == StageType::Deferred) {
            entry["ambient"] = vec3ToJson(deferred.ambient);
        } else if (stage.type == StageType::PostProcess) {
            entry["effects"] = json::array();
            for (const auto& effect : postEffects) {
                json effectJson = effect.params;
                effectJson["type"] = effect.type;
                effectJson["enabled"] = effect.enabled;
                entry["effects"].push_back(effectJson);
            }
        }
        result["stages"].push_back(entry);
    }
    return result;
}

PipelineConfig PipelineConfig::fromJson(const json& value,
                                        const std::string& sourceLabel) {
    if (!value.is_object()) {
        configError(sourceLabel, "top-level JSON value must be an object");
    }
    rejectUnknownKeys(value, {"version", "clear_color", "stages"},
                      sourceLabel, "top-level object");

    const auto versionIt = value.find("version");
    if (versionIt == value.end()) {
        throw AppError(ErrorType::StateVersion,
                       sourceLabel + ": required version field is missing");
    }
    if (!versionIt->is_number_integer()) {
        throw AppError(ErrorType::StateVersion,
                       sourceLabel + ": version must be an integer");
    }
    const int version = versionIt->get<int>();
    if (version != kFormatVersion) {
        throw AppError(ErrorType::StateVersion,
                       sourceLabel + ": unsupported version " +
                       std::to_string(version) + "; expected " +
                       std::to_string(kFormatVersion));
    }

    PipelineConfig config;
    config.version = version;
    config.clearColor = readVec3(value, "clear_color",
                                 {0.012f, 0.018f, 0.030f},
                                 0.0f, 1.0f, sourceLabel, "top-level");

    const auto stagesIt = value.find("stages");
    if (stagesIt == value.end() || !stagesIt->is_array()) {
        configError(sourceLabel, "stages must be an array");
    }

    std::set<StageType> seen;
    for (size_t i = 0; i < stagesIt->size(); ++i) {
        const json& stage = (*stagesIt)[i];
        const std::string context = "stages[" + std::to_string(i) + "]";
        if (!stage.is_object()) {
            configError(sourceLabel, context + " must be an object");
        }
        const auto nameIt = stage.find("name");
        if (nameIt == stage.end() || !nameIt->is_string()) {
            configError(sourceLabel, context + ".name must be a string");
        }
        const std::string name = nameIt->get<std::string>();
        StageType type;
        try {
            type = parseStageType(name);
        } catch (const AppError& error) {
            throw AppError(error.type(), sourceLabel + ": " + error.what());
        }
        if (!seen.insert(type).second) {
            configError(sourceLabel, "duplicate pipeline stage '" + name + "'");
        }

        StageEntry entry;
        entry.type = type;
        entry.enabled = readBool(stage, "enabled", true, sourceLabel, context);

        if (type == StageType::Geometry) {
            rejectUnknownKeys(stage, {"name", "enabled"}, sourceLabel, context);
        } else if (type == StageType::Shadow) {
            rejectUnknownKeys(stage, {"name", "enabled", "resolution", "bias",
                                      "strength"}, sourceLabel, context);
            config.shadow.resolution = readInt(stage, "resolution", 1024,
                                               256, 4096, sourceLabel, context);
            config.shadow.bias = readFloat(stage, "bias", 0.0012f,
                                           1.0e-6f, 0.05f, sourceLabel, context);
            config.shadow.strength = readFloat(stage, "strength", 0.88f,
                                               0.0f, 1.0f, sourceLabel, context);
        } else if (type == StageType::Deferred) {
            rejectUnknownKeys(stage, {"name", "enabled", "ambient"},
                              sourceLabel, context);
            config.deferred.ambient = readVec3(
                stage, "ambient", {0.090f, 0.100f, 0.130f},
                0.0f, 4.0f, sourceLabel, context);
        } else if (type == StageType::PostProcess) {
            rejectUnknownKeys(stage, {"name", "enabled", "effects"},
                              sourceLabel, context);
            const auto effectsIt = stage.find("effects");
            if (effectsIt == stage.end() || !effectsIt->is_array()) {
                configError(sourceLabel, context + ".effects must be an array");
            }
            for (size_t effectIndex = 0; effectIndex < effectsIt->size(); ++effectIndex) {
                config.postEffects.push_back(
                    parseEffect((*effectsIt)[effectIndex], effectIndex, sourceLabel));
            }
        }
        config.stages.push_back(entry);
    }

    return config;
}

PipelineConfig PipelineConfig::load(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw AppError(ErrorType::FileIo,
                       "cannot open pipeline JSON '" + path + "'");
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw AppError(ErrorType::FileIo,
                       "failed while reading pipeline JSON '" + path + "'");
    }

    json value;
    try {
        value = json::parse(buffer.str());
    } catch (const json::parse_error& error) {
        configError(path, std::string("JSON parse error at byte ") +
                            std::to_string(error.byte) + ": " + error.what());
    }
    return fromJson(value, path);
}

void PipelineConfig::save(const std::string& path) const {
    const std::filesystem::path target(path);
    std::error_code ec;
    if (!target.parent_path().empty()) {
        std::filesystem::create_directories(target.parent_path(), ec);
        if (ec) {
            throw AppError(ErrorType::FileIo,
                           "cannot create state directory '" +
                           target.parent_path().string() + "': " + ec.message());
        }
    }

    const std::filesystem::path temporary = target.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw AppError(ErrorType::FileIo,
                           "cannot open state output '" + temporary.string() + "'");
        }
        output << toJson().dump(2) << '\n';
        output.flush();
        if (!output) {
            throw AppError(ErrorType::FileIo,
                           "failed while writing state output '" +
                           temporary.string() + "'");
        }
    }

    std::filesystem::rename(temporary, target, ec);
    if (ec) {
        std::error_code removeError;
        std::filesystem::remove(target, removeError);
        ec.clear();
        std::filesystem::rename(temporary, target, ec);
        if (ec) {
            std::filesystem::remove(temporary);
            throw AppError(ErrorType::FileIo,
                           "cannot replace state file '" + path + "': " +
                           ec.message());
        }
    }
}

} // namespace scene

