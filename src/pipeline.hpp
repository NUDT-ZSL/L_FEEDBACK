#pragma once

#include "math.hpp"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace scene {

enum class StageType {
    Geometry,
    Shadow,
    Deferred,
    PostProcess
};

const char* stageName(StageType type);
StageType parseStageType(const std::string& name);

struct StageEntry {
    StageType type = StageType::Geometry;
    bool enabled = true;
};

struct ShadowSettings {
    int resolution = 1024;
    float bias = 0.0012f;
    float strength = 0.88f;
};

struct DeferredSettings {
    Vec3 ambient{0.090f, 0.100f, 0.130f};
};

struct EffectConfig {
    std::string type;
    bool enabled = true;
    nlohmann::json params = nlohmann::json::object();
};

struct PipelineConfig {
    static constexpr int kFormatVersion = 1;

    int version = kFormatVersion;
    Vec3 clearColor{0.012f, 0.018f, 0.030f};
    std::vector<StageEntry> stages;
    ShadowSettings shadow;
    DeferredSettings deferred;
    std::vector<EffectConfig> postEffects;

    bool hasStage(StageType type) const;
    bool hasEnabledStage(StageType type) const;
    const StageEntry* findStage(StageType type) const;
    const StageEntry* findEnabledStage(StageType type) const;

    nlohmann::json toJson() const;
    static PipelineConfig fromJson(const nlohmann::json& value,
                                   const std::string& sourceLabel = "<memory>");
    static PipelineConfig load(const std::string& path);
    void save(const std::string& path) const;
};

} // namespace scene

