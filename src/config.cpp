#include "config.h"
#include "error.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <cmath>
#include <initializer_list>
#include <sstream>

namespace df {
namespace {

using Json = nlohmann::json;

void requireObject(const Json& node, const char* section) {
    if (!node.is_object()) {
        throw AppError(ErrorCode::E_CONFIG_PARSE,
                       std::string("section '") + section + "' must be an object");
    }
}

void rejectUnknown(const Json& node, const char* section,
                   std::initializer_list<const char*> allowed) {
    for (const auto& item : node.items()) {
        bool known = false;
        for (const char* key : allowed) {
            if (item.key() == key) {
                known = true;
                break;
            }
        }
        if (!known) {
            throw AppError(ErrorCode::E_CONFIG_PARSE,
                           std::string("unknown key '") + item.key() + "' in '" + section + "'");
        }
    }
}

int requireInt(const Json& node, const char* name, int lo, int hi) {
    if (!node.contains(name) || !node.at(name).is_number_integer()) {
        throw AppError(ErrorCode::E_CONFIG_PARSE,
                       std::string("'") + name + "' must be an integer");
    }
    const int value = node.at(name).get<int>();
    if (value < lo || value > hi) {
        std::ostringstream out;
        out << "'" << name << "' must be in [" << lo << ", " << hi << "]";
        throw AppError(ErrorCode::E_CONFIG_PARSE, out.str());
    }
    return value;
}

float requireFloat(const Json& node, const char* name, float lo, float hi) {
    if (!node.contains(name) || !node.at(name).is_number()) {
        throw AppError(ErrorCode::E_CONFIG_PARSE,
                       std::string("'") + name + "' must be a number");
    }
    const float value = node.at(name).get<float>();
    if (!std::isfinite(value) || value < lo || value > hi) {
        std::ostringstream out;
        out << "'" << name << "' must be finite and in [" << lo << ", " << hi << "]";
        throw AppError(ErrorCode::E_CONFIG_PARSE, out.str());
    }
    return value;
}

bool requireBool(const Json& node, const char* name) {
    if (!node.contains(name) || !node.at(name).is_boolean()) {
        throw AppError(ErrorCode::E_CONFIG_PARSE,
                       std::string("'") + name + "' must be a boolean");
    }
    return node.at(name).get<bool>();
}

} // namespace

RenderConfig loadConfig(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw AppError(ErrorCode::E_CONFIG_PARSE, "cannot open config file: " + path);
    }

    Json root;
    try {
        input >> root;
    } catch (const std::exception& ex) {
        throw AppError(ErrorCode::E_CONFIG_PARSE, "invalid JSON: " + std::string(ex.what()));
    }

    if (!root.is_object()) {
        throw AppError(ErrorCode::E_CONFIG_PARSE, "root must be an object");
    }
    rejectUnknown(root, "root", {"render", "volumetric", "transparent", "lights"});

    RenderConfig config;

    if (root.contains("render")) {
        const Json& render = root.at("render");
        requireObject(render, "render");
        rejectUnknown(render, "render", {"width", "height", "offscreenSamples", "maxFrames"});
        if (render.contains("width")) config.width = requireInt(render, "width", 64, 4096);
        if (render.contains("height")) config.height = requireInt(render, "height", 64, 4096);
        if (render.contains("offscreenSamples")) {
            config.offscreenSamples = requireInt(render, "offscreenSamples", 1, 16);
        }
        if (render.contains("maxFrames")) {
            config.maxFrames = requireInt(render, "maxFrames", 0, 100000);
        }
    }

    if (root.contains("volumetric")) {
        const Json& fog = root.at("volumetric");
        requireObject(fog, "volumetric");
        rejectUnknown(fog, "volumetric",
                      {"enabled", "steps", "densityScale", "maxDistance",
                       "noiseScale", "absorption", "scattering"});
        if (fog.contains("enabled")) config.volumetric.enabled = requireBool(fog, "enabled");
        if (fog.contains("steps")) config.volumetric.steps = requireInt(fog, "steps", 16, 64);
        if (fog.contains("densityScale")) {
            config.volumetric.densityScale = requireFloat(fog, "densityScale", 0.0f, 2.0f);
        }
        if (fog.contains("maxDistance")) {
            config.volumetric.maxDistance = requireFloat(fog, "maxDistance", 1.0f, 200.0f);
        }
        if (fog.contains("noiseScale")) {
            config.volumetric.noiseScale = requireFloat(fog, "noiseScale", 0.001f, 2.0f);
        }
        if (fog.contains("absorption")) {
            config.volumetric.absorption = requireFloat(fog, "absorption", 0.0f, 8.0f);
        }
        if (fog.contains("scattering")) {
            config.volumetric.scattering = requireFloat(fog, "scattering", 0.0f, 8.0f);
        }
    }

    if (root.contains("transparent")) {
        const Json& transparent = root.at("transparent");
        requireObject(transparent, "transparent");
        rejectUnknown(transparent, "transparent",
                      {"sortingEnabled", "exactSortLimit", "bucketCount"});
        if (transparent.contains("sortingEnabled")) {
            config.transparent.sortingEnabled = requireBool(transparent, "sortingEnabled");
        }
        if (transparent.contains("exactSortLimit")) {
            config.transparent.exactSortLimit =
                requireInt(transparent, "exactSortLimit", 1, 1000000);
        }
        if (transparent.contains("bucketCount")) {
            config.transparent.bucketCount =
                requireInt(transparent, "bucketCount", 16, 16384);
        }
    }

    if (root.contains("lights")) {
        const Json& lights = root.at("lights");
        requireObject(lights, "lights");
        rejectUnknown(lights, "lights", {"maxLights"});
        if (lights.contains("maxLights")) {
            config.lights.maxLights = requireInt(lights, "maxLights", 1, 32);
        }
    }

    return config;
}

int effectiveFogSteps(const RenderConfig& config, bool offline) {
    return offline ? 32 : config.volumetric.steps;
}

} // namespace df

