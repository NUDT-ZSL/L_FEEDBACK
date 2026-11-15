#include "scene.h"

#include "json.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

namespace subdiv {
namespace {

bool readFile(const std::string& path, std::string& contents, std::string& error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "unable to open file";
        return false;
    }
    std::ostringstream stream;
    stream << file.rdbuf();
    if (!file.good() && !file.eof()) {
        error = "unable to read file";
        return false;
    }
    contents = stream.str();
    return true;
}

const JsonValue* getObject(const JsonValue& root, const std::string& key) {
    const JsonValue* value = root.find(key);
    if (value == nullptr || !value->isObject()) {
        return nullptr;
    }
    return value;
}

bool asInteger(const JsonValue& value, std::int64_t& output) {
    if (!value.isNumber() || !std::isfinite(value.number) ||
        std::floor(value.number) != value.number) {
        return false;
    }
    if (value.number < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
        value.number > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
        return false;
    }
    output = static_cast<std::int64_t>(value.number);
    return true;
}

std::string vecToString(const Vec3& value) {
    std::ostringstream stream;
    stream << '[' << value.x << ", " << value.y << ", " << value.z << ']';
    return stream.str();
}

void warnMissing(std::vector<std::string>& warnings,
                 const std::string& field,
                 const std::string& defaultValue) {
    warnings.push_back("field '" + field + "' missing; using default " + defaultValue);
}

void warnInvalid(std::vector<std::string>& warnings,
                 const std::string& field,
                 const std::string& problem,
                 const std::string& defaultValue) {
    warnings.push_back("field '" + field + "' " + problem + "; using default " + defaultValue);
}

Vec3 readVec3(const JsonValue& owner,
              const std::string& key,
              const std::string& field,
              const Vec3& defaultValue,
              float minValue,
              float maxValue,
              std::vector<std::string>& warnings) {
    const JsonValue* value = owner.find(key);
    if (value == nullptr) {
        warnMissing(warnings, field, vecToString(defaultValue));
        return defaultValue;
    }
    if (!value->isArray() || value->array.size() != 3) {
        warnInvalid(warnings, field, "must be an array of three numbers",
                    vecToString(defaultValue));
        return defaultValue;
    }

    Vec3 result = defaultValue;
    float* components[3] = {&result.x, &result.y, &result.z};
    const char* componentNames[3] = {"[0]", "[1]", "[2]"};
    for (int i = 0; i < 3; ++i) {
        const JsonValue& component = value->array[static_cast<std::size_t>(i)];
        if (!component.isNumber() || !std::isfinite(component.number) ||
            component.number < minValue || component.number > maxValue) {
            std::ostringstream range;
            range << "must be a finite number in [" << minValue << ", " << maxValue << ']';
            warnInvalid(warnings, field + componentNames[i], range.str(),
                        std::to_string(*components[i]));
            continue;
        }
        *components[i] = static_cast<float>(component.number);
    }
    return result;
}

std::pair<const JsonValue*, std::string> findAlias(
    const JsonValue& root,
    const std::string& primary,
    const std::string& secondary) {
    if (const JsonValue* value = root.find(primary)) {
        return {value, primary};
    }
    if (const JsonValue* value = root.find(secondary)) {
        return {value, secondary};
    }
    return {nullptr, primary};
}

}  // namespace

SceneLoadResult loadSceneFile(const std::string& path) {
    SceneLoadResult result;
    std::string text;
    std::string readError;
    const bool fileAvailable = readFile(path, text, readError);
    if (!fileAvailable) {
        result.warnings.push_back("cannot read scene file '" + path + "': " + readError +
                                  "; using all built-in defaults");
    }

    JsonValue root;
    std::string parseError;
    if (fileAvailable && !text.empty() && !parseJson(text, root, parseError)) {
        result.warnings.push_back("cannot parse scene file '" + path + "': " + parseError +
                                  "; using all built-in defaults");
        root = JsonValue{};
    } else if (fileAvailable && !text.empty() && !root.isObject()) {
        result.warnings.push_back("scene root must be a JSON object; using all built-in defaults");
        root = JsonValue{};
    }

    SceneSettings& settings = result.settings;

    const JsonValue* seedValue = root.find("seed");
    if (seedValue == nullptr) {
        warnMissing(result.warnings, "seed", std::to_string(settings.seed));
    } else {
        std::int64_t parsedSeed = 0;
        if (!asInteger(*seedValue, parsedSeed) || parsedSeed < 0) {
            warnInvalid(result.warnings, "seed", "must be a non-negative integer",
                        std::to_string(settings.seed));
        } else {
            settings.seed = static_cast<std::uint64_t>(parsedSeed);
        }
    }

    const JsonValue* material = getObject(root, "material");
    if (material == nullptr) {
        if (root.find("material") == nullptr) {
            warnMissing(result.warnings, "material.diffuse", vecToString(settings.diffuse));
            warnMissing(result.warnings, "material.roughness",
                        std::to_string(settings.roughness));
        } else {
            warnInvalid(result.warnings, "material", "must be an object",
                        "default material");
        }
    } else {
        if (material->find("diffuse") != nullptr) {
            settings.diffuse =
                readVec3(*material, "diffuse", "material.diffuse", settings.diffuse,
                         0.0f, 1.0f, result.warnings);
        } else if (material->find("diffuseColor") != nullptr) {
            settings.diffuse =
                readVec3(*material, "diffuseColor", "material.diffuse",
                         settings.diffuse, 0.0f, 1.0f, result.warnings);
        } else {
            warnMissing(result.warnings, "material.diffuse", vecToString(settings.diffuse));
        }
        const JsonValue* roughness = material->find("roughness");
        if (roughness == nullptr) {
            warnMissing(result.warnings, "material.roughness",
                        std::to_string(settings.roughness));
        } else if (!roughness->isNumber() || !std::isfinite(roughness->number) ||
                   roughness->number < 0.0 || roughness->number > 1.0) {
            warnInvalid(result.warnings, "material.roughness",
                        "must be a finite number in [0, 1]",
                        std::to_string(settings.roughness));
        } else {
            settings.roughness = static_cast<float>(roughness->number);
        }
    }

    const JsonValue* subdivision = getObject(root, "subdivision");
    const JsonValue* maxValue =
        subdivision != nullptr ? subdivision->find("maxLevel") : nullptr;
    std::string maxField = "subdivision.maxLevel";
    if (maxValue == nullptr) {
        maxValue = root.find("maxSubdivision");
        maxField = "maxSubdivision";
    }
    if (maxValue == nullptr) {
        maxField = "subdivision.maxLevel";
        warnMissing(result.warnings, maxField,
                    std::to_string(settings.maxSubdivision) + " (integer in [0, 6])");
    } else {
        std::int64_t parsedMax = 0;
        if (!asInteger(*maxValue, parsedMax) || parsedMax < 0 || parsedMax > 6) {
            warnInvalid(result.warnings, maxField, "must be an integer in [0, 6]",
                        std::to_string(settings.maxSubdivision));
        } else {
            settings.maxSubdivision = static_cast<int>(parsedMax);
        }
    }

    const JsonValue* initialValue =
        subdivision != nullptr ? subdivision->find("initialLevel") : nullptr;
    std::string initialField = "subdivision.initialLevel";
    if (initialValue == nullptr) {
        initialValue = root.find("initialSubdivision");
        initialField = "initialSubdivision";
    }
    if (initialValue == nullptr) {
        initialField = "subdivision.initialLevel";
        warnMissing(result.warnings, initialField,
                    std::to_string(settings.initialSubdivision));
    } else {
        std::int64_t parsedInitial = 0;
        if (!asInteger(*initialValue, parsedInitial) || parsedInitial < 0 ||
            parsedInitial > settings.maxSubdivision) {
            warnInvalid(result.warnings, initialField,
                        "must be an integer in [0, " +
                            std::to_string(settings.maxSubdivision) + "]",
                        std::to_string(std::min(settings.initialSubdivision,
                                                settings.maxSubdivision)));
        } else {
            settings.initialSubdivision = static_cast<int>(parsedInitial);
        }
    }
    settings.initialSubdivision =
        std::max(0, std::min(settings.initialSubdivision, settings.maxSubdivision));

    const auto creaseField = findAlias(root, "creaseWeights", "edgeCreaseWeights");
    if (creaseField.first == nullptr) {
        warnMissing(result.warnings, creaseField.second,
                    "deterministic built-in per-edge weights");
    } else if (creaseField.first->isArray()) {
        for (std::size_t i = 0; i < creaseField.first->array.size(); ++i) {
            const JsonValue& value = creaseField.first->array[i];
            const std::string field =
                creaseField.second + "[" + std::to_string(i) + "]";
            if (!value.isNumber() || !std::isfinite(value.number) ||
                value.number < 0.0 || value.number > 1.0) {
                warnInvalid(result.warnings, field,
                            "must be a finite number in [0, 1]",
                            "built-in weight for this edge");
                continue;
            }
            settings.creaseOverrides.emplace_back(
                static_cast<int>(i), static_cast<float>(value.number));
        }
    } else if (creaseField.first->isObject()) {
        for (const auto& member : creaseField.first->object) {
            const std::string field = creaseField.second + "." + member.first;
            char* end = nullptr;
            const long index = std::strtol(member.first.c_str(), &end, 10);
            if (end == nullptr || *end != '\0' || index < 0 ||
                index > std::numeric_limits<int>::max()) {
                warnInvalid(result.warnings, field,
                            "object keys must be non-negative edge indices",
                            "built-in edge weights");
                continue;
            }
            const JsonValue& value = member.second;
            if (!value.isNumber() || !std::isfinite(value.number) ||
                value.number < 0.0 || value.number > 1.0) {
                warnInvalid(result.warnings, field,
                            "must be a finite number in [0, 1]",
                            "built-in weight for this edge");
                continue;
            }
            settings.creaseOverrides.emplace_back(
                static_cast<int>(index), static_cast<float>(value.number));
        }
    } else {
        warnInvalid(result.warnings, creaseField.second,
                    "must be an array or an object keyed by edge index",
                    "deterministic built-in per-edge weights");
    }

    const JsonValue* camera = getObject(root, "camera");
    if (camera == nullptr) {
        if (root.find("camera") == nullptr) {
            warnMissing(result.warnings, "camera.position", vecToString(settings.camera.position));
            warnMissing(result.warnings, "camera.target", vecToString(settings.camera.target));
            warnMissing(result.warnings, "camera.up", vecToString(settings.camera.up));
            warnMissing(result.warnings, "camera.fovDegrees",
                        std::to_string(settings.camera.fovDegrees));
        } else {
            warnInvalid(result.warnings, "camera", "must be an object", "default camera");
        }
    } else {
        settings.camera.position =
            readVec3(*camera, "position", "camera.position", settings.camera.position,
                     -1000.0f, 1000.0f, result.warnings);
        settings.camera.target =
            readVec3(*camera, "target", "camera.target", settings.camera.target,
                     -1000.0f, 1000.0f, result.warnings);
        settings.camera.up =
            readVec3(*camera, "up", "camera.up", settings.camera.up,
                     -1.0e6f, 1.0e6f, result.warnings);
        if (lengthSquared(settings.camera.up) < 1.0e-12f) {
            warnInvalid(result.warnings, "camera.up", "must not be a zero vector",
                        vecToString(CameraSettings{}.up));
            settings.camera.up = CameraSettings{}.up;
        }
        const JsonValue* fov = camera->find("fovDegrees");
        std::string fovField = "camera.fovDegrees";
        if (fov == nullptr) {
            fov = camera->find("fov");
            fovField = "camera.fov";
        }
        if (fov == nullptr) {
            warnMissing(result.warnings, "camera.fovDegrees",
                        std::to_string(settings.camera.fovDegrees));
        } else if (!fov->isNumber() || !std::isfinite(fov->number) ||
                   fov->number < 20.0 || fov->number > 100.0) {
            warnInvalid(result.warnings, fovField,
                        "must be a finite number in [20, 100]",
                        std::to_string(settings.camera.fovDegrees));
        } else {
            settings.camera.fovDegrees = static_cast<float>(fov->number);
        }
    }

    return result;
}

}  // namespace subdiv

