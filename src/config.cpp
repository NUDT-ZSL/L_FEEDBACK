#include "config.hpp"

#include "error.hpp"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string_view>

#include <glm/geometric.hpp>

namespace preview {
namespace {

using json::Value;

constexpr int kMaxDimension = 16384;

[[noreturn]] void configError(const std::string& message) {
    throw AppError(ErrorCode::Config, message);
}

std::filesystem::path pathFromUtf8(const std::string& value) {
    return std::filesystem::u8path(value);
}

const Value& requireMember(
    const Value& object,
    std::string_view key,
    std::string_view context) {
    if (!object.isObject()) {
        configError(std::string(context) + " must be an object");
    }
    const auto it = object.object.find(std::string(key));
    if (it == object.object.end()) {
        configError(
            "missing required field '" + std::string(context) + "." +
            std::string(key) + "'");
    }
    return it->second;
}

const Value* optionalMember(const Value& object, const std::string& key) {
    const auto it = object.object.find(key);
    return it == object.object.end() ? nullptr : &it->second;
}

float readFloat(const Value& value, const std::string& context) {
    if (!value.isNumber() || !std::isfinite(value.number)) {
        configError(context + " must be a finite number");
    }
    return static_cast<float>(value.number);
}

float readFiniteFloat(const Value& value, const std::string& context) {
    const float result = readFloat(value, context);
    if (!std::isfinite(result)) {
        configError(context + " is outside the supported numeric range");
    }
    return result;
}

int readPositiveInt(const Value& value, const std::string& context, int maximum) {
    if (!value.isNumber() || value.number != std::floor(value.number)) {
        configError(context + " must be an integer");
    }
    if (value.number <= 0.0 || value.number > static_cast<double>(maximum)) {
        configError(context + " must be in the range 1.." + std::to_string(maximum));
    }
    return static_cast<int>(value.number);
}

glm::vec3 readVec3(const Value& value, const std::string& context) {
    if (!value.isArray() || value.array.size() != 3) {
        configError(context + " must be an array of three numbers");
    }
    return glm::vec3(
        readFiniteFloat(value.array[0], context + "[0]"),
        readFiniteFloat(value.array[1], context + "[1]"),
        readFiniteFloat(value.array[2], context + "[2]"));
}

glm::vec3 readColor(const Value& value, const std::string& context) {
    const glm::vec3 color = readVec3(value, context);
    for (int component = 0; component < 3; ++component) {
        if (color[component] < 0.0f || color[component] > 1.0f) {
            configError(context + " components must be in the range 0..1");
        }
    }
    return color;
}

std::string readNonEmptyString(const Value& value, const std::string& context) {
    if (!value.isString() || value.string.empty()) {
        configError(context + " must be a non-empty string");
    }
    return value.string;
}

std::filesystem::path resolvePath(
    const std::filesystem::path& baseDirectory,
    const std::string& value) {
    const std::filesystem::path path = pathFromUtf8(value);
    return path.is_absolute() ? path : baseDirectory / path;
}

bool readBoolean(const Value& value, const std::string& context) {
    if (!value.isBoolean()) {
        configError(context + " must be a boolean");
    }
    return value.boolean;
}

Attenuation readAttenuation(const Value& value, const std::string& context) {
    if (!value.isObject()) {
        configError(context + " must be an object");
    }
    Attenuation result;
    if (const Value* constant = optionalMember(value, "constant")) {
        result.constant = readFiniteFloat(*constant, context + ".constant");
    }
    if (const Value* linear = optionalMember(value, "linear")) {
        result.linear = readFiniteFloat(*linear, context + ".linear");
    }
    if (const Value* quadratic = optionalMember(value, "quadratic")) {
        result.quadratic = readFiniteFloat(*quadratic, context + ".quadratic");
    }
    if (result.constant <= 0.0f || result.linear < 0.0f || result.quadratic < 0.0f) {
        configError(context + " requires constant > 0 and non-negative linear/quadratic terms");
    }
    return result;
}

LightConfig readLight(const Value& value, const std::string& context) {
    if (!value.isObject()) {
        configError(context + " must be an object");
    }
    LightConfig light;
    const std::string type = readNonEmptyString(
        requireMember(value, "type", context), context + ".type");
    if (type == "directional") {
        light.type = LightConfig::Type::Directional;
        light.direction = readVec3(
            requireMember(value, "direction", context), context + ".direction");
        if (glm::length(light.direction) <= 1.0e-8f) {
            configError(context + ".direction must not be zero");
        }
        light.direction = glm::normalize(light.direction);
    } else if (type == "point") {
        light.type = LightConfig::Type::Point;
        light.position = readVec3(
            requireMember(value, "position", context), context + ".position");
        if (const Value* attenuation = optionalMember(value, "attenuation")) {
            light.attenuation = readAttenuation(*attenuation, context + ".attenuation");
        }
    } else {
        configError(context + ".type must be 'directional' or 'point'");
    }

    light.color = readColor(requireMember(value, "color", context), context + ".color");
    light.intensity = readFiniteFloat(
        requireMember(value, "intensity", context), context + ".intensity");
    if (light.intensity < 0.0f) {
        configError(context + ".intensity must be non-negative");
    }
    return light;
}

CameraConfig readCamera(const Value& value) {
    if (!value.isObject()) {
        configError("camera must be an object");
    }
    CameraConfig camera;
    camera.position = readVec3(
        requireMember(value, "position", "camera"), "camera.position");
    camera.target = readVec3(
        requireMember(value, "target", "camera"), "camera.target");
    camera.up = readVec3(
        requireMember(value, "up", "camera"), "camera.up");
    camera.verticalFovDegrees = readFiniteFloat(
        requireMember(value, "vertical_fov_degrees", "camera"),
        "camera.vertical_fov_degrees");
    camera.nearPlane = readFiniteFloat(
        requireMember(value, "near", "camera"), "camera.near");
    camera.farPlane = readFiniteFloat(
        requireMember(value, "far", "camera"), "camera.far");

    if (camera.verticalFovDegrees <= 0.0f || camera.verticalFovDegrees >= 180.0f) {
        configError("camera.vertical_fov_degrees must be greater than 0 and less than 180");
    }
    if (camera.nearPlane <= 0.0f || camera.farPlane <= 0.0f ||
        camera.nearPlane >= camera.farPlane) {
        configError("camera requires 0 < near < far");
    }
    if (glm::length(camera.up) <= 1.0e-8f) {
        configError("camera.up must not be zero");
    }
    const glm::vec3 viewDirection = camera.target - camera.position;
    if (glm::length(viewDirection) <= 1.0e-8f) {
        configError("camera.position and camera.target must differ");
    }
    if (glm::length(glm::cross(glm::normalize(viewDirection), glm::normalize(camera.up))) <=
        1.0e-6f) {
        configError("camera.up must not be parallel to the view direction");
    }
    return camera;
}

} // namespace

RenderConfig loadConfig(const std::filesystem::path& configPath) {
    std::ifstream input(configPath, std::ios::binary);
    if (!input) {
        configError("cannot open config file: " + configPath.u8string());
    }
    const std::string text{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};

    Value root;
    try {
        root = json::parse(text);
    } catch (const json::ParseError& error) {
        configError(error.what());
    }
    if (!root.isObject()) {
        configError("config root must be an object");
    }

    RenderConfig config;
    config.width = readPositiveInt(
        requireMember(root, "width", "config"), "width", kMaxDimension);
    config.height = readPositiveInt(
        requireMember(root, "height", "config"), "height", kMaxDimension);
    config.objPath = resolvePath(
        configPath.parent_path(),
        readNonEmptyString(requireMember(root, "obj_path", "config"), "obj_path"));
    config.outputPath = resolvePath(
        configPath.parent_path(),
        readNonEmptyString(requireMember(root, "output_path", "config"), "output_path"));
    config.camera = readCamera(requireMember(root, "camera", "config"));

    if (const Value* doubleSided = optionalMember(root, "double_sided")) {
        config.doubleSided = readBoolean(*doubleSided, "double_sided");
    }
    if (const Value* ambient = optionalMember(root, "ambient")) {
        config.ambient = readColor(*ambient, "ambient");
    }
    if (const Value* background = optionalMember(root, "background")) {
        config.background = readColor(*background, "background");
    }

    const Value& lights = requireMember(root, "lights", "config");
    if (!lights.isArray() || lights.array.empty()) {
        configError("lights must be a non-empty array");
    }
    config.lights.reserve(lights.array.size());
    for (std::size_t i = 0; i < lights.array.size(); ++i) {
        config.lights.push_back(readLight(lights.array[i], "lights[" + std::to_string(i) + "]"));
    }

    return config;
}

} // namespace preview
