#include "config.h"

#include "json.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace ct {
namespace {

using json::Value;

bool readTextFile(const std::string& path, std::string& text, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open config file '" + path + "'";
        return false;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        error = "failed while reading config file '" + path + "'";
        return false;
    }
    text = buffer.str();
    return true;
}

const Value& requireObject(const Value& parent, const char* key, const char* context,
                           std::string& error) {
    const Value* value = parent.find(key);
    if (value == nullptr) {
        error = std::string(context) + " is missing required object '" + key + "'";
        return parent;
    }
    if (!value->isObject()) {
        error = std::string(context) + "." + key + " must be an object";
        return parent;
    }
    return *value;
}

bool getNumber(const Value& object, const char* key, double& result, const std::string& context,
               std::string& error, bool required = true) {
    const Value* value = object.find(key);
    if (value == nullptr) {
        if (required) {
            error = context + " is missing required number '" + key + "'";
            return false;
        }
        return true;
    }
    if (!value->isNumber() || !std::isfinite(value->asNumber())) {
        error = context + "." + key + " must be a finite number";
        return false;
    }
    result = value->asNumber();
    return true;
}

bool getInteger(const Value& object, const char* key, int& result, const std::string& context,
                std::string& error, bool required = true) {
    double number_value = 0.0;
    if (!getNumber(object, key, number_value, context, error, required)) {
        return false;
    }
    const Value* value = object.find(key);
    if (value == nullptr) {
        return true;
    }
    if (number_value < static_cast<double>(std::numeric_limits<int>::min()) ||
        number_value > static_cast<double>(std::numeric_limits<int>::max()) ||
        std::floor(number_value) != number_value) {
        error = context + "." + key + " must be an integer";
        return false;
    }
    result = static_cast<int>(number_value);
    return true;
}

bool getUnsigned(const Value& object, const char* key, std::uint32_t& result,
                 const std::string& context, std::string& error) {
    double number_value = 0.0;
    if (!getNumber(object, key, number_value, context, error)) {
        return false;
    }
    if (number_value < 0.0 || number_value > 4294967295.0 ||
        std::floor(number_value) != number_value) {
        error = context + "." + key + " must be an integer in [0, 4294967295]";
        return false;
    }
    result = static_cast<std::uint32_t>(number_value);
    return true;
}

bool getString(const Value& object, const char* key, std::string& result,
               const std::string& context, std::string& error) {
    const Value* value = object.find(key);
    if (value == nullptr) {
        error = context + " is missing required string '" + key + "'";
        return false;
    }
    if (!value->isString() || value->asString().empty()) {
        error = context + "." + key + " must be a non-empty string";
        return false;
    }
    result = value->asString();
    return true;
}

bool getVec3(const Value& object, const char* key, Vec3& result, const std::string& context,
             std::string& error) {
    const Value* value = object.find(key);
    if (value == nullptr) {
        error = context + " is missing required vector '" + key + "'";
        return false;
    }
    if (!value->isArray() || value->asArray().size() != 3) {
        error = context + "." + key + " must be an array of exactly 3 numbers";
        return false;
    }
    double components[3]{};
    for (std::size_t i = 0; i < 3; ++i) {
        const Value& component = value->asArray()[i];
        if (!component.isNumber() || !std::isfinite(component.asNumber())) {
            error = context + "." + key + " components must be finite numbers";
            return false;
        }
        components[i] = component.asNumber();
    }
    result = {components[0], components[1], components[2]};
    return true;
}

bool validColor(const Vec3& color) {
    return color.x >= 0.0 && color.y >= 0.0 && color.z >= 0.0 && color.x <= 16.0 &&
           color.y <= 16.0 && color.z <= 16.0;
}

bool hasExtension(const std::string& path, const char* extension) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == extension;
}

}  // namespace

bool loadConfig(const std::string& path, AppConfig& config, std::string& error) {
    error.clear();
    std::string text;
    if (!readTextFile(path, text, error)) {
        return false;
    }

    const Value root = Value::parse(text, error);
    if (!error.empty()) {
        error = "invalid JSON in '" + path + "': " + error;
        return false;
    }
    if (!root.isObject()) {
        error = "config root must be a JSON object";
        return false;
    }

    const Value& heightfield = requireObject(root, "heightfield", "config", error);
    if (!error.empty()) return false;
    if (!getInteger(heightfield, "size", config.heightfield.size, "heightfield", error)) {
        return false;
    }
    if (!getNumber(heightfield, "frequency", config.heightfield.base_frequency, "heightfield",
                   error) ||
        !getNumber(heightfield, "amplitude", config.heightfield.amplitude, "heightfield", error) ||
        !getUnsigned(heightfield, "seed", config.heightfield.seed, "heightfield", error) ||
        !getInteger(heightfield, "octaves", config.heightfield.octaves, "heightfield", error)) {
        return false;
    }
    if (config.heightfield.size < 64 || config.heightfield.size > 512) {
        error = "heightfield.size must be in [64, 512]";
        return false;
    }
    if (!(config.heightfield.base_frequency > 0.0) ||
        config.heightfield.base_frequency > 0.25) {
        error = "heightfield.frequency must be in (0, 0.25]";
        return false;
    }
    if (!(config.heightfield.amplitude > 0.0) || config.heightfield.amplitude > 200.0) {
        error = "heightfield.amplitude must be in (0, 200]";
        return false;
    }
    if (config.heightfield.octaves < 1 || config.heightfield.octaves > 8) {
        error = "heightfield.octaves must be in [1, 8]";
        return false;
    }

    const Value& camera = requireObject(root, "camera", "config", error);
    if (!error.empty()) return false;
    if (!getVec3(camera, "position", config.camera.position, "camera", error) ||
        !getVec3(camera, "target", config.camera.target, "camera", error) ||
        !getVec3(camera, "up", config.camera.up, "camera", error) ||
        !getNumber(camera, "fov_degrees", config.camera.fov_degrees, "camera", error) ||
        !getNumber(camera, "near", config.camera.near_plane, "camera", error) ||
        !getNumber(camera, "far", config.camera.far_plane, "camera", error)) {
        return false;
    }
    const Vec3 forward = config.camera.target - config.camera.position;
    if (length(forward) < 1e-6) {
        error = "camera.position and camera.target must be different";
        return false;
    }
    if (length(config.camera.up) < 1e-6 ||
        length(cross(normalize(forward), config.camera.up)) < 1e-6) {
        error = "camera.up must be non-zero and not parallel to the view direction";
        return false;
    }
    if (!(config.camera.fov_degrees > 5.0) || config.camera.fov_degrees >= 175.0) {
        error = "camera.fov_degrees must be in (5, 175)";
        return false;
    }
    if (!(config.camera.near_plane > 0.0) ||
        !(config.camera.far_plane > config.camera.near_plane)) {
        error = "camera requires 0 < near < far";
        return false;
    }

    const Value& render = requireObject(root, "render", "config", error);
    if (!error.empty()) return false;
    if (!getInteger(render, "width", config.render.width, "render", error) ||
        !getInteger(render, "height", config.render.height, "render", error) ||
        !getInteger(render, "supersampling", config.render.supersampling, "render", error)) {
        return false;
    }
    if (config.render.width < 64 || config.render.width > 4096 ||
        config.render.height < 64 || config.render.height > 4096) {
        error = "render.width and render.height must be in [64, 4096]";
        return false;
    }
    if (config.render.supersampling != 1 && config.render.supersampling != 2 &&
        config.render.supersampling != 4) {
        error = "render.supersampling must be one of 1, 2, 4";
        return false;
    }
    const long long internal_pixels =
        static_cast<long long>(config.render.width) * config.render.height *
        config.render.supersampling * config.render.supersampling;
    if (internal_pixels > 32000000LL) {
        error = "render internal resolution exceeds the 32M-sample safety limit";
        return false;
    }

    const Value& lod = requireObject(root, "lod", "config", error);
    if (!error.empty()) return false;
    if (!getInteger(lod, "block_size", config.lod.block_size, "lod", error) ||
        !getNumber(lod, "lod0_screen_area", config.lod.lod0_screen_area, "lod", error) ||
        !getNumber(lod, "lod1_screen_area", config.lod.lod1_screen_area, "lod", error) ||
        !getNumber(lod, "lod0_distance", config.lod.lod0_distance, "lod", error) ||
        !getNumber(lod, "lod1_distance", config.lod.lod1_distance, "lod", error)) {
        return false;
    }
    if (config.lod.block_size < 8 || config.lod.block_size > 64 ||
        config.lod.block_size % 4 != 0) {
        error = "lod.block_size must be a multiple of 4 in [8, 64]";
        return false;
    }
    if (!(config.lod.lod0_screen_area > config.lod.lod1_screen_area) ||
        config.lod.lod1_screen_area < 0.0) {
        error = "lod screen-area thresholds require lod0_screen_area > lod1_screen_area >= 0";
        return false;
    }
    if (!(config.lod.lod0_distance > 0.0) ||
        !(config.lod.lod1_distance > config.lod.lod0_distance)) {
        error = "lod distance thresholds require 0 < lod0_distance < lod1_distance";
        return false;
    }

    const Value& texture = requireObject(root, "texture", "config", error);
    if (!error.empty()) return false;
    if (!getInteger(texture, "size", config.texture.size, "texture", error) ||
        !getNumber(texture, "scale", config.texture.scale, "texture", error)) {
        return false;
    }
    if (config.texture.size < 16 || config.texture.size > 1024 ||
        (config.texture.size & (config.texture.size - 1)) != 0) {
        error = "texture.size must be a power of two in [16, 1024]";
        return false;
    }
    if (!(config.texture.scale > 0.0) || config.texture.scale > 32.0) {
        error = "texture.scale must be in (0, 32]";
        return false;
    }

    const Value& lighting = requireObject(root, "lighting", "config", error);
    if (!error.empty()) return false;
    if (!getVec3(lighting, "ambient", config.ambient, "lighting", error)) {
        return false;
    }
    if (!validColor(config.ambient)) {
        error = "lighting.ambient components must be in [0, 16]";
        return false;
    }
    const Value* directional = lighting.find("directional");
    const Value* point = lighting.find("point");
    if (directional == nullptr || !directional->isArray()) {
        error = "lighting.directional must be an array";
        return false;
    }
    if (point == nullptr || !point->isArray()) {
        error = "lighting.point must be an array";
        return false;
    }
    if (directional->asArray().size() < 4) {
        error = "lighting.directional must contain at least 4 lights";
        return false;
    }
    if (point->asArray().size() < 8) {
        error = "lighting.point must contain at least 8 lights";
        return false;
    }
    if (directional->asArray().size() > 64 || point->asArray().size() > 64) {
        error = "each lighting array may contain at most 64 lights";
        return false;
    }

    config.directional_lights.clear();
    for (std::size_t i = 0; i < directional->asArray().size(); ++i) {
        const Value& item = directional->asArray()[i];
        const std::string context = "lighting.directional[" + std::to_string(i) + "]";
        if (!item.isObject()) {
            error = context + " must be an object";
            return false;
        }
        DirectionalLight light;
        if (!getVec3(item, "direction", light.direction, context, error) ||
            !getVec3(item, "color", light.color, context, error) ||
            !getNumber(item, "intensity", light.intensity, context, error)) {
            return false;
        }
        if (length(light.direction) < 1e-8) {
            error = context + ".direction must be non-zero";
            return false;
        }
        if (!validColor(light.color) || light.intensity < 0.0 || light.intensity > 10000.0) {
            error = context + " has invalid color or intensity";
            return false;
        }
        light.direction = normalize(light.direction);
        config.directional_lights.push_back(light);
    }

    config.point_lights.clear();
    for (std::size_t i = 0; i < point->asArray().size(); ++i) {
        const Value& item = point->asArray()[i];
        const std::string context = "lighting.point[" + std::to_string(i) + "]";
        if (!item.isObject()) {
            error = context + " must be an object";
            return false;
        }
        PointLight light;
        if (!getVec3(item, "position", light.position, context, error) ||
            !getVec3(item, "color", light.color, context, error) ||
            !getNumber(item, "intensity", light.intensity, context, error) ||
            !getNumber(item, "range", light.range, context, error)) {
            return false;
        }
        if (!validColor(light.color) || light.intensity < 0.0 || light.intensity > 10000.0 ||
            !(light.range > 0.0) || light.range > 100000.0) {
            error = context + " has invalid color, intensity, or range";
            return false;
        }
        config.point_lights.push_back(light);
    }

    const Value& output = requireObject(root, "output", "config", error);
    if (!error.empty()) return false;
    if (!getString(output, "image", config.output.image_path, "output", error) ||
        !getString(output, "report", config.output.report_path, "output", error)) {
        return false;
    }
    if (!hasExtension(config.output.image_path, ".bmp") &&
        !hasExtension(config.output.image_path, ".ppm")) {
        error = "output.image must use a .bmp or .ppm extension";
        return false;
    }
    if (!hasExtension(config.output.report_path, ".json")) {
        error = "output.report must use a .json extension";
        return false;
    }

    const std::filesystem::path base = std::filesystem::absolute(path).parent_path();
    const auto resolve = [&base](const std::string& value) {
        std::filesystem::path candidate(value);
        if (candidate.is_relative()) {
            candidate = base / candidate;
        }
        return candidate.lexically_normal();
    };
    config.output.image_path = resolve(config.output.image_path).string();
    config.output.report_path = resolve(config.output.report_path).string();
    if (std::filesystem::path(config.output.image_path) ==
        std::filesystem::path(config.output.report_path)) {
        error = "output.image and output.report must be different files";
        return false;
    }
    return true;
}

}  // namespace ct
