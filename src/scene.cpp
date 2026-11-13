#include "scene.hpp"

#include "errors.hpp"
#include "json.hpp"
#include "sha256.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string_view>

namespace lfb {

namespace {

[[noreturn]] void parseError(const std::string& message) {
    throw BusinessError(ExitCode::E_SCENE_PARSE, message);
}

[[noreturn]] void validError(const std::string& message) {
    throw BusinessError(ExitCode::E_SCENE_VALID, message);
}

const json::Value* findAlias(const json::Value& object,
                             std::initializer_list<std::string_view> names) {
    for (const auto name : names) {
        const json::Value* value = object.find(name);
        if (value != nullptr) {
            return value;
        }
    }
    return nullptr;
}

const json::Value& requireAlias(const json::Value& object,
                                std::initializer_list<std::string_view> names,
                                const std::string& context) {
    if (const json::Value* value = findAlias(object, names)) {
        return *value;
    }
    std::string joined;
    for (const auto name : names) {
        if (!joined.empty()) {
            joined += " or ";
        }
        joined += "'";
        joined += name;
        joined += "'";
    }
    parseError("missing required field " + joined + " in " + context);
}

const json::Value& requireObject(const json::Value& value, const std::string& context) {
    if (!value.isObject()) {
        parseError(context + " must be an object, got " + json::typeName(value.type()));
    }
    return value;
}

void validateArray(const json::Value& value, const std::string& context) {
    if (!value.isArray()) {
        parseError(context + " must be an array, got " + json::typeName(value.type()));
    }
}

double numberValue(const json::Value& value, const std::string& context) {
    if (!value.isNumber()) {
        parseError(context + " must be a number, got " + json::typeName(value.type()));
    }
    return value.asNumber();
}

double numberField(const json::Value& object,
                   std::initializer_list<std::string_view> names,
                   const std::string& context) {
    return numberValue(requireAlias(object, names, context), context);
}

int integerValue(const json::Value& value, const std::string& context) {
    const double number = numberValue(value, context);
    if (!std::isfinite(number) || std::floor(number) != number ||
        number < static_cast<double>(std::numeric_limits<int>::min()) ||
        number > static_cast<double>(std::numeric_limits<int>::max())) {
        validError(context + " must be a finite integer");
    }
    return static_cast<int>(number);
}

std::uint32_t frameIndexValue(const json::Value& value, const std::string& context) {
    const double number = numberValue(value, context);
    if (!std::isfinite(number) || std::floor(number) != number || number < 0.0 ||
        number > 4294967295.0) {
        validError(context + " must be a non-negative integer");
    }
    return static_cast<std::uint32_t>(number);
}

Vec3 vec3Value(const json::Value& value, const std::string& context) {
    if (value.isArray()) {
        const auto& array = value.asArray();
        if (array.size() != 3) {
            parseError(context + " must contain exactly 3 numbers");
        }
        const Vec3 result{
            numberValue(array[0], context + "[0]"),
            numberValue(array[1], context + "[1]"),
            numberValue(array[2], context + "[2]"),
        };
        if (!std::isfinite(result.x) || !std::isfinite(result.y) || !std::isfinite(result.z)) {
            validError(context + " contains a non-finite component");
        }
        return result;
    }

    if (value.isObject()) {
        const double x = numberField(value, {"x", "r"}, context + ".x");
        const double y = numberField(value, {"y", "g"}, context + ".y");
        const double z = numberField(value, {"z", "b"}, context + ".z");
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
            validError(context + " contains a non-finite component");
        }
        return Vec3{x, y, z};
    }

    parseError(context + " must be an array or object of 3 numbers");
}

Color colorValue(const json::Value& value, const std::string& context) {
    Color color;
    if (value.isArray()) {
        const auto& array = value.asArray();
        if (array.size() != 3 && array.size() != 4) {
            parseError(context + " must contain 3 or 4 numbers");
        }
        color = Color{
            numberValue(array[0], context + "[0]"),
            numberValue(array[1], context + "[1]"),
            numberValue(array[2], context + "[2]"),
        };
    } else if (value.isObject()) {
        color = Color{
            numberField(value, {"r", "red"}, context + ".r"),
            numberField(value, {"g", "green"}, context + ".g"),
            numberField(value, {"b", "blue"}, context + ".b"),
        };
    } else {
        parseError(context + " must be an RGB array or object, got " +
                   json::typeName(value.type()));
    }

    const double rgb[3] = {color.r, color.g, color.b};
    bool usesEightBitScale = false;
    for (double component : rgb) {
        if (!std::isfinite(component) || component < 0.0 || component > 255.0) {
            validError(context + " components must be finite and within [0, 255]");
        }
        if (component > 1.0) {
            usesEightBitScale = true;
        }
    }
    if (usesEightBitScale) {
        color.r /= 255.0;
        color.g /= 255.0;
        color.b /= 255.0;
    }
    return color;
}

std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        parseError("cannot open scene file '" + path.string() + "'");
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof()) {
        parseError("cannot read scene file '" + path.string() + "'");
    }
    return contents.str();
}

void parseResolution(const json::Value& root, Scene& scene) {
    const json::Value* resolution = findAlias(root, {"resolution", "renderResolution", "render_resolution"});
    if (resolution == nullptr) {
        const json::Value* width = findAlias(
            root, {"width", "outputWidth", "output_width", "renderWidth", "render_width"});
        const json::Value* height = findAlias(
            root, {"height", "outputHeight", "output_height", "renderHeight", "render_height"});
        if (width == nullptr || height == nullptr) {
            parseError("missing resolution (expected resolution object/array or width and height)");
        }
        scene.width = integerValue(*width, "width");
        scene.height = integerValue(*height, "height");
        return;
    }
    if (resolution->isArray()) {
        const auto& array = resolution->asArray();
        if (array.size() != 2) {
            parseError("resolution array must contain exactly [width, height]");
        }
        scene.width = integerValue(array[0], "resolution[0]");
        scene.height = integerValue(array[1], "resolution[1]");
        return;
    }
    if (resolution->isObject()) {
        scene.width = integerValue(
            requireAlias(*resolution, {"width", "w"}, "resolution"), "resolution.width");
        scene.height = integerValue(
            requireAlias(*resolution, {"height", "h"}, "resolution"), "resolution.height");
        return;
    }
    parseError("resolution must be an object or a [width, height] array");
}

std::vector<FrameDesc> parseFrames(const json::Value& root) {
    const json::Value* samples = findAlias(
        root,
        {"samples", "sampleDirections", "sample_directions", "samplingDirections",
         "sampling_directions", "viewDirections", "view_directions", "cameraDirections",
         "camera_directions", "sampleDirectionList", "sample_direction_list",
         "directions", "frames", "frameList", "frame_list"});
    if (samples == nullptr) {
        parseError("missing required field 'samples'/'directions'/'frames'");
    }
    validateArray(*samples, "sample directions");
    const auto& array = samples->asArray();
    if (array.empty()) {
        validError("sample direction list must not be empty");
    }
    if (array.size() > kMaxFrames) {
        validError("frame count exceeds limit of " + std::to_string(kMaxFrames));
    }

    std::vector<FrameDesc> frames;
    frames.reserve(array.size());
    bool anyExplicitIndex = false;
    for (std::size_t i = 0; i < array.size(); ++i) {
        const std::string context = "frames[" + std::to_string(i) + "]";
        FrameDesc frame;
        frame.index = static_cast<std::uint32_t>(i);
        if (array[i].isArray()) {
            frame.direction = vec3Value(array[i], context);
        } else if (array[i].isObject()) {
            const auto& object = array[i];
            frame.direction = vec3Value(
                requireAlias(object,
                             {"direction", "dir", "sampleDirection", "sample_direction",
                              "viewDirection", "view_direction", "cameraDirection",
                              "camera_direction"},
                             context),
                context + ".direction");
            if (const json::Value* index = object.find("index")) {
                frame.index = frameIndexValue(*index, context + ".index");
                anyExplicitIndex = true;
            }
        } else {
            parseError(context + " must be a direction array or object");
        }
        if (length(frame.direction) < 1e-12) {
            validError(context + ".direction must be non-zero");
        }
        frame.direction = normalized(frame.direction);
        frames.push_back(frame);
    }

    std::set<std::uint32_t> indices;
    for (const auto& frame : frames) {
        if (!indices.insert(frame.index).second) {
            validError("frame indices must be unique");
        }
    }
    if (anyExplicitIndex) {
        if (indices.size() != frames.size() || *indices.begin() != 0 ||
            *indices.rbegin() != static_cast<std::uint32_t>(frames.size() - 1)) {
            validError("frame indices must form a contiguous 0..N-1 sequence");
        }
    }
    std::sort(frames.begin(), frames.end(), [](const FrameDesc& a, const FrameDesc& b) {
        return a.index < b.index;
    });
    return frames;
}

std::vector<MaterialLayer> parseLayers(const json::Value& root) {
    const json::Value* layersValue = findAlias(
        root, {"layers", "materials", "materialLayers", "material_layers"});
    if (layersValue == nullptr) {
        parseError("missing required field 'layers'/'materials'");
    }
    validateArray(*layersValue, "material layers");
    const auto& array = layersValue->asArray();
    if (array.empty()) {
        validError("material layer list must not be empty");
    }
    if (array.size() > kMaxMaterialLayers) {
        validError("material layer count exceeds limit of " +
                   std::to_string(kMaxMaterialLayers));
    }

    std::vector<MaterialLayer> layers;
    layers.reserve(array.size());
    for (std::size_t i = 0; i < array.size(); ++i) {
        const std::string context = "layers[" + std::to_string(i) + "]";
        const auto& object = requireObject(array[i], context);
        MaterialLayer layer;
        layer.baseColor = colorValue(
            requireAlias(object,
                         {"baseColor", "base_color", "baseColour", "base_colour", "base",
                          "albedo", "diffuse", "diffuseColor", "diffuse_color"},
                         context),
            context + ".baseColor");
        layer.roughness = numberField(object, {"roughness"}, context);
        layer.normalStrength = numberField(
            object,
            {"normalStrength", "normal_strength", "normalDistortion", "normal_distortion",
             "normalScale", "normal_scale", "bumpStrength", "bump_strength"},
            context);
        layer.emissionColor = colorValue(
            requireAlias(object,
                         {"emissionColor", "emission_color", "emissiveColor", "emissive_color",
                          "emission", "emissive"},
                         context),
            context + ".emissionColor");

        if (!std::isfinite(layer.roughness) || layer.roughness < 0.0 || layer.roughness > 1.0) {
            validError(context + ".roughness must be within [0, 1]");
        }
        if (!std::isfinite(layer.normalStrength) || layer.normalStrength < 0.0 ||
            layer.normalStrength > 1.0) {
            validError(context + ".normalStrength must be within [0, 1]");
        }
        layers.push_back(layer);
    }
    return layers;
}

std::vector<Light> parseLights(const json::Value& root) {
    const json::Value* lightsValue =
        findAlias(root, {"lights", "lighting", "lightSamples", "light_samples"});
    if (lightsValue == nullptr) {
        parseError("missing required field 'lights'");
    }
    validateArray(*lightsValue, "lights");
    const auto& array = lightsValue->asArray();
    std::vector<Light> lights;
    lights.reserve(array.size());
    for (std::size_t i = 0; i < array.size(); ++i) {
        const std::string context = "lights[" + std::to_string(i) + "]";
        if (array[i].isArray()) {
            Light light;
            light.direction = vec3Value(array[i], context);
            light.color = Color{1.0, 1.0, 1.0};
            light.intensity = 1.0;
            if (length(light.direction) < 1e-12) {
                validError(context + ".direction must be non-zero");
            }
            light.direction = normalized(light.direction);
            lights.push_back(light);
            continue;
        }

        const auto& object = requireObject(array[i], context);
        Light light;
        light.direction = vec3Value(
            requireAlias(object, {"direction", "dir", "vector"}, context),
            context + ".direction");
        if (const json::Value* color = findAlias(object, {"color", "colour", "rgb"})) {
            light.color = colorValue(*color, context + ".color");
        } else {
            light.color = Color{1.0, 1.0, 1.0};
        }
        if (const json::Value* intensity =
                findAlias(object, {"intensity", "power", "strength"})) {
            light.intensity = numberValue(*intensity, context + ".intensity");
        } else {
            light.intensity = 1.0;
        }
        if (length(light.direction) < 1e-12) {
            validError(context + ".direction must be non-zero");
        }
        if (!std::isfinite(light.intensity) || light.intensity < 0.0 ||
            light.intensity > 1e6) {
            validError(context + ".intensity must be finite and within [0, 1000000]");
        }
        light.direction = normalized(light.direction);
        lights.push_back(light);
    }
    return lights;
}

std::string formatDouble(double value) {
    if (value == 0.0) {
        value = 0.0;
    }
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    return buffer;
}

std::string formatVec(Vec3 value) {
    return formatDouble(value.x) + "," + formatDouble(value.y) + "," + formatDouble(value.z);
}

std::string formatColor(Color value) {
    return formatDouble(value.r) + "," + formatDouble(value.g) + "," + formatDouble(value.b);
}

}  // namespace

Vec3 normalized(Vec3 value) {
    const double len = length(value);
    if (len <= 0.0) {
        return Vec3{0.0, 0.0, 0.0};
    }
    return Vec3{value.x / len, value.y / len, value.z / len};
}

double length(Vec3 value) {
    return std::hypot(value.x, value.y, value.z);
}

Scene loadScene(const std::filesystem::path& scenePath) {
    const std::string text = readTextFile(scenePath);
    std::string_view jsonText(text);
    if (jsonText.size() >= 3 &&
        static_cast<unsigned char>(jsonText[0]) == 0xEF &&
        static_cast<unsigned char>(jsonText[1]) == 0xBB &&
        static_cast<unsigned char>(jsonText[2]) == 0xBF) {
        jsonText.remove_prefix(3);
    }
    json::Value root;
    try {
        root = json::parse(jsonText);
    } catch (const json::ParseError& error) {
        parseError(error.what());
    }
    if (!root.isObject()) {
        parseError("scene root must be a JSON object");
    }

    Scene scene;
    parseResolution(root, scene);
    if (scene.width <= 0 || scene.height <= 0 || scene.width > kMaxResolution ||
        scene.height > kMaxResolution) {
        validError("resolution must be within 1.." + std::to_string(kMaxResolution) +
                   " for both width and height");
    }
    scene.frames = parseFrames(root);
    scene.layers = parseLayers(root);
    scene.lights = parseLights(root);
    return scene;
}

std::string globalCacheKey(const Scene& scene) {
    std::ostringstream canonical;
    canonical << "LFBAKE-GLOBAL-v1\n";
    canonical << "pipeline=soft-raster-v1\n";
    canonical << "width=" << scene.width << "\n";
    canonical << "height=" << scene.height << "\n";
    canonical << "mesh=lat48-lon96\n";
    canonical << "fovY=45\n";
    canonical << "cameraDistance=3.2\n";
    canonical << "frameCount=" << scene.frames.size() << "\n";
    canonical << "frameIndices=";
    for (std::size_t i = 0; i < scene.frames.size(); ++i) {
        if (i != 0) {
            canonical << ",";
        }
        canonical << scene.frames[i].index;
    }
    canonical << "\n";
    return Sha256::hash(canonical.str());
}

std::string frameCacheKey(const Scene& scene, const FrameDesc& frame) {
    std::ostringstream canonical;
    canonical << "LFBAKE-FRAME-v1\n";
    canonical << "pipeline=soft-raster-v1\n";
    canonical << "layers=" << scene.layers.size() << "\n";
    for (std::size_t i = 0; i < scene.layers.size(); ++i) {
        const auto& layer = scene.layers[i];
        canonical << "L" << i << "|base=" << formatColor(layer.baseColor)
                  << "|rough=" << formatDouble(layer.roughness)
                  << "|normal=" << formatDouble(layer.normalStrength)
                  << "|emission=" << formatColor(layer.emissionColor) << "\n";
    }
    canonical << "lights=" << scene.lights.size() << "\n";
    for (std::size_t i = 0; i < scene.lights.size(); ++i) {
        const auto& light = scene.lights[i];
        canonical << "G" << i << "|direction=" << formatVec(light.direction)
                  << "|color=" << formatColor(light.color)
                  << "|intensity=" << formatDouble(light.intensity) << "\n";
    }
    canonical << "direction=" << formatVec(frame.direction) << "\n";
    return Sha256::hash(canonical.str());
}

}  // namespace lfb
