#include "raster/scene.h"

#include "raster/json.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace raster {
namespace {

constexpr int kMaxResolution = 16384;

[[noreturn]] void sceneFail(const std::string& context,
                            const std::string& message) {
    throw SceneError(context + ": " + message);
}

const JsonValue* findAny(const JsonValue& object,
                         std::initializer_list<std::string_view> keys) {
    for (std::string_view key : keys) {
        if (const JsonValue* value = object.find(key)) {
            return value;
        }
    }
    return nullptr;
}

const JsonValue& requireField(const JsonValue& object,
                              std::initializer_list<std::string_view> keys,
                              const std::string& context) {
    if (!object.isObject()) {
        sceneFail(context, "expected an object");
    }
    const JsonValue* value = findAny(object, keys);
    if (value == nullptr) {
        std::string joined;
        bool first = true;
        for (std::string_view key : keys) {
            if (!first) {
                joined += "/";
            }
            first = false;
            joined += std::string(key);
        }
        sceneFail(context, "missing required field '" + joined + "'");
    }
    return *value;
}

void requireObject(const JsonValue& value, const std::string& context) {
    if (!value.isObject()) {
        sceneFail(context, "expected an object");
    }
}

void requireArray(const JsonValue& value, const std::string& context) {
    if (!value.isArray()) {
        sceneFail(context, "expected an array");
    }
}

double requireNumber(const JsonValue& value, const std::string& context) {
    if (!value.isNumber() || !std::isfinite(value.number)) {
        sceneFail(context, "expected a finite number");
    }
    return value.number;
}

std::uint64_t requireUnsignedIntegerFromDouble(double number,
                                               const std::string& context,
                                               std::uint64_t maximum) {
    if (number < 0.0 || std::floor(number) != number ||
        number > static_cast<double>(maximum)) {
        sceneFail(context, "expected an integer in [0, " +
                               std::to_string(maximum) + "]");
    }
    return static_cast<std::uint64_t>(number);
}

std::uint64_t requireUnsignedInteger(const JsonValue& value,
                                     const std::string& context,
                                     std::uint64_t maximum) {
    return requireUnsignedIntegerFromDouble(requireNumber(value, context),
                                            context, maximum);
}

std::string requireString(const JsonValue& value, const std::string& context) {
    if (!value.isString()) {
        sceneFail(context, "expected a string");
    }
    return value.string;
}

void flattenNumbers(const JsonValue& value, std::vector<double>& output,
                    const std::string& context, int depth = 0) {
    if (depth > 32) {
        sceneFail(context, "numeric array nesting is too deep");
    }
    if (value.isNumber()) {
        if (!std::isfinite(value.number)) {
            sceneFail(context, "array contains a non-finite number");
        }
        output.push_back(value.number);
        return;
    }
    if (!value.isArray()) {
        sceneFail(context, "expected a number or an array of numbers");
    }
    for (const JsonValue& element : value.array) {
        flattenNumbers(element, output, context, depth + 1);
    }
}

std::vector<double> numbersFrom(const JsonValue& value,
                                const std::string& context) {
    std::vector<double> result;
    flattenNumbers(value, result, context);
    return result;
}

void requireUnitRange(const Vec3& value, const std::string& context) {
    if (!isFinite(value) || value.x < 0.0 || value.x > 1.0 || value.y < 0.0 ||
        value.y > 1.0 || value.z < 0.0 || value.z > 1.0) {
        sceneFail(context, "expected color components in [0, 1]");
    }
}

void requireUnitRange(const Vec4& value, const std::string& context) {
    if (!isFinite(value) || value.x < 0.0 || value.x > 1.0 || value.y < 0.0 ||
        value.y > 1.0 || value.z < 0.0 || value.z > 1.0 || value.w < 0.0 ||
        value.w > 1.0) {
        sceneFail(context, "expected color/alpha components in [0, 1]");
    }
}

Vec3 parseVec3(const JsonValue& value, const std::string& context) {
    const std::vector<double> values = numbersFrom(value, context);
    if (values.size() != 3U) {
        sceneFail(context, "expected exactly 3 components");
    }
    const Vec3 result{values[0], values[1], values[2]};
    if (!isFinite(result)) {
        sceneFail(context, "components must be finite");
    }
    return result;
}

std::vector<Vec3> parseVec3List(const JsonValue& value,
                                const std::string& context) {
    const std::vector<double> values = numbersFrom(value, context);
    if (values.size() % 3U != 0U) {
        sceneFail(context, "component count must be a multiple of 3");
    }
    std::vector<Vec3> result;
    result.reserve(values.size() / 3U);
    for (std::size_t i = 0; i < values.size(); i += 3U) {
        const Vec3 v{values[i], values[i + 1], values[i + 2]};
        if (!isFinite(v)) {
            sceneFail(context, "component must be finite");
        }
        result.push_back(v);
    }
    return result;
}

std::vector<Vec2> parseVec2List(const JsonValue& value,
                                const std::string& context) {
    const std::vector<double> values = numbersFrom(value, context);
    if (values.size() % 2U != 0U) {
        sceneFail(context, "component count must be a multiple of 2");
    }
    std::vector<Vec2> result;
    result.reserve(values.size() / 2U);
    for (std::size_t i = 0; i < values.size(); i += 2U) {
        const Vec2 v{values[i], values[i + 1]};
        if (!isFinite(v)) {
            sceneFail(context, "component must be finite");
        }
        result.push_back(v);
    }
    return result;
}

std::vector<Vec4> parseColorList(const JsonValue& value,
                                 std::size_t expectedVertexCount,
                                 const std::string& context) {
    const std::vector<double> values = numbersFrom(value, context);
    const bool rgba = values.size() == expectedVertexCount * 4U;
    const bool rgb = values.size() == expectedVertexCount * 3U;
    if (!rgba && !rgb) {
        sceneFail(context,
                  "expected 3 (RGB) or 4 (RGBA) components per vertex");
    }
    std::vector<Vec4> result;
    result.reserve(expectedVertexCount);
    const std::size_t stride = rgba ? 4U : 3U;
    for (std::size_t i = 0; i < values.size(); i += stride) {
        const Vec4 v{values[i], values[i + 1], values[i + 2],
                     rgba ? values[i + 3] : 1.0};
        if (!isFinite(v)) {
            sceneFail(context, "component must be finite");
        }
        result.push_back(v);
    }
    return result;
}

std::vector<std::uint32_t> parseIndices(const JsonValue& value,
                                        const std::string& context) {
    const std::vector<double> values = numbersFrom(value, context);
    std::vector<std::uint32_t> result;
    result.reserve(values.size());
    for (double number : values) {
        if (!std::isfinite(number) || number < 0.0 ||
            std::floor(number) != number ||
            number > static_cast<double>(
                         std::numeric_limits<std::uint32_t>::max())) {
            sceneFail(context, "indices must be non-negative integers");
        }
        result.push_back(static_cast<std::uint32_t>(number));
    }
    return result;
}

Mat4 rotationX(double radians) {
    Mat4 result = Mat4::identity();
    const double c = std::cos(radians);
    const double s = std::sin(radians);
    result.at(1, 1) = c;
    result.at(1, 2) = -s;
    result.at(2, 1) = s;
    result.at(2, 2) = c;
    return result;
}

Mat4 rotationY(double radians) {
    Mat4 result = Mat4::identity();
    const double c = std::cos(radians);
    const double s = std::sin(radians);
    result.at(0, 0) = c;
    result.at(0, 2) = s;
    result.at(2, 0) = -s;
    result.at(2, 2) = c;
    return result;
}

Mat4 rotationZ(double radians) {
    Mat4 result = Mat4::identity();
    const double c = std::cos(radians);
    const double s = std::sin(radians);
    result.at(0, 0) = c;
    result.at(0, 1) = -s;
    result.at(1, 0) = s;
    result.at(1, 1) = c;
    return result;
}

Mat4 translation(const Vec3& value) {
    Mat4 result = Mat4::identity();
    result.at(0, 3) = value.x;
    result.at(1, 3) = value.y;
    result.at(2, 3) = value.z;
    return result;
}

Mat4 scaleMatrix(const Vec3& value) {
    Mat4 result = Mat4::identity();
    result.at(0, 0) = value.x;
    result.at(1, 1) = value.y;
    result.at(2, 2) = value.z;
    return result;
}

Mat4 parseModelMatrix(const JsonValue* value, const std::string& context) {
    if (value == nullptr) {
        return Mat4::identity();
    }

    if (value->isArray() || value->isNumber()) {
        const std::vector<double> values = numbersFrom(*value, context);
        if (values.size() != 16U) {
            sceneFail(context, "world/model matrix must have 16 components");
        }
        Mat4 result;
        for (std::size_t i = 0; i < 16U; ++i) {
            if (!std::isfinite(values[i])) {
                sceneFail(context, "matrix contains a non-finite value");
            }
            result.m[i] = values[i];
        }
        return result;
    }

    if (value->isObject()) {
        const Vec3 translate =
            findAny(*value, {"translation", "position"})
                ? parseVec3(*findAny(*value, {"translation", "position"}),
                            context + ".translation")
                : Vec3{};
        const Vec3 rotationDegrees =
            findAny(*value, {"rotation", "rotationDegrees"})
                ? parseVec3(*findAny(*value, {"rotation", "rotationDegrees"}),
                            context + ".rotation")
                : Vec3{};
        const Vec3 scale =
            findAny(*value, {"scale"})
                ? parseVec3(*findAny(*value, {"scale"}), context + ".scale")
                : Vec3{1.0, 1.0, 1.0};
        if (scale.x == 0.0 || scale.y == 0.0 || scale.z == 0.0) {
            sceneFail(context, "scale components must be non-zero");
        }
        const double toRadians = kPi / 180.0;
        const Mat4 rotation =
            rotationZ(rotationDegrees.z * toRadians) *
            rotationY(rotationDegrees.y * toRadians) *
            rotationX(rotationDegrees.x * toRadians);
        return translation(translate) * rotation * scaleMatrix(scale);
    }

    sceneFail(context,
              "expected a 16-element matrix or a translation/rotation/scale "
              "object");
}

std::pair<int, int> parseResolution(const JsonValue& value,
                                    const std::string& context) {
    if (value.isArray() || value.isNumber()) {
        const std::vector<double> values = numbersFrom(value, context);
        if (values.size() != 2U) {
            sceneFail(context, "expected [width, height]");
        }
        const auto width = requireUnsignedIntegerFromDouble(
            values[0], context + "[0]", kMaxResolution);
        const auto height = requireUnsignedIntegerFromDouble(
            values[1], context + "[1]", kMaxResolution);
        if (width == 0U || height == 0U) {
            sceneFail(context, "width and height must be at least 1");
        }
        return {static_cast<int>(width), static_cast<int>(height)};
    }

    if (value.isObject()) {
        const auto width = requireUnsignedInteger(
            requireField(value, {"width"}, context), context + ".width",
            kMaxResolution);
        const auto height = requireUnsignedInteger(
            requireField(value, {"height"}, context), context + ".height",
            kMaxResolution);
        if (width == 0U || height == 0U) {
            sceneFail(context, "width and height must be at least 1");
        }
        return {static_cast<int>(width), static_cast<int>(height)};
    }

    sceneFail(context, "expected [width, height] or {width, height}");
}

Camera parseCamera(const JsonValue& value, const std::string& context) {
    requireObject(value, context);

    Camera camera;
    camera.position = parseVec3(
        requireField(value, {"position", "eye"}, context), context + ".position");

    if (const JsonValue* target =
            findAny(value, {"target", "lookAt", "look_at"})) {
        const Vec3 targetPosition = parseVec3(*target, context + ".target");
        camera.direction = targetPosition - camera.position;
    } else {
        const JsonValue& direction =
            requireField(value, {"direction", "forward"}, context);
        camera.direction = parseVec3(direction, context + ".direction");
    }

    if (const JsonValue* up = findAny(value, {"up"})) {
        camera.up = parseVec3(*up, context + ".up");
    }

    const JsonValue& fovValue =
        requireField(value, {"fovY", "fov", "fovDegrees", "verticalFov"},
                     context);
    const double fovDegrees = requireNumber(fovValue, context + ".fovY");
    if (!(fovDegrees > 0.0 && fovDegrees < 180.0)) {
        sceneFail(context + ".fovY", "must be in the open interval (0, 180)");
    }

    const double nearPlane = requireNumber(
        requireField(value, {"near", "nearPlane"}, context), context + ".near");
    const double farPlane = requireNumber(
        requireField(value, {"far", "farPlane"}, context), context + ".far");
    if (!(nearPlane > 0.0) || !(farPlane > nearPlane)) {
        sceneFail(context, "require 0 < near < far");
    }

    if (!isFinite(camera.position) || !isFinite(camera.direction) ||
        !isFinite(camera.up)) {
        sceneFail(context, "camera vectors must be finite");
    }
    if (!(lengthSquared(camera.direction) > kEpsilon)) {
        sceneFail(context + ".direction", "must be non-zero");
    }
    if (!(lengthSquared(camera.up) > kEpsilon) ||
        lengthSquared(cross(camera.direction, camera.up)) <= kEpsilon) {
        sceneFail(context + ".up", "must be non-zero and not parallel to direction");
    }

    camera.fovYRadians = fovDegrees * kPi / 180.0;
    camera.nearPlane = nearPlane;
    camera.farPlane = farPlane;
    return camera;
}

DirectionalLight parseLight(const JsonValue& value,
                            const std::string& context) {
    requireObject(value, context);
    DirectionalLight light;
    light.direction = parseVec3(
        requireField(value, {"direction"}, context), context + ".direction");
    light.color = parseVec3(requireField(value, {"color", "colour"}, context),
                            context + ".color");
    light.intensity = requireNumber(
        requireField(value, {"intensity"}, context), context + ".intensity");

    if (!(lengthSquared(light.direction) > kEpsilon)) {
        sceneFail(context + ".direction", "must be non-zero");
    }
    requireUnitRange(light.color, context + ".color");
    if (!std::isfinite(light.intensity) || light.intensity < 0.0) {
        sceneFail(context + ".intensity", "must be finite and non-negative");
    }
    return light;
}

std::optional<std::string> parseOptionalPath(const JsonValue& object,
                                             std::initializer_list<
                                                 std::string_view> keys,
                                             const std::string& context) {
    const JsonValue* value = findAny(object, keys);
    if (value == nullptr || value->isNull()) {
        return std::nullopt;
    }
    std::string path = requireString(*value, context);
    if (path.empty()) {
        return std::nullopt;
    }
    return path;
}

Material parseMaterial(const JsonValue& value, std::size_t index,
                       const std::string& context) {
    requireObject(value, context);
    Material material;
    if (const JsonValue* name = findAny(value, {"name"})) {
        material.name = requireString(*name, context + ".name");
    } else {
        material.name = "material[" + std::to_string(index) + "]";
    }

    material.diffuse = parseVec3(
        requireField(value, {"diffuse", "diffuseColor", "albedo"}, context),
        context + ".diffuse");
    requireUnitRange(material.diffuse, context + ".diffuse");
    material.baseColor = material.diffuse;
    material.detailColor = material.diffuse;
    if (const JsonValue* baseColor = findAny(value, {"baseColor"})) {
        material.baseColor = parseVec3(*baseColor, context + ".baseColor");
        requireUnitRange(material.baseColor, context + ".baseColor");
    }
    if (const JsonValue* detailColor = findAny(value, {"detailColor"})) {
        material.detailColor =
            parseVec3(*detailColor, context + ".detailColor");
        requireUnitRange(material.detailColor, context + ".detailColor");
    }

    material.baseTexture = parseOptionalPath(
        value, {"baseTexture", "baseMap", "texture"}, context + ".baseTexture");
    material.detailTexture =
        parseOptionalPath(value, {"detailTexture", "detailMap"},
                          context + ".detailTexture");

    if (const JsonValue* weight =
            findAny(value, {"baseWeight", "baseMix"})) {
        material.baseWeight =
            requireNumber(*weight, context + ".baseWeight");
    }
    if (const JsonValue* weight =
            findAny(value, {"detailWeight", "detailMix"})) {
        material.detailWeight =
            requireNumber(*weight, context + ".detailWeight");
    }
    if (!std::isfinite(material.baseWeight) || material.baseWeight < 0.0 ||
        !std::isfinite(material.detailWeight) ||
        material.detailWeight < 0.0) {
        sceneFail(context, "layer weights must be finite and non-negative");
    }
    return material;
}

Mesh parseMesh(const JsonValue& value, std::size_t index,
               const std::string& context, std::size_t materialCount) {
    requireObject(value, context);
    Mesh mesh;
    if (const JsonValue* name = findAny(value, {"name"})) {
        mesh.name = requireString(*name, context + ".name");
    } else {
        mesh.name = "mesh[" + std::to_string(index) + "]";
    }

    const std::vector<Vec3> positions = parseVec3List(
        requireField(value, {"positions", "vertices"}, context),
        context + ".positions");
    const std::vector<Vec3> normals = parseVec3List(
        requireField(value, {"normals"}, context), context + ".normals");
    const std::vector<Vec2> uvs = parseVec2List(
        requireField(value, {"uvs", "uv", "texcoords"}, context),
        context + ".uvs");
    std::vector<Vec4> colors;
    if (const JsonValue* colorValue =
            findAny(value, {"colors", "vertexColors", "color"})) {
        colors = parseColorList(*colorValue, positions.size(),
                                context + ".colors");
    } else {
        colors.assign(positions.size(), Vec4{1.0, 1.0, 1.0, 1.0});
    }

    if (positions.empty()) {
        sceneFail(context + ".positions", "must contain at least one vertex");
    }
    if (normals.size() != positions.size()) {
        sceneFail(context, "normal count does not match vertex count");
    }
    if (uvs.size() != positions.size()) {
        sceneFail(context, "UV count does not match vertex count");
    }
    if (colors.size() != positions.size()) {
        sceneFail(context, "vertex color count does not match vertex count");
    }

    mesh.vertices.reserve(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        requireUnitRange(colors[i], context + ".colors[" + std::to_string(i) +
                                         "]");
        mesh.vertices.push_back(
            Vertex{positions[i], normals[i], uvs[i], colors[i]});
    }

    mesh.indices = parseIndices(requireField(value, {"indices"}, context),
                                context + ".indices");
    if (mesh.indices.size() % 3U != 0U) {
        sceneFail(context + ".indices",
                  "index count must be a multiple of 3");
    }
    for (std::uint32_t vertexIndex : mesh.indices) {
        if (vertexIndex >= mesh.vertices.size()) {
            sceneFail(context + ".indices",
                      "vertex index " + std::to_string(vertexIndex) +
                          " is out of range");
        }
    }

    const std::uint64_t materialIndex = requireUnsignedInteger(
        requireField(value, {"material", "materialIndex"}, context),
        context + ".material", std::numeric_limits<std::uint32_t>::max());
    if (materialIndex >= materialCount) {
        sceneFail(context + ".material",
                  "material index " + std::to_string(materialIndex) +
                      " is out of range");
    }
    mesh.materialIndex = static_cast<std::size_t>(materialIndex);

    Mat4 model = Mat4::identity();
    if (const JsonValue* modelValue =
            findAny(value, {"model", "world", "transform"})) {
        model = parseModelMatrix(modelValue, context + ".model");
    }
    if (!isFinite(model)) {
        sceneFail(context + ".model", "matrix must be finite");
    }
    Mat4 inverse;
    if (!invert(model, inverse)) {
        sceneFail(context + ".model", "matrix is singular");
    }
    mesh.model = model;
    mesh.normalMatrix = normalMatrix(model);
    return mesh;
}

}  // namespace

Scene parseScene(const JsonValue& root, const std::filesystem::path& sourcePath) {
    requireObject(root, "scene");
    Scene scene;
    scene.sourceDirectory = sourcePath.parent_path();
    if (scene.sourceDirectory.empty()) {
        scene.sourceDirectory = ".";
    }

    bool resolutionSpecified = false;
    if (const JsonValue* resolution = findAny(root, {"resolution"})) {
        const auto [width, height] =
            parseResolution(*resolution, "scene.resolution");
        scene.width = width;
        scene.height = height;
        resolutionSpecified = true;
    } else if (const JsonValue* width = findAny(root, {"width"})) {
        const auto parsedWidth = requireUnsignedInteger(
            *width, "scene.width", kMaxResolution);
        const auto parsedHeight = requireUnsignedInteger(
            requireField(root, {"height"}, "scene"), "scene.height",
            kMaxResolution);
        if (parsedWidth == 0U || parsedHeight == 0U) {
            sceneFail("scene", "width and height must be at least 1");
        }
        scene.width = static_cast<int>(parsedWidth);
        scene.height = static_cast<int>(parsedHeight);
        resolutionSpecified = true;
    }

    scene.camera = parseCamera(
        requireField(root, {"camera"}, "scene"), "scene.camera");

    if (!resolutionSpecified) {
        const JsonValue* cameraObject = findAny(root, {"camera"});
        if (const JsonValue* resolution =
                cameraObject ? findAny(*cameraObject, {"resolution"})
                             : nullptr) {
            const auto [width, height] =
                parseResolution(*resolution, "scene.camera.resolution");
            scene.width = width;
            scene.height = height;
        }
    }

    const JsonValue* lightsField = &requireField(root, {"lights"}, "scene");
    requireArray(*lightsField, "scene.lights");
    const JsonValue& lights = *lightsField;
    if (lights.array.empty()) {
        sceneFail("scene.lights", "must contain at least one directional light");
    }
    scene.lights.reserve(lights.array.size());
    for (std::size_t i = 0; i < lights.array.size(); ++i) {
        scene.lights.push_back(parseLight(
            lights.array[i], "scene.lights[" + std::to_string(i) + "]"));
    }

    const JsonValue* materialsField =
        &requireField(root, {"materials"}, "scene");
    requireArray(*materialsField, "scene.materials");
    const JsonValue& materials = *materialsField;
    if (materials.array.empty()) {
        sceneFail("scene.materials", "must contain at least one material");
    }
    scene.materials.reserve(materials.array.size());
    for (std::size_t i = 0; i < materials.array.size(); ++i) {
        scene.materials.push_back(parseMaterial(
            materials.array[i], i,
            "scene.materials[" + std::to_string(i) + "]"));
    }

    const JsonValue* meshesField = &requireField(root, {"meshes"}, "scene");
    requireArray(*meshesField, "scene.meshes");
    const JsonValue& meshes = *meshesField;
    if (meshes.array.empty()) {
        sceneFail("scene.meshes", "must contain at least one mesh");
    }
    scene.meshes.reserve(meshes.array.size());
    for (std::size_t i = 0; i < meshes.array.size(); ++i) {
        scene.meshes.push_back(parseMesh(
            meshes.array[i], i, "scene.meshes[" + std::to_string(i) + "]",
            scene.materials.size()));
    }

    return scene;
}

}  // namespace raster
