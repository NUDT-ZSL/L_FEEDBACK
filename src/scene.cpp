#include "scene.h"

#include "json.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string_view>

namespace softraster {
namespace {

class Validator {
public:
    std::optional<RenderError> error;

    const JsonValue* require(const JsonValue& parent, std::string_view key,
                             const std::string& path) {
        if (!parent.isObject()) {
            fail(ErrorCode::TypeMismatch, "expected JSON object", path);
            return nullptr;
        }
        const JsonValue* value = parent.find(key);
        if (value == nullptr) {
            fail(ErrorCode::MissingField, "missing required field '" + std::string(key) + "'",
                 path);
        }
        return value;
    }

    const JsonValue* optional(const JsonValue& parent, std::string_view key) {
        if (!parent.isObject()) {
            return nullptr;
        }
        return parent.find(key);
    }

    bool requireObject(const JsonValue& value, const std::string& path) {
        if (!value.isObject()) {
            return fail(ErrorCode::TypeMismatch, "expected JSON object", path);
        }
        return true;
    }

    bool requireArray(const JsonValue& value, const std::string& path) {
        if (!value.isArray()) {
            return fail(ErrorCode::TypeMismatch, "expected JSON array", path);
        }
        return true;
    }

    bool requireBool(const JsonValue& value, const std::string& path, bool& output) {
        if (!value.isBool()) {
            return fail(ErrorCode::TypeMismatch, "expected boolean", path);
        }
        output = value.asBool();
        return true;
    }

    bool requireString(const JsonValue& value, const std::string& path, std::string& output) {
        if (!value.isString()) {
            return fail(ErrorCode::TypeMismatch, "expected string", path);
        }
        output = value.asString();
        return true;
    }

    bool requireNumber(const JsonValue& value, const std::string& path, double& output) {
        if (!value.isNumber()) {
            return fail(ErrorCode::TypeMismatch, "expected number", path);
        }
        output = value.asNumber();
        if (!std::isfinite(output)) {
            return fail(ErrorCode::NonFiniteFloat, "floating-point value is not finite", path);
        }
        return true;
    }

    bool requireFloat(const JsonValue& value, const std::string& path, float& output,
                      float minimum, float maximum, bool minimumExclusive = false) {
        double number = 0.0;
        if (!requireNumber(value, path, number)) {
            return false;
        }
        if (!std::isfinite(static_cast<float>(number))) {
            return fail(ErrorCode::NonFiniteFloat,
                        "value is outside the finite 32-bit float range", path);
        }
        const float converted = static_cast<float>(number);
        if ((minimumExclusive ? !(converted > minimum) : converted < minimum) ||
            converted > maximum) {
            return fail(ErrorCode::InvalidValue, "numeric value is outside the allowed range",
                        path);
        }
        output = converted;
        return true;
    }

    bool requireInt(const JsonValue& value, const std::string& path, int& output, int minimum,
                    int maximum) {
        double number = 0.0;
        if (!requireNumber(value, path, number)) {
            return false;
        }
        if (std::floor(number) != number || number < static_cast<double>(minimum) ||
            number > static_cast<double>(maximum)) {
            return fail(ErrorCode::InvalidValue, "expected an integer in range", path);
        }
        output = static_cast<int>(number);
        return true;
    }

    bool requireSize(const JsonValue& value, const std::string& path, std::size_t& output,
                     std::size_t maximum, ErrorCode rangeError = ErrorCode::InvalidValue) {
        double number = 0.0;
        if (!requireNumber(value, path, number)) {
            return false;
        }
        if (std::floor(number) != number || number < 0.0 ||
            number > static_cast<double>(maximum)) {
            return fail(rangeError, "integer index or count is outside the allowed range", path);
        }
        output = static_cast<std::size_t>(number);
        return true;
    }

    bool requireIndex(const JsonValue& value, const std::string& path, int& output,
                      int maximum) {
        double number = 0.0;
        if (!requireNumber(value, path, number)) {
            return false;
        }
        if (std::floor(number) != number || number < 0.0) {
            return fail(ErrorCode::IndexOutOfRange,
                        "index must be a non-negative integer", path);
        }
        if (number > static_cast<double>(maximum)) {
            return fail(ErrorCode::IndexOutOfRange,
                        "index exceeds the referenced array size", path);
        }
        output = static_cast<int>(number);
        return true;
    }

    bool requireVec2(const JsonValue& value, const std::string& path, Vec2& output) {
        if (!requireArray(value, path) || value.asArray().size() != 2) {
            if (value.isArray()) {
                fail(ErrorCode::InvalidValue, "expected exactly 2 numbers", path);
            }
            return false;
        }
        if (!requireFloat(value.asArray()[0], path + "[0]", output.x,
                          -std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::max()) ||
            !requireFloat(value.asArray()[1], path + "[1]", output.y,
                          -std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::max())) {
            return false;
        }
        return true;
    }

    bool requireVec3(const JsonValue& value, const std::string& path, Vec3& output,
                     float minimum = -std::numeric_limits<float>::max(),
                     float maximum = std::numeric_limits<float>::max()) {
        if (!requireArray(value, path) || value.asArray().size() != 3) {
            if (value.isArray()) {
                fail(ErrorCode::InvalidValue, "expected exactly 3 numbers", path);
            }
            return false;
        }
        return requireFloat(value.asArray()[0], path + "[0]", output.x, minimum, maximum) &&
               requireFloat(value.asArray()[1], path + "[1]", output.y, minimum, maximum) &&
               requireFloat(value.asArray()[2], path + "[2]", output.z, minimum, maximum);
    }

    bool fail(ErrorCode code, std::string message, const std::string& path) {
        if (!error.has_value()) {
            error = makeError(code, std::move(message), path);
        }
        return false;
    }
};

bool readFile(const std::string& path, std::string& contents, RenderError& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = makeError(ErrorCode::Io, "unable to open scene file", path);
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0) {
        error = makeError(ErrorCode::Io, "unable to determine scene file size", path);
        return false;
    }
    if (static_cast<std::uint64_t>(size) > kMaxSceneFileBytes) {
        error = makeError(ErrorCode::MemoryLimitExceeded,
                          "scene JSON exceeds the 64 MiB input limit", path);
        return false;
    }
    input.seekg(0, std::ios::beg);
    contents.resize(static_cast<std::size_t>(size));
    if (size > 0 && !input.read(contents.data(), static_cast<std::streamsize>(size))) {
        error = makeError(ErrorCode::Io, "unable to read scene file", path);
        return false;
    }
    return true;
}

bool parseCamera(const JsonValue& root, Camera& camera, Validator& validator) {
    const JsonValue* node = validator.require(root, "camera", "camera");
    if (node == nullptr || !validator.requireObject(*node, "camera")) {
        return false;
    }
    const JsonValue* value = nullptr;
    if ((value = validator.require(*node, "position", "camera.position")) == nullptr ||
        !validator.requireVec3(*value, "camera.position", camera.position)) {
        return false;
    }
    if ((value = validator.require(*node, "target", "camera.target")) == nullptr ||
        !validator.requireVec3(*value, "camera.target", camera.target)) {
        return false;
    }
    if ((value = validator.require(*node, "up", "camera.up")) == nullptr ||
        !validator.requireVec3(*value, "camera.up", camera.up)) {
        return false;
    }
    if ((value = validator.require(*node, "fovYDegrees", "camera.fovYDegrees")) == nullptr ||
        !validator.requireFloat(*value, "camera.fovYDegrees", camera.fovYDegrees, 1.0F,
                                179.0F)) {
        return false;
    }
    if ((value = validator.require(*node, "near", "camera.near")) == nullptr ||
        !validator.requireFloat(*value, "camera.near", camera.nearPlane, 0.0F,
                                std::numeric_limits<float>::max(), true)) {
        return false;
    }
    if ((value = validator.require(*node, "far", "camera.far")) == nullptr ||
        !validator.requireFloat(*value, "camera.far", camera.farPlane, 0.0F,
                                std::numeric_limits<float>::max(), true)) {
        return false;
    }
    if (!(camera.farPlane > camera.nearPlane)) {
        return validator.fail(ErrorCode::InvalidValue, "camera.far must exceed camera.near",
                              "camera.far");
    }
    const Vec3 forward = camera.target - camera.position;
    const Vec3 side = cross(normalize(forward, {0.0F, 0.0F, 0.0F}), camera.up);
    if (lengthSquared(forward) <= 1.0e-12F || lengthSquared(camera.up) <= 1.0e-12F ||
        lengthSquared(side) <= 1.0e-12F) {
        return validator.fail(ErrorCode::InvalidValue,
                              "camera position, target and up must form a valid view basis",
                              "camera");
    }
    return true;
}

bool parseImage(const JsonValue& root, Scene& scene, Validator& validator) {
    const JsonValue* image = validator.require(root, "image", "image");
    if (image == nullptr || !validator.requireObject(*image, "image")) {
        return false;
    }
    const JsonValue* value = nullptr;
    if ((value = validator.require(*image, "width", "image.width")) == nullptr ||
        !validator.requireInt(*value, "image.width", scene.imageWidth, 1, kMaxImageDimension)) {
        return false;
    }
    if ((value = validator.require(*image, "height", "image.height")) == nullptr ||
        !validator.requireInt(*value, "image.height", scene.imageHeight, 1,
                              kMaxImageDimension)) {
        return false;
    }
    const JsonValue* viewport = validator.require(*image, "viewport", "image.viewport");
    if (viewport == nullptr || !validator.requireObject(*viewport, "image.viewport")) {
        return false;
    }
    if ((value = validator.require(*viewport, "x", "image.viewport.x")) == nullptr ||
        !validator.requireInt(*value, "image.viewport.x", scene.viewport.x, 0,
                              scene.imageWidth - 1)) {
        return false;
    }
    if ((value = validator.require(*viewport, "y", "image.viewport.y")) == nullptr ||
        !validator.requireInt(*value, "image.viewport.y", scene.viewport.y, 0,
                              scene.imageHeight - 1)) {
        return false;
    }
    if ((value = validator.require(*viewport, "width", "image.viewport.width")) == nullptr ||
        !validator.requireInt(*value, "image.viewport.width", scene.viewport.width, 1,
                              scene.imageWidth - scene.viewport.x)) {
        return false;
    }
    if ((value = validator.require(*viewport, "height", "image.viewport.height")) == nullptr ||
        !validator.requireInt(*value, "image.viewport.height", scene.viewport.height, 1,
                              scene.imageHeight - scene.viewport.y)) {
        return false;
    }
    return true;
}

bool parseDirectionalLight(const JsonValue& root, DirectionalLight& light,
                           Validator& validator) {
    const JsonValue* node = validator.require(root, "directionalLight", "directionalLight");
    if (node == nullptr || !validator.requireObject(*node, "directionalLight")) {
        return false;
    }
    const JsonValue* value = nullptr;
    if ((value = validator.require(*node, "direction", "directionalLight.direction")) ==
            nullptr ||
        !validator.requireVec3(*value, "directionalLight.direction", light.directionToLight)) {
        return false;
    }
    if (lengthSquared(light.directionToLight) <= 1.0e-12F) {
        return validator.fail(ErrorCode::InvalidValue,
                              "directional light direction must be non-zero",
                              "directionalLight.direction");
    }
    light.directionToLight = normalize(light.directionToLight);
    if ((value = validator.require(*node, "color", "directionalLight.color")) == nullptr ||
        !validator.requireVec3(*value, "directionalLight.color", light.color, 0.0F, 1.0F)) {
        return false;
    }
    if ((value = validator.require(*node, "intensity", "directionalLight.intensity")) ==
            nullptr ||
        !validator.requireFloat(*value, "directionalLight.intensity", light.intensity, 0.0F,
                                1.0e6F)) {
        return false;
    }

    const JsonValue* shadow = validator.require(*node, "shadow", "directionalLight.shadow");
    if (shadow == nullptr || !validator.requireObject(*shadow, "directionalLight.shadow")) {
        return false;
    }
    if ((value = validator.require(*shadow, "enabled", "directionalLight.shadow.enabled")) ==
            nullptr ||
        !validator.requireBool(*value, "directionalLight.shadow.enabled", light.shadowEnabled)) {
        return false;
    }
    if ((value = validator.require(*shadow, "mapSize", "directionalLight.shadow.mapSize")) ==
        nullptr) {
        return false;
    }
    double mapSizeNumber = 0.0;
    if (!validator.requireNumber(*value, "directionalLight.shadow.mapSize", mapSizeNumber)) {
        return false;
    }
    if (mapSizeNumber > static_cast<double>(kMaxShadowMapSize)) {
        return validator.fail(ErrorCode::ShadowMapTooLarge,
                              "shadow map size must not exceed 8192",
                              "directionalLight.shadow.mapSize");
    }
    if (std::floor(mapSizeNumber) != mapSizeNumber || mapSizeNumber < 16.0) {
        return validator.fail(ErrorCode::InvalidValue,
                              "shadow map size must be an integer of at least 16",
                              "directionalLight.shadow.mapSize");
    }
    light.shadowMapSize = static_cast<int>(mapSizeNumber);
    if ((value = validator.require(*shadow, "depthBias", "directionalLight.shadow.depthBias")) ==
            nullptr ||
        !validator.requireFloat(*value, "directionalLight.shadow.depthBias", light.depthBias,
                                -0.1F, 0.1F)) {
        return false;
    }
    if ((value = validator.require(*shadow, "normalBias",
                                   "directionalLight.shadow.normalBias")) == nullptr ||
        !validator.requireFloat(*value, "directionalLight.shadow.normalBias", light.normalBias,
                                0.0F, 10.0F)) {
        return false;
    }
    if ((value = validator.require(*shadow, "pcfRadius", "directionalLight.shadow.pcfRadius")) ==
            nullptr ||
        !validator.requireInt(*value, "directionalLight.shadow.pcfRadius", light.pcfRadius, 0,
                              4)) {
        return false;
    }
    return true;
}

bool parsePointLights(const JsonValue& root, std::vector<PointLight>& lights,
                      Validator& validator) {
    const JsonValue* array = validator.require(root, "pointLights", "pointLights");
    if (array == nullptr || !validator.requireArray(*array, "pointLights")) {
        return false;
    }
    if (array->asArray().size() > static_cast<std::size_t>(kMaxPointLights)) {
        return validator.fail(ErrorCode::InvalidValue, "at most 8 point lights are supported",
                              "pointLights");
    }
    lights.reserve(array->asArray().size());
    for (std::size_t i = 0; i < array->asArray().size(); ++i) {
        const std::string path = "pointLights[" + std::to_string(i) + "]";
        const JsonValue& node = array->asArray()[i];
        if (!validator.requireObject(node, path)) {
            return false;
        }
        PointLight light;
        const JsonValue* value = validator.require(node, "position", path + ".position");
        if (value == nullptr || !validator.requireVec3(*value, path + ".position",
                                                       light.position)) {
            return false;
        }
        value = validator.require(node, "color", path + ".color");
        if (value == nullptr ||
            !validator.requireVec3(*value, path + ".color", light.color, 0.0F, 1.0F)) {
            return false;
        }
        value = validator.require(node, "intensity", path + ".intensity");
        if (value == nullptr ||
            !validator.requireFloat(*value, path + ".intensity", light.intensity, 0.0F,
                                    1.0e6F)) {
            return false;
        }
        lights.push_back(light);
    }
    return true;
}

bool parseNormalTexture(const JsonValue& node, const std::string& path,
                        NormalTexture& texture, Validator& validator) {
    if (!validator.requireObject(node, path)) {
        return false;
    }
    const JsonValue* value = validator.require(node, "width", path + ".width");
    if (value == nullptr ||
        !validator.requireInt(*value, path + ".width", texture.width, 1, 2048)) {
        return false;
    }
    value = validator.require(node, "height", path + ".height");
    if (value == nullptr ||
        !validator.requireInt(*value, path + ".height", texture.height, 1, 2048)) {
        return false;
    }
    value = validator.require(node, "strength", path + ".strength");
    if (value == nullptr ||
        !validator.requireFloat(*value, path + ".strength", texture.strength, 0.0F, 4.0F)) {
        return false;
    }
    value = validator.require(node, "data", path + ".data");
    if (value == nullptr || !validator.requireArray(*value, path + ".data")) {
        return false;
    }
    const std::size_t pixelCount =
        static_cast<std::size_t>(texture.width) * static_cast<std::size_t>(texture.height);
    if (value->asArray().size() != pixelCount * 3ULL) {
        return validator.fail(ErrorCode::InvalidValue,
                              "normal texture data length must equal width * height * 3",
                              path + ".data");
    }
    texture.pixels.reserve(pixelCount);
    for (std::size_t i = 0; i < pixelCount; ++i) {
        Vec3 pixel;
        const std::size_t base = i * 3ULL;
        if (!validator.requireFloat(value->asArray()[base],
                                    path + ".data[" + std::to_string(base) + "]", pixel.x,
                                    -1.0e6F, 1.0e6F) ||
            !validator.requireFloat(value->asArray()[base + 1],
                                    path + ".data[" + std::to_string(base + 1) + "]", pixel.y,
                                    -1.0e6F, 1.0e6F) ||
            !validator.requireFloat(value->asArray()[base + 2],
                                    path + ".data[" + std::to_string(base + 2) + "]", pixel.z,
                                    -1.0e6F, 1.0e6F)) {
            return false;
        }
        texture.pixels.push_back(pixel);
    }
    return true;
}

bool parseMaterials(const JsonValue& root, std::vector<Material>& materials,
                    Validator& validator) {
    const JsonValue* array = validator.require(root, "materials", "materials");
    if (array == nullptr || !validator.requireArray(*array, "materials")) {
        return false;
    }
    if (array->asArray().empty()) {
        return validator.fail(ErrorCode::InvalidValue, "at least one material is required",
                              "materials");
    }
    materials.reserve(array->asArray().size());
    for (std::size_t i = 0; i < array->asArray().size(); ++i) {
        const std::string path = "materials[" + std::to_string(i) + "]";
        const JsonValue& node = array->asArray()[i];
        if (!validator.requireObject(node, path)) {
            return false;
        }
        Material material;
        const JsonValue* value = validator.require(node, "name", path + ".name");
        if (value == nullptr || !validator.requireString(*value, path + ".name", material.name)) {
            return false;
        }
        value = validator.require(node, "baseColor", path + ".baseColor");
        if (value == nullptr ||
            !validator.requireVec3(*value, path + ".baseColor", material.baseColor, 0.0F,
                                   1.0F)) {
            return false;
        }
        value = validator.require(node, "specularColor", path + ".specularColor");
        if (value == nullptr ||
            !validator.requireVec3(*value, path + ".specularColor", material.specularColor,
                                   0.0F, 1.0F)) {
            return false;
        }
        value = validator.require(node, "shininess", path + ".shininess");
        if (value == nullptr ||
            !validator.requireFloat(*value, path + ".shininess", material.shininess, 0.0F,
                                    1.0e6F, true)) {
            return false;
        }
        value = validator.require(node, "opacity", path + ".opacity");
        if (value == nullptr ||
            !validator.requireFloat(*value, path + ".opacity", material.opacity, 0.0F, 1.0F)) {
            return false;
        }
        value = validator.optional(node, "normalMap");
        if (value != nullptr) {
            NormalTexture texture;
            if (!parseNormalTexture(*value, path + ".normalMap", texture, validator)) {
                return false;
            }
            material.normalTexture = std::move(texture);
        }
        materials.push_back(std::move(material));
    }
    return true;
}

bool parseMesh(const JsonValue& node, const std::string& path, Mesh& mesh,
               Validator& validator) {
    if (!validator.requireObject(node, path)) {
        return false;
    }
    const JsonValue* value = validator.require(node, "name", path + ".name");
    if (value == nullptr || !validator.requireString(*value, path + ".name", mesh.name)) {
        return false;
    }
    value = validator.require(node, "vertices", path + ".vertices");
    if (value == nullptr || !validator.requireArray(*value, path + ".vertices")) {
        return false;
    }
    const auto& vertices = value->asArray();
    if (vertices.empty() || vertices.size() > kMaxVertices) {
        return validator.fail(ErrorCode::InvalidValue,
                              "mesh vertex count must be between 1 and 2,000,000",
                              path + ".vertices");
    }
    mesh.vertices.reserve(vertices.size());
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const std::string vertexPath = path + ".vertices[" + std::to_string(i) + "]";
        if (!validator.requireObject(vertices[i], vertexPath)) {
            return false;
        }
        Vertex vertex;
        const JsonValue* field = validator.require(vertices[i], "position", vertexPath + ".position");
        if (field == nullptr ||
            !validator.requireVec3(*field, vertexPath + ".position", vertex.position)) {
            return false;
        }
        field = validator.require(vertices[i], "normal", vertexPath + ".normal");
        if (field == nullptr ||
            !validator.requireVec3(*field, vertexPath + ".normal", vertex.normal)) {
            return false;
        }
        field = validator.require(vertices[i], "uv", vertexPath + ".uv");
        if (field == nullptr || !validator.requireVec2(*field, vertexPath + ".uv", vertex.uv)) {
            return false;
        }
        mesh.vertices.push_back(vertex);
    }

    value = validator.require(node, "indices", path + ".indices");
    if (value == nullptr || !validator.requireArray(*value, path + ".indices")) {
        return false;
    }
    const auto& indices = value->asArray();
    if (indices.empty() || indices.size() > kMaxIndices || indices.size() % 3ULL != 0ULL) {
        return validator.fail(ErrorCode::InvalidValue,
                              "index count must be non-zero, divisible by 3 and at most 6,000,000",
                              path + ".indices");
    }
    mesh.indices.reserve(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const std::string indexPath = path + ".indices[" + std::to_string(i) + "]";
        std::size_t index = 0;
        if (!validator.requireSize(indices[i], indexPath, index,
                                   std::numeric_limits<std::uint32_t>::max(),
                                   ErrorCode::IndexOutOfRange)) {
            return false;
        }
        if (index >= mesh.vertices.size()) {
            return validator.fail(ErrorCode::IndexOutOfRange,
                                  "mesh index exceeds the vertex count", indexPath);
        }
        mesh.indices.push_back(static_cast<std::uint32_t>(index));
    }
    return true;
}

bool parseMeshes(const JsonValue& root, std::vector<Mesh>& meshes, Validator& validator) {
    const JsonValue* array = validator.require(root, "meshes", "meshes");
    if (array == nullptr || !validator.requireArray(*array, "meshes")) {
        return false;
    }
    if (array->asArray().empty()) {
        return validator.fail(ErrorCode::InvalidValue, "at least one mesh is required", "meshes");
    }
    if (array->asArray().size() > 4096ULL) {
        return validator.fail(ErrorCode::InvalidValue, "at most 4096 meshes are supported",
                              "meshes");
    }
    meshes.reserve(array->asArray().size());
    for (std::size_t i = 0; i < array->asArray().size(); ++i) {
        Mesh mesh;
        if (!parseMesh(array->asArray()[i], "meshes[" + std::to_string(i) + "]", mesh,
                       validator)) {
            return false;
        }
        meshes.push_back(std::move(mesh));
    }
    return true;
}

bool parseObjects(const JsonValue& root, std::vector<SceneObject>& objects,
                  const std::vector<Mesh>& meshes, const std::vector<Material>& materials,
                  Validator& validator) {
    const JsonValue* array = validator.require(root, "objects", "objects");
    if (array == nullptr || !validator.requireArray(*array, "objects")) {
        return false;
    }
    if (array->asArray().empty()) {
        return validator.fail(ErrorCode::InvalidValue, "at least one object is required",
                              "objects");
    }
    if (array->asArray().size() > 65536ULL) {
        return validator.fail(ErrorCode::InvalidValue, "at most 65536 objects are supported",
                              "objects");
    }
    objects.reserve(array->asArray().size());
    for (std::size_t i = 0; i < array->asArray().size(); ++i) {
        const std::string path = "objects[" + std::to_string(i) + "]";
        const JsonValue& node = array->asArray()[i];
        if (!validator.requireObject(node, path)) {
            return false;
        }
        SceneObject object;
        const JsonValue* value = validator.require(node, "name", path + ".name");
        if (value == nullptr || !validator.requireString(*value, path + ".name", object.name)) {
            return false;
        }
        value = validator.require(node, "mesh", path + ".mesh");
        if (value == nullptr ||
            !validator.requireIndex(*value, path + ".mesh", object.meshIndex,
                                    static_cast<int>(meshes.size()) - 1)) {
            return false;
        }
        value = validator.require(node, "material", path + ".material");
        if (value == nullptr ||
            !validator.requireIndex(*value, path + ".material", object.materialIndex,
                                    static_cast<int>(materials.size()) - 1)) {
            return false;
        }
        value = validator.require(node, "model", path + ".model");
        if (value == nullptr || !validator.requireArray(*value, path + ".model")) {
            return false;
        }
        if (value->asArray().size() != 16ULL) {
            return validator.fail(ErrorCode::InvalidValue,
                                  "model matrix must contain exactly 16 row-major values",
                                  path + ".model");
        }
        for (std::size_t element = 0; element < 16; ++element) {
            if (!validator.requireFloat(
                    value->asArray()[element],
                    path + ".model[" + std::to_string(element) + "]",
                    object.model.m[element], -1.0e12F, 1.0e12F)) {
                return false;
            }
        }
        bool invertible = false;
        (void)inverse(object.model, invertible);
        if (!invertible) {
            return validator.fail(ErrorCode::InvalidValue,
                                  "object model matrix must be invertible", path + ".model");
        }
        objects.push_back(object);
    }
    return true;
}

bool parseTransparency(const JsonValue& root, Scene& scene, Validator& validator) {
    const JsonValue* node = validator.require(root, "transparency", "transparency");
    if (node == nullptr || !validator.requireObject(*node, "transparency")) {
        return false;
    }
    const JsonValue* value =
        validator.require(*node, "depthWriteMaxOpacity", "transparency.depthWriteMaxOpacity");
    if (value == nullptr ||
        !validator.requireFloat(*value, "transparency.depthWriteMaxOpacity",
                                scene.transparentDepthWriteMaxOpacity, 0.0F, 1.0F)) {
        return false;
    }
    return true;
}

bool parseLimits(const JsonValue& root, Scene& scene, Validator& validator) {
    const JsonValue* node = validator.optional(root, "limits");
    if (node == nullptr) {
        return true;
    }
    if (!validator.requireObject(*node, "limits")) {
        return false;
    }
    const JsonValue* value = validator.optional(*node, "maxMemoryBytes");
    if (value == nullptr) {
        return true;
    }
    double number = 0.0;
    if (!validator.requireNumber(*value, "limits.maxMemoryBytes", number)) {
        return false;
    }
    if (std::floor(number) != number || number < 1024.0 ||
        number > static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
        return validator.fail(ErrorCode::InvalidValue,
                              "maxMemoryBytes must be a positive integer of at least 1024",
                              "limits.maxMemoryBytes");
    }
    scene.maxMemoryBytes = static_cast<std::uint64_t>(number);
    return true;
}

}  // namespace

SceneLoadResult loadScene(const std::string& path) {
    SceneLoadResult result;
    std::string contents;
    RenderError ioError;
    if (!readFile(path, contents, ioError)) {
        result.error = std::move(ioError);
        return result;
    }
    JsonParseResult parsed = parseJson(contents);
    if (!parsed.value.has_value()) {
        RenderError error = makeError(ErrorCode::JsonParse, parsed.error, path);
        error.message += " at byte offset " + std::to_string(parsed.errorOffset);
        result.error = std::move(error);
        return result;
    }

    Validator validator;
    Scene scene;
    scene.sourceBytes = contents.size();
    const JsonValue& root = *parsed.value;
    if (!validator.requireObject(root, "$")) {
        result.error = std::move(validator.error);
        return result;
    }

    const JsonValue* versionValue = validator.require(root, "version", "version");
    if (versionValue == nullptr) {
        result.error = std::move(validator.error);
        return result;
    }
    int version = 0;
    if (!validator.requireInt(*versionValue, "version", version, 1, 1)) {
        result.error = std::move(validator.error);
        return result;
    }

    if (!parseImage(root, scene, validator) || !parseCamera(root, scene.camera, validator) ||
        !parseDirectionalLight(root, scene.directionalLight, validator) ||
        !parsePointLights(root, scene.pointLights, validator)) {
        result.error = std::move(validator.error);
        return result;
    }

    const JsonValue* ambient = validator.require(root, "ambientColor", "ambientColor");
    if (ambient == nullptr ||
        !validator.requireVec3(*ambient, "ambientColor", scene.ambientColor, 0.0F, 1.0F)) {
        result.error = std::move(validator.error);
        return result;
    }

    const JsonValue* background = validator.optional(root, "background");
    if (background != nullptr &&
        !validator.requireVec3(*background, "background", scene.backgroundColor, 0.0F, 1.0F)) {
        result.error = std::move(validator.error);
        return result;
    }

    if (!parseMaterials(root, scene.materials, validator) ||
        !parseMeshes(root, scene.meshes, validator) ||
        !parseObjects(root, scene.objects, scene.meshes, scene.materials, validator) ||
        !parseTransparency(root, scene, validator) || !parseLimits(root, scene, validator)) {
        result.error = std::move(validator.error);
        return result;
    }

    result.scene = std::move(scene);
    return result;
}

}  // namespace softraster
