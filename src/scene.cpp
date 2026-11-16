#include "renderer/scene.hpp"

#include "renderer/error.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace renderer {
namespace {

using Json = nlohmann::json;
using OrderedJson = nlohmann::ordered_json;

constexpr std::size_t kMaxLights = 32;
constexpr int kPointCubeFaces = 6;

[[noreturn]] void invalid(const std::string& path, const std::string& message) {
    fail("E_INVALID_SCENE", path + ": " + message);
}

const Json& require_member(const Json& object, const char* key, const std::string& path) {
    if (!object.is_object()) {
        invalid(path, "expected an object");
    }
    const auto iterator = object.find(key);
    if (iterator == object.end()) {
        invalid(path + "." + key, "missing required field");
    }
    return *iterator;
}

double parse_number(const Json& value, const std::string& path) {
    if (!value.is_number()) {
        invalid(path, "expected a number");
    }
    const double number = value.get<double>();
    if (!std::isfinite(number)) {
        invalid(path, "number must be finite");
    }
    return number;
}

int parse_int(const Json& value, const std::string& path) {
    if (!value.is_number_integer()) {
        invalid(path, "expected an integer");
    }
    try {
        return value.get<int>();
    } catch (const std::exception&) {
        invalid(path, "integer is outside the supported range");
    }
}

bool parse_bool(const Json& value, const std::string& path) {
    if (!value.is_boolean()) {
        invalid(path, "expected a boolean");
    }
    return value.get<bool>();
}

std::string parse_string(const Json& value, const std::string& path) {
    if (!value.is_string()) {
        invalid(path, "expected a string");
    }
    const std::string result = value.get<std::string>();
    if (result.empty()) {
        invalid(path, "must not be empty");
    }
    return result;
}

Vec3 parse_vec3(const Json& value, const std::string& path) {
    if (!value.is_array() || value.size() != 3) {
        invalid(path, "expected an array of three numbers");
    }
    return {
        static_cast<float>(parse_number(value[0], path + "[0]")),
        static_cast<float>(parse_number(value[1], path + "[1]")),
        static_cast<float>(parse_number(value[2], path + "[2]")),
    };
}

Vec3 parse_direction(const Json& value, const std::string& path) {
    const Vec3 result = parse_vec3(value, path);
    if (length(result) <= 1.0e-8f) {
        invalid(path, "direction must be non-zero");
    }
    return normalize(result);
}

std::array<float, 16> parse_transform(const Json& value, const std::string& path) {
    if (!value.is_array() || value.size() != 16) {
        invalid(path, "expected an array of 16 row-major numbers");
    }
    std::array<float, 16> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<float>(
            parse_number(value[index], path + "[" + std::to_string(index) + "]"));
    }
    return result;
}

void require_range(const std::string& path, float lower, float value, float upper,
                   bool includeUpper = true) {
    if (value < lower || (includeUpper ? value > upper : value >= upper)) {
        invalid(path, "value is outside the accepted range");
    }
}

ModeSettings parse_mode(const Json& settings, const char* modeName, int pcfKernel,
                        const std::string& path) {
    const Json& mode = require_member(settings, "modes", path);
    const Json& modeObject = require_member(mode, modeName, path + ".modes");
    const std::string modePath = path + ".modes." + modeName;

    ModeSettings result;
    result.shadowSamples = parse_int(
        require_member(modeObject, "shadowSamples", modePath), modePath + ".shadowSamples");
    result.postprocessIterations = parse_int(
        require_member(modeObject, "postprocessIterations", modePath),
        modePath + ".postprocessIterations");
    result.atlasScale = static_cast<float>(parse_number(
        require_member(modeObject, "atlasScale", modePath), modePath + ".atlasScale"));
    result.gaussianFilter = parse_bool(
        require_member(modeObject, "gaussianFilter", modePath),
        modePath + ".gaussianFilter");

    if (result.shadowSamples < 1 || result.shadowSamples > pcfKernel * pcfKernel) {
        invalid(modePath + ".shadowSamples",
                "must be between 1 and pcfKernel * pcfKernel");
    }
    if (result.postprocessIterations < 1 || result.postprocessIterations > 8) {
        invalid(modePath + ".postprocessIterations", "must be between 1 and 8");
    }
    if (!(result.atlasScale > 0.0f && result.atlasScale <= 1.0f)) {
        invalid(modePath + ".atlasScale", "must be in (0, 1]");
    }
    return result;
}

RenderSettings parse_settings(const Json& root, const std::string& path) {
    const Json& settings = require_member(root, "settings", path);
    const std::string settingsPath = "settings";

    RenderSettings result;
    result.width = parse_int(require_member(settings, "width", settingsPath),
                             settingsPath + ".width");
    result.height = parse_int(require_member(settings, "height", settingsPath),
                              settingsPath + ".height");
    result.output = parse_string(require_member(settings, "output", settingsPath),
                                 settingsPath + ".output");
    result.background = parse_vec3(require_member(settings, "background", settingsPath),
                                   settingsPath + ".background");
    result.exposure = static_cast<float>(
        parse_number(require_member(settings, "exposure", settingsPath),
                     settingsPath + ".exposure"));
    result.toneMapping = parse_string(
        require_member(settings, "toneMapping", settingsPath),
        settingsPath + ".toneMapping");

    if (result.width < 1 || result.width > 16384
        || result.height < 1 || result.height > 16384) {
        invalid(settingsPath + ".width/height", "must be between 1 and 16384");
    }
    require_range(settingsPath + ".background", 0.0f, result.background.x,
                  std::numeric_limits<float>::max());
    require_range(settingsPath + ".background", 0.0f, result.background.y,
                  std::numeric_limits<float>::max());
    require_range(settingsPath + ".background", 0.0f, result.background.z,
                  std::numeric_limits<float>::max());
    if (!(result.exposure > 0.0f && result.exposure <= 100.0f)) {
        invalid(settingsPath + ".exposure", "must be in (0, 100]");
    }
    if (result.toneMapping != "aces" && result.toneMapping != "reinhard") {
        invalid(settingsPath + ".toneMapping", "expected 'aces' or 'reinhard'");
    }

    const Json& atlas = require_member(settings, "shadowAtlas", settingsPath);
    const std::string atlasPath = settingsPath + ".shadowAtlas";
    result.shadowAtlas.tileResolution = parse_int(
        require_member(atlas, "tileResolution", atlasPath),
        atlasPath + ".tileResolution");
    result.shadowAtlas.columns = parse_int(
        require_member(atlas, "columns", atlasPath), atlasPath + ".columns");
    result.shadowAtlas.rows = parse_int(
        require_member(atlas, "rows", atlasPath), atlasPath + ".rows");
    result.shadowAtlas.pcfKernel = parse_int(
        require_member(atlas, "pcfKernel", atlasPath), atlasPath + ".pcfKernel");
    result.shadowAtlas.pcfStep = static_cast<float>(parse_number(
        require_member(atlas, "pcfStep", atlasPath), atlasPath + ".pcfStep"));

    if (result.shadowAtlas.tileResolution < 4
        || result.shadowAtlas.tileResolution > 8192) {
        invalid(atlasPath + ".tileResolution", "must be between 4 and 8192");
    }
    if (result.shadowAtlas.columns < 1 || result.shadowAtlas.columns > 32
        || result.shadowAtlas.rows < 1 || result.shadowAtlas.rows > 32) {
        invalid(atlasPath + ".columns/rows", "must be between 1 and 32");
    }
    if (result.shadowAtlas.pcfKernel < 1 || result.shadowAtlas.pcfKernel > 15
        || result.shadowAtlas.pcfKernel % 2 == 0) {
        invalid(atlasPath + ".pcfKernel", "must be an odd integer between 1 and 15");
    }
    if (!(result.shadowAtlas.pcfStep > 0.0f && result.shadowAtlas.pcfStep <= 8.0f)) {
        invalid(atlasPath + ".pcfStep", "must be in (0, 8]");
    }

    const Json& denoise = require_member(settings, "denoise", settingsPath);
    const std::string denoisePath = settingsPath + ".denoise";
    result.denoise.filterRadius = parse_int(
        require_member(denoise, "filterRadius", denoisePath),
        denoisePath + ".filterRadius");
    const Json& weights = require_member(denoise, "weights", denoisePath);
    if (!weights.is_array() || weights.size() < 3 || weights.size() % 2 == 0
        || weights.size() > 17) {
        invalid(denoisePath + ".weights", "expected an odd array of 3 to 17 numbers");
    }
    float weightSum = 0.0f;
    for (std::size_t index = 0; index < weights.size(); ++index) {
        const float weight = static_cast<float>(
            parse_number(weights[index], denoisePath + ".weights[" + std::to_string(index) + "]"));
        if (weight < 0.0f) {
            invalid(denoisePath + ".weights", "weights must be non-negative");
        }
        weightSum += weight;
        result.denoise.weights.push_back(weight);
    }
    if (weightSum <= 0.0f) {
        invalid(denoisePath + ".weights", "at least one weight must be positive");
    }
    const int impliedRadius = static_cast<int>(result.denoise.weights.size() / 2);
    if (result.denoise.filterRadius != impliedRadius || impliedRadius < 1
        || impliedRadius > 8) {
        invalid(denoisePath + ".filterRadius",
                "must equal (weights.size() - 1) / 2 and be between 1 and 8");
    }

    result.fast = parse_mode(settings, "fast", result.shadowAtlas.pcfKernel,
                             settingsPath);
    result.high = parse_mode(settings, "high", result.shadowAtlas.pcfKernel,
                             settingsPath);

    if (result.fast.gaussianFilter || !result.high.gaussianFilter) {
        invalid(settingsPath + ".modes",
                "fast must skip Gaussian filtering and high must enable it");
    }
    if (result.high.shadowSamples <= result.fast.shadowSamples) {
        invalid(settingsPath + ".modes.high.shadowSamples",
                "high mode must use more shadow samples than fast mode");
    }
    if (result.high.postprocessIterations <= result.fast.postprocessIterations) {
        invalid(settingsPath + ".modes.high.postprocessIterations",
                "high mode must use more post-process iterations than fast mode");
    }
    if (!(result.high.atlasScale > result.fast.atlasScale)) {
        invalid(settingsPath + ".modes.high.atlasScale",
                "high mode must use a larger atlas scale than fast mode");
    }

    for (const ModeSettings* mode : {&result.fast, &result.high}) {
        const int tile = std::max(
            1, static_cast<int>(std::lround(result.shadowAtlas.tileResolution * mode->atlasScale)));
        const long long width = static_cast<long long>(tile) * result.shadowAtlas.columns;
        const long long height = static_cast<long long>(tile) * result.shadowAtlas.rows;
        if (width > 16384 || height > 16384) {
            invalid(atlasPath, "effective atlas dimensions must not exceed 16384");
        }
    }
    return result;
}

Light parse_light(const Json& value, std::size_t index) {
    const std::string path = "lights[" + std::to_string(index) + "]";
    if (!value.is_object()) {
        invalid(path, "expected an object");
    }

    Light light;
    const std::string type = parse_string(require_member(value, "type", path), path + ".type");
    if (type == "point") {
        light.type = LightType::Point;
    } else if (type == "directional") {
        light.type = LightType::Directional;
    } else if (type == "spot") {
        light.type = LightType::Spot;
    } else {
        invalid(path + ".type", "expected 'point', 'directional', or 'spot'");
    }

    light.color = parse_vec3(require_member(value, "color", path), path + ".color");
    require_range(path + ".color", 0.0f, light.color.x,
                  std::numeric_limits<float>::max());
    require_range(path + ".color", 0.0f, light.color.y,
                  std::numeric_limits<float>::max());
    require_range(path + ".color", 0.0f, light.color.z,
                  std::numeric_limits<float>::max());
    light.intensity = static_cast<float>(parse_number(
        require_member(value, "intensity", path), path + ".intensity"));
    if (light.intensity < 0.0f || !std::isfinite(light.intensity)) {
        invalid(path + ".intensity", "must be finite and non-negative");
    }
    light.castShadow = parse_bool(
        require_member(value, "castShadow", path), path + ".castShadow");

    if (light.type == LightType::Point || light.type == LightType::Spot) {
        light.position = parse_vec3(require_member(value, "position", path),
                                    path + ".position");
        light.range = static_cast<float>(parse_number(
            require_member(value, "range", path), path + ".range"));
        if (!(light.range > 0.0f)) {
            invalid(path + ".range", "must be positive");
        }
    }
    if (light.type == LightType::Directional || light.type == LightType::Spot) {
        light.direction = parse_direction(require_member(value, "direction", path),
                                          path + ".direction");
    }
    if (light.type == LightType::Directional) {
        light.orthoSize = static_cast<float>(parse_number(
            require_member(value, "orthoSize", path), path + ".orthoSize"));
        if (!(light.orthoSize > 0.0f)) {
            invalid(path + ".orthoSize", "must be positive");
        }
    }
    if (light.type == LightType::Spot) {
        light.innerConeDegrees = static_cast<float>(parse_number(
            require_member(value, "innerConeDegrees", path),
            path + ".innerConeDegrees"));
        light.outerConeDegrees = static_cast<float>(parse_number(
            require_member(value, "outerConeDegrees", path),
            path + ".outerConeDegrees"));
        if (light.innerConeDegrees < 0.0f
            || light.outerConeDegrees <= light.innerConeDegrees
            || light.outerConeDegrees >= 89.0f) {
            invalid(path + ".outerConeDegrees",
                    "cones must satisfy 0 <= inner < outer < 89 degrees");
        }
    }
    if (light.castShadow) {
        light.nearPlane = static_cast<float>(parse_number(
            require_member(value, "near", path), path + ".near"));
        light.farPlane = static_cast<float>(parse_number(
            require_member(value, "far", path), path + ".far"));
        light.shadowBias = static_cast<float>(parse_number(
            require_member(value, "shadowBias", path), path + ".shadowBias"));
        if (!(light.nearPlane > 0.0f && light.farPlane > light.nearPlane)) {
            invalid(path + ".near/far", "require 0 < near < far");
        }
        if ((light.type == LightType::Point || light.type == LightType::Spot)
            && light.farPlane < light.range) {
            invalid(path + ".far", "shadow far plane must cover the light range");
        }
        if (light.shadowBias < 0.0f || light.shadowBias > 0.25f) {
            invalid(path + ".shadowBias", "must be in [0, 0.25]");
        }
    }
    return light;
}

Mesh parse_mesh(const Json& value, std::size_t index, std::size_t materialCount) {
    const std::string path = "meshes[" + std::to_string(index) + "]";
    if (!value.is_object()) {
        invalid(path, "expected an object");
    }

    Mesh mesh;
    mesh.name = parse_string(require_member(value, "name", path), path + ".name");
    mesh.material = parse_int(require_member(value, "material", path),
                              path + ".material");
    if (mesh.material < 0
        || static_cast<std::size_t>(mesh.material) >= materialCount) {
        invalid(path + ".material", "material index is out of range");
    }

    mesh.transform = mat4FromRowMajor(
        parse_transform(require_member(value, "transform", path), path + ".transform"));
    if (std::abs(determinant3(mesh.transform)) <= 1.0e-12f) {
        invalid(path + ".transform", "transform must be invertible");
    }

    const Json& vertices = require_member(value, "vertices", path);
    if (!vertices.is_array() || vertices.size() < 3) {
        invalid(path + ".vertices", "expected at least three vertices");
    }
    if (vertices.size() > 5000000) {
        invalid(path + ".vertices", "too many vertices");
    }
    mesh.vertices.reserve(vertices.size());
    for (std::size_t vertexIndex = 0; vertexIndex < vertices.size(); ++vertexIndex) {
        const std::string vertexPath =
            path + ".vertices[" + std::to_string(vertexIndex) + "]";
        const Json& vertex = vertices[vertexIndex];
        if (!vertex.is_array() || vertex.size() != 6) {
            invalid(vertexPath, "expected [x, y, z, nx, ny, nz]");
        }
        Vertex parsed;
        parsed.position = {
            static_cast<float>(parse_number(vertex[0], vertexPath + "[0]")),
            static_cast<float>(parse_number(vertex[1], vertexPath + "[1]")),
            static_cast<float>(parse_number(vertex[2], vertexPath + "[2]")),
        };
        parsed.normal = {
            static_cast<float>(parse_number(vertex[3], vertexPath + "[3]")),
            static_cast<float>(parse_number(vertex[4], vertexPath + "[4]")),
            static_cast<float>(parse_number(vertex[5], vertexPath + "[5]")),
        };
        if (length(parsed.normal) <= 1.0e-8f) {
            invalid(vertexPath, "normal must be non-zero");
        }
        parsed.normal = normalize(parsed.normal);
        mesh.vertices.push_back(parsed);
    }

    const Json& indices = require_member(value, "indices", path);
    if (!indices.is_array() || indices.size() < 3 || indices.size() % 3 != 0) {
        invalid(path + ".indices",
                "expected a triangle index array with a multiple-of-three length");
    }
    mesh.indices.reserve(indices.size());
    for (std::size_t indexValue = 0; indexValue < indices.size(); ++indexValue) {
        const std::string indexPath =
            path + ".indices[" + std::to_string(indexValue) + "]";
        const int parsed = parse_int(indices[indexValue], indexPath);
        if (parsed < 0
            || static_cast<std::size_t>(parsed) >= mesh.vertices.size()) {
            invalid(indexPath, "vertex index is out of range");
        }
        mesh.indices.push_back(static_cast<std::uint32_t>(parsed));
    }
    return mesh;
}

void compute_bounds(Scene& scene) {
    scene.bounds = {};
    for (const Mesh& mesh : scene.meshes) {
        for (const Vertex& vertex : mesh.vertices) {
            scene.bounds.expand(transformPoint(mesh.transform, vertex.position));
        }
    }
    if (!scene.bounds.valid()) {
        invalid("meshes", "scene has no valid bounds");
    }
}

std::string read_text_file(const std::string& path) {
    std::ifstream stream(std::filesystem::path(path), std::ios::in | std::ios::binary);
    if (!stream) {
        fail("E_SCENE_IO", "cannot open scene file: " + path);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    if (!stream.good() && !stream.eof()) {
        fail("E_SCENE_IO", "failed while reading scene file: " + path);
    }
    return buffer.str();
}

OrderedJson vec3_json(Vec3 value) {
    return OrderedJson::array({value.x, value.y, value.z});
}

OrderedJson identity_transform_json() {
    return OrderedJson::array({
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    });
}

struct MeshData {
    std::vector<std::array<float, 6>> vertices;
    std::vector<std::uint32_t> indices;
};

MeshData make_plane(float halfX, float halfZ, float y) {
    MeshData mesh;
    mesh.vertices = {
        {-halfX, y, -halfZ, 0.0f, 1.0f, 0.0f},
        {-halfX, y, halfZ, 0.0f, 1.0f, 0.0f},
        {halfX, y, halfZ, 0.0f, 1.0f, 0.0f},
        {halfX, y, -halfZ, 0.0f, 1.0f, 0.0f},
    };
    mesh.indices = {0, 1, 2, 0, 2, 3};
    return mesh;
}

MeshData make_sphere(Vec3 center, float radius, int slices, int stacks) {
    MeshData mesh;
    for (int stack = 0; stack <= stacks; ++stack) {
        const float theta = kPi * static_cast<float>(stack) / static_cast<float>(stacks);
        const float sinTheta = std::sin(theta);
        const float cosTheta = std::cos(theta);
        for (int slice = 0; slice < slices; ++slice) {
            const float phi = 2.0f * kPi * static_cast<float>(slice)
                            / static_cast<float>(slices);
            const Vec3 normal{
                sinTheta * std::cos(phi),
                cosTheta,
                sinTheta * std::sin(phi),
            };
            const Vec3 position{
                center.x + normal.x * radius,
                center.y + normal.y * radius,
                center.z + normal.z * radius,
            };
            mesh.vertices.push_back({
                position.x, position.y, position.z,
                normal.x, normal.y, normal.z,
            });
        }
    }

    for (int stack = 0; stack < stacks; ++stack) {
        for (int slice = 0; slice < slices; ++slice) {
            const std::uint32_t a =
                static_cast<std::uint32_t>(stack * slices + slice);
            const std::uint32_t b =
                static_cast<std::uint32_t>(stack * slices + (slice + 1) % slices);
            const std::uint32_t c =
                static_cast<std::uint32_t>((stack + 1) * slices + slice);
            const std::uint32_t d =
                static_cast<std::uint32_t>((stack + 1) * slices + (slice + 1) % slices);
            mesh.indices.push_back(a);
            mesh.indices.push_back(b);
            mesh.indices.push_back(c);
            mesh.indices.push_back(b);
            mesh.indices.push_back(d);
            mesh.indices.push_back(c);
        }
    }
    return mesh;
}

OrderedJson mesh_json(const std::string& name, int material, const MeshData& mesh) {
    OrderedJson vertices = OrderedJson::array();
    for (const auto& vertex : mesh.vertices) {
        vertices.push_back(OrderedJson::array(
            {vertex[0], vertex[1], vertex[2], vertex[3], vertex[4], vertex[5]}));
    }
    OrderedJson indices = OrderedJson::array();
    for (const std::uint32_t index : mesh.indices) {
        indices.push_back(index);
    }
    return OrderedJson{
        {"name", name},
        {"material", material},
        {"transform", identity_transform_json()},
        {"vertices", std::move(vertices)},
        {"indices", std::move(indices)},
    };
}

void write_json(const std::filesystem::path& path, const OrderedJson& value) {
    std::ofstream stream(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!stream) {
        fail("E_SAMPLE_IO", "cannot write generated sample: " + path.string());
    }
    stream << value.dump(2) << '\n';
    if (!stream) {
        fail("E_SAMPLE_IO", "failed while writing generated sample: " + path.string());
    }
}

}  // namespace

Scene load_scene(const std::string& path) {
    const std::string text = read_text_file(path);
    Json root;
    try {
        root = Json::parse(text);
    } catch (const nlohmann::json::parse_error& error) {
        fail("E_INVALID_SCENE", "JSON parse error: " + std::string(error.what()));
    }
    if (!root.is_object()) {
        invalid("root", "expected a JSON object");
    }

    Scene scene;
    scene.sourcePath = path;
    const int version = parse_int(require_member(root, "version", "root"), "version");
    if (version != 1) {
        invalid("version", "unsupported scene version");
    }

    const Json& camera = require_member(root, "camera", "root");
    scene.camera.position = parse_vec3(require_member(camera, "position", "camera"),
                                       "camera.position");
    scene.camera.target = parse_vec3(require_member(camera, "target", "camera"),
                                     "camera.target");
    scene.camera.up = parse_vec3(require_member(camera, "up", "camera"), "camera.up");
    scene.camera.fovYDegrees = static_cast<float>(parse_number(
        require_member(camera, "fovYDegrees", "camera"), "camera.fovYDegrees"));
    scene.camera.nearPlane = static_cast<float>(parse_number(
        require_member(camera, "near", "camera"), "camera.near"));
    scene.camera.farPlane = static_cast<float>(parse_number(
        require_member(camera, "far", "camera"), "camera.far"));
    if (!(length(scene.camera.up) > 1.0e-8f)) {
        invalid("camera.up", "must be non-zero");
    }
    scene.camera.up = normalize(scene.camera.up);
    const Vec3 view = normalize(scene.camera.target - scene.camera.position);
    if (length(view) <= 1.0e-8f || length(cross(view, scene.camera.up)) <= 1.0e-8f) {
        invalid("camera", "position, target, and up do not define a view");
    }
    if (!(scene.camera.fovYDegrees > 0.0f && scene.camera.fovYDegrees < 180.0f)) {
        invalid("camera.fovYDegrees", "must be in (0, 180)");
    }
    if (!(scene.camera.nearPlane > 0.0f
          && scene.camera.farPlane > scene.camera.nearPlane)) {
        invalid("camera.near/far", "require 0 < near < far");
    }

    const Json& materials = require_member(root, "materials", "root");
    if (!materials.is_array() || materials.empty()) {
        invalid("materials", "expected a non-empty array");
    }
    for (std::size_t index = 0; index < materials.size(); ++index) {
        const std::string materialPath =
            "materials[" + std::to_string(index) + "]";
        const Json& materialJson = materials[index];
        Material material;
        material.name = parse_string(
            require_member(materialJson, "name", materialPath),
            materialPath + ".name");
        material.albedo = parse_vec3(
            require_member(materialJson, "albedo", materialPath),
            materialPath + ".albedo");
        require_range(materialPath + ".albedo", 0.0f, material.albedo.x, 1.0f);
        require_range(materialPath + ".albedo", 0.0f, material.albedo.y, 1.0f);
        require_range(materialPath + ".albedo", 0.0f, material.albedo.z, 1.0f);
        material.metallic = static_cast<float>(parse_number(
            require_member(materialJson, "metallic", materialPath),
            materialPath + ".metallic"));
        material.roughness = static_cast<float>(parse_number(
            require_member(materialJson, "roughness", materialPath),
            materialPath + ".roughness"));
        require_range(materialPath + ".metallic", 0.0f, material.metallic, 1.0f);
        require_range(materialPath + ".roughness", 0.0f, material.roughness, 1.0f);
        scene.materials.push_back(std::move(material));
    }

    const Json& meshes = require_member(root, "meshes", "root");
    if (!meshes.is_array() || meshes.empty()) {
        invalid("meshes", "expected a non-empty array");
    }
    for (std::size_t index = 0; index < meshes.size(); ++index) {
        scene.meshes.push_back(parse_mesh(meshes[index], index, scene.materials.size()));
    }
    compute_bounds(scene);

    const Json& lights = require_member(root, "lights", "root");
    if (!lights.is_array() || lights.empty()) {
        invalid("lights", "expected a non-empty array");
    }
    if (lights.size() > kMaxLights) {
        invalid("lights", "at most 32 lights are supported");
    }
    for (std::size_t index = 0; index < lights.size(); ++index) {
        scene.lights.push_back(parse_light(lights[index], index));
    }

    scene.settings = parse_settings(root, "root");
    if (shadow_tile_demand(scene) > shadow_tile_capacity(scene)) {
        std::ostringstream message;
        message << "shadow atlas has " << shadow_tile_capacity(scene)
                << " tiles but the lights require " << shadow_tile_demand(scene);
        invalid("settings.shadowAtlas", message.str());
    }

    for (std::size_t index = 0; index < scene.lights.size(); ++index) {
        const Light& light = scene.lights[index];
        if (light.castShadow) {
            const int faces = light.type == LightType::Point ? kPointCubeFaces : 1;
            if (light.type == LightType::Directional && light.farPlane <= light.nearPlane) {
                invalid("lights[" + std::to_string(index) + "].far",
                        "invalid directional shadow range");
            }
            (void)faces;
        }
    }
    return scene;
}

void generate_samples(const std::string& directory) {
    const std::filesystem::path root(directory);
    std::error_code error;
    std::filesystem::create_directories(root, error);
    if (error) {
        fail("E_SAMPLE_IO", "cannot create sample directory: " + error.message());
    }

    const MeshData floor = make_plane(7.0f, 5.0f, 0.0f);
    const MeshData largeSphere = make_sphere({-1.35f, 0.95f, 0.0f}, 0.95f, 28, 16);
    const MeshData smallSphere = make_sphere({1.15f, 0.62f, 0.45f}, 0.62f, 24, 14);
    const MeshData backSphere = make_sphere({0.1f, 0.42f, -1.9f}, 0.42f, 20, 12);

    OrderedJson sceneJson = OrderedJson::object();
    sceneJson["version"] = 1;
    sceneJson["camera"] = OrderedJson{
        {"position", vec3_json({4.6f, 3.2f, 7.5f})},
        {"target", vec3_json({0.0f, 1.05f, -0.2f})},
        {"up", vec3_json({0.0f, 1.0f, 0.0f})},
        {"fovYDegrees", 43.0f},
        {"near", 0.1f},
        {"far", 60.0f},
    };
    sceneJson["materials"] = OrderedJson::array({
        OrderedJson{
            {"name", "polished_copper"},
            {"albedo", vec3_json({0.72f, 0.28f, 0.12f})},
            {"metallic", 0.92f},
            {"roughness", 0.24f},
        },
        OrderedJson{
            {"name", "blue_rubber"},
            {"albedo", vec3_json({0.09f, 0.20f, 0.42f})},
            {"metallic", 0.05f},
            {"roughness", 0.68f},
        },
        OrderedJson{
            {"name", "warm_stone"},
            {"albedo", vec3_json({0.34f, 0.33f, 0.30f})},
            {"metallic", 0.0f},
            {"roughness", 0.83f},
        },
        OrderedJson{
            {"name", "green_ceramic"},
            {"albedo", vec3_json({0.12f, 0.43f, 0.28f})},
            {"metallic", 0.15f},
            {"roughness", 0.41f},
        },
    });
    sceneJson["meshes"] = OrderedJson::array({
        mesh_json("floor", 2, floor),
        mesh_json("copper_orb", 0, largeSphere),
        mesh_json("blue_orb", 1, smallSphere),
        mesh_json("ceramic_orb", 3, backSphere),
    });
    sceneJson["lights"] = OrderedJson::array({
        OrderedJson{
            {"type", "directional"},
            {"color", vec3_json({1.0f, 0.90f, 0.72f})},
            {"intensity", 2.4f},
            {"direction", vec3_json(normalize({0.5f, -1.0f, 0.25f}))},
            {"orthoSize", 8.0f},
            {"castShadow", true},
            {"near", 0.1f},
            {"far", 30.0f},
            {"shadowBias", 0.0015f},
        },
        OrderedJson{
            {"type", "point"},
            {"position", vec3_json({2.8f, 2.25f, 1.25f})},
            {"color", vec3_json({0.18f, 0.50f, 1.0f})},
            {"intensity", 22.0f},
            {"range", 9.0f},
            {"castShadow", true},
            {"near", 0.1f},
            {"far", 10.0f},
            {"shadowBias", 0.0025f},
        },
        OrderedJson{
            {"type", "spot"},
            {"position", vec3_json({-2.8f, 3.1f, 0.5f})},
            {"color", vec3_json({1.0f, 0.38f, 0.12f})},
            {"intensity", 38.0f},
            {"range", 12.0f},
            {"direction", vec3_json(normalize({0.65f, -1.0f, -0.15f}))},
            {"innerConeDegrees", 15.0f},
            {"outerConeDegrees", 28.0f},
            {"castShadow", true},
            {"near", 0.1f},
            {"far", 13.0f},
            {"shadowBias", 0.0018f},
        },
    });
    sceneJson["settings"] = OrderedJson{
        {"width", 640},
        {"height", 360},
        {"output", "preview.png"},
        {"background", vec3_json({0.008f, 0.010f, 0.018f})},
        {"exposure", 1.15f},
        {"toneMapping", "aces"},
        {"shadowAtlas", OrderedJson{
            {"tileResolution", 256},
            {"columns", 4},
            {"rows", 4},
            {"pcfKernel", 5},
            {"pcfStep", 1.0f},
        }},
        {"denoise", OrderedJson{
            {"filterRadius", 2},
            {"weights", OrderedJson::array({0.06136f, 0.24477f, 0.38774f,
                                             0.24477f, 0.06136f})},
        }},
        {"modes", OrderedJson{
            {"fast", OrderedJson{
                {"shadowSamples", 1},
                {"postprocessIterations", 1},
                {"atlasScale", 0.5f},
                {"gaussianFilter", false},
            }},
            {"high", OrderedJson{
                {"shadowSamples", 25},
                {"postprocessIterations", 2},
                {"atlasScale", 1.0f},
                {"gaussianFilter", true},
            }},
        }},
    };

    write_json(root / "scene.json", sceneJson);

    OrderedJson missingCamera = sceneJson;
    missingCamera.erase("camera");
    write_json(root / "invalid_missing_camera.json", missingCamera);

    OrderedJson insufficientAtlas = sceneJson;
    insufficientAtlas["settings"]["shadowAtlas"]["columns"] = 1;
    insufficientAtlas["settings"]["shadowAtlas"]["rows"] = 2;
    write_json(root / "invalid_atlas_capacity.json", insufficientAtlas);
}

int shadow_tile_demand(const Scene& scene) {
    int demand = 0;
    for (const Light& light : scene.lights) {
        if (!light.castShadow) {
            continue;
        }
        demand += light.type == LightType::Point ? kPointCubeFaces : 1;
    }
    return demand;
}

int shadow_tile_capacity(const Scene& scene) {
    return scene.settings.shadowAtlas.columns * scene.settings.shadowAtlas.rows;
}

const ModeSettings& mode_settings(const Scene& scene, const std::string& mode) {
    if (mode == "fast") {
        return scene.settings.fast;
    }
    if (mode == "high") {
        return scene.settings.high;
    }
    fail("E_USAGE", "mode must be 'fast' or 'high'");
}

int effective_tile_resolution(const Scene& scene, const std::string& mode) {
    const ModeSettings& settings = mode_settings(scene, mode);
    return std::max(
        1, static_cast<int>(std::lround(
               static_cast<double>(scene.settings.shadowAtlas.tileResolution)
               * static_cast<double>(settings.atlasScale))));
}

bool is_valid_mode(const std::string& mode) {
    return mode == "fast" || mode == "high";
}

}  // namespace renderer
