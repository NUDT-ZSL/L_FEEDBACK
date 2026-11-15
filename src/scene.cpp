#include "scene.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace pbr {
namespace {

using json = nlohmann::json;

class ParseError : public std::runtime_error {
public:
    explicit ParseError(const std::string& message) : std::runtime_error(message) {}
};

[[noreturn]] void fail(const std::string& path, const std::string& message) {
    throw ParseError("'" + path + "': " + message);
}

const json& requireMember(const json& object, const char* key, const std::string& path) {
    if (!object.is_object()) {
        fail(path, "expected object");
    }
    const auto it = object.find(key);
    if (it == object.end()) {
        fail(path + "." + key, "missing required field");
    }
    return *it;
}

const json& requireObject(const json& object, const char* key, const std::string& path) {
    const json& value = requireMember(object, key, path);
    if (!value.is_object()) {
        fail(path + "." + key, "expected object");
    }
    return value;
}

const json& requireArray(const json& object, const char* key, const std::string& path) {
    const json& value = requireMember(object, key, path);
    if (!value.is_array()) {
        fail(path + "." + key, "expected array");
    }
    return value;
}

std::string requireString(const json& object, const char* key, const std::string& path) {
    const json& value = requireMember(object, key, path);
    if (!value.is_string()) {
        fail(path + "." + key, "expected string");
    }
    return value.get<std::string>();
}

float requireNumber(const json& object, const char* key, const std::string& path) {
    const json& value = requireMember(object, key, path);
    if (!value.is_number()) {
        fail(path + "." + key, "expected number");
    }
    const double parsed = value.get<double>();
    if (!std::isfinite(parsed)) {
        fail(path + "." + key, "number must be finite");
    }
    return static_cast<float>(parsed);
}

int requireInt(const json& object, const char* key, const std::string& path) {
    const json& value = requireMember(object, key, path);
    if (!value.is_number_integer()) {
        fail(path + "." + key, "expected integer");
    }
    return value.get<int>();
}

Vec3 parseVec3Value(const json& value, const std::string& path) {
    if (!value.is_array() || value.size() != 3) {
        fail(path, "expected array of three numbers");
    }
    Vec3 result;
    const char* components[3] = {"x", "y", "z"};
    float* destination[3] = {&result.x, &result.y, &result.z};
    for (int i = 0; i < 3; ++i) {
        if (!value[static_cast<size_t>(i)].is_number()) {
            fail(path + "." + components[i], "expected number");
        }
        const double parsed = value[static_cast<size_t>(i)].get<double>();
        if (!std::isfinite(parsed)) {
            fail(path + "." + components[i], "number must be finite");
        }
        *destination[i] = static_cast<float>(parsed);
    }
    return result;
}

Vec3 requireVec3(const json& object, const char* key, const std::string& path) {
    return parseVec3Value(requireMember(object, key, path), path + "." + key);
}

Vec2 requireSize2(const json& object, const char* key, const std::string& path) {
    const json& value = requireMember(object, key, path);
    if (!value.is_array() || value.size() != 2) {
        fail(path + "." + key, "expected array of two numbers");
    }
    Vec2 result;
    for (int i = 0; i < 2; ++i) {
        if (!value[static_cast<size_t>(i)].is_number()) {
            fail(path + "." + key, "expected numbers");
        }
        const double parsed = value[static_cast<size_t>(i)].get<double>();
        if (!std::isfinite(parsed)) {
            fail(path + "." + key, "numbers must be finite");
        }
        (i == 0 ? result.x : result.y) = static_cast<float>(parsed);
    }
    return result;
}

float nonNegative(float value, const std::string& path) {
    if (value < 0.0f) {
        fail(path, "must be non-negative");
    }
    return value;
}

float unitRange(float value, const std::string& path) {
    if (value < 0.0f || value > 1.0f) {
        fail(path, "must be in [0, 1]");
    }
    return value;
}

void parseCamera(const json& root, Camera& camera) {
    const json& object = requireObject(root, "camera", "scene");
    camera.position = requireVec3(object, "position", "scene.camera");
    camera.target = requireVec3(object, "target", "scene.camera");
    camera.up = requireVec3(object, "up", "scene.camera");
    camera.focalLengthMm = requireNumber(object, "focal_length_mm", "scene.camera");
    const Vec2 sensor = requireSize2(object, "sensor_size_mm", "scene.camera");
    camera.sensorWidthMm = sensor.x;
    camera.sensorHeightMm = sensor.y;
    camera.fNumber = requireNumber(object, "f_number", "scene.camera");
    camera.focusDistance = requireNumber(object, "focus_distance", "scene.camera");
    camera.exposureTime = requireNumber(object, "exposure_time", "scene.camera");

    if (length(camera.target - camera.position) < 1.0e-6f) {
        fail("scene.camera.target", "must differ from camera position");
    }
    if (lengthSquared(cross(normalize(camera.target - camera.position), normalize(camera.up))) < 1.0e-10f) {
        fail("scene.camera.up", "must not be parallel to view direction");
    }
    if (camera.focalLengthMm <= 0.0f) {
        fail("scene.camera.focal_length_mm", "must be positive");
    }
    if (camera.sensorWidthMm <= 0.0f || camera.sensorHeightMm <= 0.0f) {
        fail("scene.camera.sensor_size_mm", "dimensions must be positive");
    }
    if (camera.fNumber <= 0.0f) {
        fail("scene.camera.f_number", "must be positive");
    }
    if (camera.focusDistance <= 0.0f) {
        fail("scene.camera.focus_distance", "must be positive");
    }
    if (camera.exposureTime <= 0.0f) {
        fail("scene.camera.exposure_time", "must be positive");
    }
}

void parseMotion(const json& root, Motion& motion) {
    if (!root.contains("motion")) {
        return;
    }
    const json& object = requireObject(root, "motion", "scene");
    if (object.contains("lateral")) {
        motion.lateral = requireNumber(object, "lateral", "scene.motion");
    }
    if (object.contains("vertical")) {
        motion.vertical = requireNumber(object, "vertical", "scene.motion");
    }
    if (object.contains("depth")) {
        motion.depth = requireNumber(object, "depth", "scene.motion");
    }
    if (object.contains("cycles")) {
        motion.cycles = requireNumber(object, "cycles", "scene.motion");
    }
    if (motion.cycles <= 0.0f) {
        fail("scene.motion.cycles", "must be positive");
    }
}

void parseEnvironment(const json& root, Environment& environment) {
    if (!root.contains("environment")) {
        return;
    }
    const json& object = requireObject(root, "environment", "scene");
    if (object.contains("zenith")) {
        environment.zenith = requireVec3(object, "zenith", "scene.environment");
    }
    if (object.contains("horizon")) {
        environment.horizon = requireVec3(object, "horizon", "scene.environment");
    }
    if (object.contains("ground")) {
        environment.ground = requireVec3(object, "ground", "scene.environment");
    }
    if (object.contains("sun_direction")) {
        environment.sunDirection = normalize(requireVec3(object, "sun_direction", "scene.environment"));
    }
    if (lengthSquared(environment.sunDirection) < 1.0e-8f) {
        fail("scene.environment.sun_direction", "must be non-zero");
    }
    if (object.contains("sun_radiance")) {
        environment.sunRadiance = requireVec3(object, "sun_radiance", "scene.environment");
    }
    if (object.contains("sun_angular_radius_degrees")) {
        environment.sunAngularRadiusDegrees =
            requireNumber(object, "sun_angular_radius_degrees", "scene.environment");
        if (environment.sunAngularRadiusDegrees <= 0.0f || environment.sunAngularRadiusDegrees >= 30.0f) {
            fail("scene.environment.sun_angular_radius_degrees", "must be in (0, 30)");
        }
    }
    nonNegative(environment.zenith.x, "scene.environment.zenith");
    nonNegative(environment.zenith.y, "scene.environment.zenith");
    nonNegative(environment.zenith.z, "scene.environment.zenith");
    nonNegative(environment.horizon.x, "scene.environment.horizon");
    nonNegative(environment.horizon.y, "scene.environment.horizon");
    nonNegative(environment.horizon.z, "scene.environment.horizon");
    nonNegative(environment.ground.x, "scene.environment.ground");
    nonNegative(environment.ground.y, "scene.environment.ground");
    nonNegative(environment.ground.z, "scene.environment.ground");
    nonNegative(environment.sunRadiance.x, "scene.environment.sun_radiance");
    nonNegative(environment.sunRadiance.y, "scene.environment.sun_radiance");
    nonNegative(environment.sunRadiance.z, "scene.environment.sun_radiance");
}

void parseMaterials(const json& root, Scene& scene) {
    const json& array = requireArray(root, "materials", "scene");
    if (array.empty() || array.size() > static_cast<size_t>(kMaxMaterials)) {
        fail("scene.materials", "must contain 1..16 materials");
    }

    std::unordered_map<std::string, int> nameToId;
    for (size_t i = 0; i < array.size(); ++i) {
        const std::string path = "scene.materials[" + std::to_string(i) + "]";
        const json& object = array[i];
        if (!object.is_object()) {
            fail(path, "expected object");
        }
        Material material;
        material.name = requireString(object, "name", path);
        if (material.name.empty()) {
            fail(path + ".name", "must not be empty");
        }
        if (nameToId.count(material.name) != 0) {
            fail(path + ".name", "duplicate material name '" + material.name + "'");
        }
        material.baseColor = requireVec3(object, "base_color", path);
        material.metallic = unitRange(requireNumber(object, "metallic", path), path + ".metallic");
        material.roughness = requireNumber(object, "roughness", path);
        if (material.roughness < 0.02f || material.roughness > 1.0f) {
            fail(path + ".roughness", "must be in [0.02, 1]");
        }
        material.anisotropy = unitRange(requireNumber(object, "anisotropy", path), path + ".anisotropy");
        if (object.contains("anisotropy_rotation_degrees")) {
            const float degrees = requireNumber(object, "anisotropy_rotation_degrees", path);
            material.anisotropyRotationRadians = degrees * static_cast<float>(kPi / 180.0);
        }
        if (object.contains("ior")) {
            material.ior = requireNumber(object, "ior", path);
            if (material.ior <= 1.0f || material.ior > 3.0f) {
                fail(path + ".ior", "must be in (1, 3]");
            }
        }
        if (material.baseColor.x < 0.0f || material.baseColor.y < 0.0f || material.baseColor.z < 0.0f) {
            fail(path + ".base_color", "components must be non-negative");
        }
        nameToId.emplace(material.name, static_cast<int>(i));
        scene.materials.push_back(material);
    }
    scene.materials.shrink_to_fit();

    // The map is consumed below while parsing spheres. Store it temporarily on the
    // scene's material vector through the parser function's second pass.
    const json& spheres = requireArray(root, "spheres", "scene");
    if (spheres.empty() || spheres.size() > static_cast<size_t>(kMaxSpheres)) {
        fail("scene.spheres", "must contain 1..24 spheres");
    }
    for (size_t i = 0; i < spheres.size(); ++i) {
        const std::string path = "scene.spheres[" + std::to_string(i) + "]";
        const json& object = spheres[i];
        if (!object.is_object()) {
            fail(path, "expected object");
        }
        Sphere sphere;
        sphere.basePosition = requireVec3(object, "position", path);
        sphere.radius = requireNumber(object, "radius", path);
        if (sphere.radius <= 0.0f) {
            fail(path + ".radius", "must be positive");
        }

        const bool hasId = object.contains("material_id");
        const bool hasName = object.contains("material");
        if (!hasId && !hasName) {
            fail(path, "missing required field 'material_id' or 'material'");
        }
        if (hasId) {
            sphere.materialId = requireInt(object, "material_id", path);
            if (sphere.materialId < 0 || sphere.materialId >= static_cast<int>(scene.materials.size())) {
                fail(path + ".material_id",
                     "material ID " + std::to_string(sphere.materialId) + " is out of range [0, " +
                         std::to_string(scene.materials.size() - 1) + "]");
            }
        }
        if (hasName) {
            const std::string materialName = requireString(object, "material", path);
            const auto found = nameToId.find(materialName);
            if (found == nameToId.end()) {
                fail(path + ".material", "unknown material '" + materialName + "'");
            }
            if (hasId && sphere.materialId != found->second) {
                fail(path, "material_id and material name refer to different materials");
            }
            sphere.materialId = found->second;
        }
        scene.spheres.push_back(sphere);
    }
}

void parseLights(const json& root, Scene& scene) {
    const json& array = requireArray(root, "lights", "scene");
    for (size_t i = 0; i < array.size(); ++i) {
        const std::string path = "scene.lights[" + std::to_string(i) + "]";
        const json& object = array[i];
        if (!object.is_object()) {
            fail(path, "expected object");
        }
        const std::string type = requireString(object, "type", path);
        if (type == "point") {
            if (scene.pointLights.size() >= static_cast<size_t>(kMaxPointLights)) {
                fail(path, "too many point lights (maximum 8)");
            }
            PointLight light;
            light.position = requireVec3(object, "position", path);
            light.color = requireVec3(object, "color", path);
            light.power = requireNumber(object, "power", path);
            if (light.power < 0.0f) {
                fail(path + ".power", "must be non-negative");
            }
            if (light.color.x < 0.0f || light.color.y < 0.0f || light.color.z < 0.0f) {
                fail(path + ".color", "components must be non-negative");
            }
            scene.pointLights.push_back(light);
        } else if (type == "rect") {
            if (scene.rectLights.size() >= static_cast<size_t>(kMaxRectLights)) {
                fail(path, "too many rectangular lights (maximum 4)");
            }
            RectLight light;
            light.center = requireVec3(object, "center", path);
            light.normal = normalize(requireVec3(object, "normal", path));
            light.tangent = normalize(requireVec3(object, "tangent", path));
            const Vec2 halfSize = requireSize2(object, "half_size", path);
            light.halfWidth = halfSize.x;
            light.halfHeight = halfSize.y;
            light.radiance = requireVec3(object, "radiance", path);
            if (lengthSquared(light.normal) < 1.0e-8f) {
                fail(path + ".normal", "must be non-zero");
            }
            if (lengthSquared(light.tangent) < 1.0e-8f) {
                fail(path + ".tangent", "must be non-zero");
            }
            if (std::fabs(dot(light.normal, light.tangent)) > 1.0e-4f) {
                fail(path + ".tangent", "must be perpendicular to normal");
            }
            if (light.halfWidth <= 0.0f || light.halfHeight <= 0.0f) {
                fail(path + ".half_size", "components must be positive");
            }
            if (light.radiance.x < 0.0f || light.radiance.y < 0.0f || light.radiance.z < 0.0f) {
                fail(path + ".radiance", "components must be non-negative");
            }
            light.bitangent = normalize(cross(light.normal, light.tangent));
            scene.rectLights.push_back(light);
        } else {
            fail(path + ".type", "expected 'point' or 'rect'");
        }
    }
}

}  // namespace

bool loadScene(const std::string& path, Scene& scene, std::string& error) {
    scene = Scene{};
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "Scene load error: cannot open '" + path + "'";
        return false;
    }
    try {
        json root;
        input >> root;
        if (!root.is_object()) {
            fail("scene", "root must be an object");
        }
        parseCamera(root, scene.camera);
        parseMotion(root, scene.motion);
        parseEnvironment(root, scene.environment);
        parseMaterials(root, scene);
        parseLights(root, scene);
    } catch (const json::exception& exception) {
        error = "Scene load error: JSON parse/type failure in '" + path + "': " + exception.what();
        return false;
    } catch (const std::exception& exception) {
        error = "Scene load error: " + std::string(exception.what());
        return false;
    }
    return true;
}

Vec3 spherePositionAt(const Scene& scene, int sphereIndex, float normalizedTime) {
    const Sphere& sphere = scene.spheres[static_cast<size_t>(sphereIndex)];
    const float phase = 2.0f * static_cast<float>(kPi) *
                        (normalizedTime * scene.motion.cycles + static_cast<float>(sphereIndex) * 0.137f);
    const float shutterSeconds = scene.camera.exposureTime;
    Vec3 position = sphere.basePosition;
    position.x += scene.motion.lateral * shutterSeconds * std::sin(phase);
    position.y += scene.motion.vertical * shutterSeconds * std::sin(phase * 0.75f + 1.1f);
    position.z += scene.motion.depth * shutterSeconds * std::sin(phase * 0.5f + 0.4f);
    return position;
}

}  // namespace pbr
