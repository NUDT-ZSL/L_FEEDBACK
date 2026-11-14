#include "scene.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>

namespace dr {
namespace {

using json = nlohmann::json;

Vec3 readVec3(const json& value, const char* field, const Vec3& fallback) {
    if (!value.contains(field)) {
        return fallback;
    }
    const auto& array = value.at(field);
    if (!array.is_array() || array.size() != 3) {
        throw std::runtime_error(std::string(field) + " must contain three numbers");
    }
    return {array.at(0).get<float>(), array.at(1).get<float>(), array.at(2).get<float>()};
}

void appendSphere(CpuMesh& mesh, int slices, int stacks) {
    slices = std::max(slices, 3);
    stacks = std::max(stacks, 2);

    for (int stack = 0; stack <= stacks; ++stack) {
        const float v = static_cast<float>(stack) / static_cast<float>(stacks);
        const float phi = v * 3.14159265358979323846f;
        const float y = std::cos(phi);
        const float ringRadius = std::sin(phi);
        for (int slice = 0; slice <= slices; ++slice) {
            const float u = static_cast<float>(slice) / static_cast<float>(slices);
            const float theta = u * 2.0f * 3.14159265358979323846f;
            const float x = std::cos(theta) * ringRadius;
            const float z = std::sin(theta) * ringRadius;
            mesh.vertices.push_back({{x, y, z}, {x, y, z}});
        }
    }

    const int row = slices + 1;
    for (int stack = 0; stack < stacks; ++stack) {
        for (int slice = 0; slice < slices; ++slice) {
            const std::uint32_t topLeft = static_cast<std::uint32_t>(stack * row + slice);
            const std::uint32_t bottomLeft = static_cast<std::uint32_t>((stack + 1) * row + slice);
            mesh.indices.insert(mesh.indices.end(),
                                {topLeft, bottomLeft, topLeft + 1,
                                 topLeft + 1, bottomLeft, bottomLeft + 1});
        }
    }
}

void appendPlane(CpuMesh& mesh, float sizeX, float sizeZ) {
    const float halfX = sizeX * 0.5f;
    const float halfZ = sizeZ * 0.5f;
    mesh.vertices = {
        {{-halfX, 0.0f, -halfZ}, {0.0f, 1.0f, 0.0f}},
        {{halfX, 0.0f, -halfZ}, {0.0f, 1.0f, 0.0f}},
        {{halfX, 0.0f, halfZ}, {0.0f, 1.0f, 0.0f}},
        {{-halfX, 0.0f, halfZ}, {0.0f, 1.0f, 0.0f}},
    };
    mesh.indices = {0, 1, 2, 0, 2, 3};
}

float fract(float value) {
    return value - std::floor(value);
}

} // namespace

std::size_t LightSet::pointCount() const {
    return static_cast<std::size_t>(std::count_if(
        lights.begin(), lights.end(), [](const Light& light) { return light.type == LightType::Point; }));
}

std::size_t LightSet::directionalCount() const {
    return static_cast<std::size_t>(std::count_if(
        lights.begin(), lights.end(), [](const Light& light) { return light.type == LightType::Directional; }));
}

bool loadScene(const std::string& path, Scene& scene, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open scene file: " + path;
        return false;
    }

    try {
        json root;
        input >> root;

        scene = Scene{};
        scene.name = root.value("name", "unnamed");
        scene.sceneId = root.value("scene_id", scene.name);
        scene.lightVersion = root.value("light_version", std::uint64_t{1});
        scene.ambient = readVec3(root, "ambient", scene.ambient);
        scene.exposure = root.value("exposure", 1.0f);

        if (root.contains("camera")) {
            const auto& cameraJson = root.at("camera");
            scene.camera.position = readVec3(cameraJson, "position", scene.camera.position);
            scene.camera.target = readVec3(cameraJson, "target", scene.camera.target);
            scene.camera.up = readVec3(cameraJson, "up", scene.camera.up);
            scene.camera.fovYDegrees = cameraJson.value("fov_y", scene.camera.fovYDegrees);
            scene.camera.nearPlane = cameraJson.value("near", scene.camera.nearPlane);
            scene.camera.farPlane = cameraJson.value("far", scene.camera.farPlane);
        }

        const auto& meshes = root.at("meshes");
        if (!meshes.is_array() || meshes.empty()) {
            throw std::runtime_error("meshes must be a non-empty array");
        }
        for (const auto& meshJson : meshes) {
            MeshDefinition mesh;
            mesh.name = meshJson.value("name", "mesh");
            const std::string primitive = meshJson.value("primitive", "plane");
            if (primitive == "plane") {
                mesh.primitive = PrimitiveType::Plane;
            } else if (primitive == "sphere_grid") {
                mesh.primitive = PrimitiveType::SphereGrid;
            } else {
                throw std::runtime_error("unknown primitive: " + primitive);
            }
            mesh.center = readVec3(meshJson, "center", mesh.center);
            mesh.size = readVec3(meshJson, "size", mesh.size);
            mesh.countX = meshJson.value("count_x", 1);
            mesh.countZ = meshJson.value("count_z", 1);
            mesh.spacing = meshJson.value("spacing", 1.0f);
            mesh.radius = meshJson.value("radius", 0.5f);
            mesh.slices = meshJson.value("slices", 32);
            mesh.stacks = meshJson.value("stacks", 16);
            if (meshJson.contains("material")) {
                const auto& materialJson = meshJson.at("material");
                mesh.material.albedo = readVec3(materialJson, "albedo", mesh.material.albedo);
                mesh.material.specular = materialJson.value("specular", mesh.material.specular);
            }
            scene.meshes.push_back(std::move(mesh));
        }

        const auto& lights = root.at("lights");
        if (!lights.is_array()) {
            throw std::runtime_error("lights must be an array");
        }
        for (const auto& lightJson : lights) {
            Light light;
            light.id = lightJson.at("id").get<std::string>();
            const std::string type = lightJson.at("type").get<std::string>();
            if (type == "point") {
                light.type = LightType::Point;
                light.positionOrDirection = readVec3(lightJson, "position", {});
                light.radius = lightJson.value("radius", 1.0f);
            } else if (type == "directional") {
                light.type = LightType::Directional;
                light.positionOrDirection = readVec3(lightJson, "direction", {0.0f, -1.0f, 0.0f});
                light.radius = 0.0f;
            } else {
                throw std::runtime_error("unknown light type: " + type);
            }
            light.color = readVec3(lightJson, "color", light.color);
            light.intensity = lightJson.value("intensity", 1.0f);
            scene.lights.push_back(std::move(light));
        }
    } catch (const std::exception& exception) {
        error = std::string("scene parse error: ") + exception.what();
        return false;
    }

    return validateLights(scene.lights, error);
}

bool validateLights(const std::vector<Light>& lights, std::string& error) {
    std::set<std::string> ids;
    std::size_t pointCount = 0;
    std::size_t directionalCount = 0;
    for (const Light& light : lights) {
        if (light.id.empty()) {
            error = "light id must not be empty";
            return false;
        }
        if (!ids.insert(light.id).second) {
            error = "duplicate light id: " + light.id;
            return false;
        }
        if (light.type == LightType::Point) {
            ++pointCount;
            if (!(light.radius > 0.0f)) {
                error = "point light radius must be positive: " + light.id;
                return false;
            }
        } else {
            ++directionalCount;
        }
        if (!(light.intensity >= 0.0f) ||
            light.color.x < 0.0f || light.color.y < 0.0f || light.color.z < 0.0f) {
            error = "light color/intensity must be non-negative: " + light.id;
            return false;
        }
    }
    if (pointCount > static_cast<std::size_t>(kMaxPointLights)) {
        error = "too many point lights";
        return false;
    }
    if (directionalCount > static_cast<std::size_t>(kMaxDirectionalLights)) {
        error = "too many directional lights";
        return false;
    }
    return true;
}

SceneAssets buildSceneAssets(const Scene& scene) {
    SceneAssets assets;
    assets.viewProjection =
        Mat4::perspective(scene.camera.fovYDegrees,
                          static_cast<float>(kRenderWidth) / static_cast<float>(kRenderHeight),
                          scene.camera.nearPlane,
                          scene.camera.farPlane) *
        Mat4::lookAt(scene.camera.position, scene.camera.target, scene.camera.up);

    for (const MeshDefinition& definition : scene.meshes) {
        CpuMesh mesh;
        if (definition.primitive == PrimitiveType::Plane) {
            appendPlane(mesh, definition.size.x, definition.size.z);
            const std::size_t meshIndex = assets.meshes.size();
            assets.meshes.push_back(std::move(mesh));
            assets.objects.push_back(
                {meshIndex, Mat4::translationScale(definition.center, 1.0f), definition.material});
        } else {
            appendSphere(mesh, definition.slices, definition.stacks);
            const std::size_t meshIndex = assets.meshes.size();
            assets.meshes.push_back(std::move(mesh));

            const int countX = std::max(definition.countX, 1);
            const int countZ = std::max(definition.countZ, 1);
            for (int z = 0; z < countZ; ++z) {
                for (int x = 0; x < countX; ++x) {
                    const float offsetX =
                        (static_cast<float>(x) - static_cast<float>(countX - 1) * 0.5f) * definition.spacing;
                    const float offsetZ =
                        (static_cast<float>(z) - static_cast<float>(countZ - 1) * 0.5f) * definition.spacing;
                    const Vec3 position{definition.center.x + offsetX,
                                        definition.center.y,
                                        definition.center.z + offsetZ};

                    const float tint = fract(static_cast<float>(x) * 0.173f +
                                             static_cast<float>(z) * 0.097f);
                    Material material = definition.material;
                    material.albedo.x = std::min(1.0f, material.albedo.x * (0.72f + tint * 0.38f));
                    material.albedo.y =
                        std::min(1.0f, material.albedo.y * (0.80f + (1.0f - tint) * 0.30f));
                    material.albedo.z =
                        std::min(1.0f, material.albedo.z * (0.76f + fract(tint * 3.7f) * 0.34f));
                    assets.objects.push_back(
                        {meshIndex, Mat4::translationScale(position, definition.radius), material});
                }
            }
        }
    }
    return assets;
}

LightSet lightSetFromScene(const Scene& scene) {
    return {scene.lights, scene.lightVersion};
}

} // namespace dr
