#pragma once

#include "raster/json.h"
#include "raster/math3d.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace raster {

class SceneError : public std::runtime_error {
public:
    explicit SceneError(const std::string& message) : std::runtime_error(message) {}
};

struct Camera {
    Vec3 position;
    Vec3 direction;
    Vec3 up{0.0, 1.0, 0.0};
    double fovYRadians = 60.0 * kPi / 180.0;
    double nearPlane = 0.1;
    double farPlane = 100.0;
};

struct DirectionalLight {
    Vec3 direction;
    Vec3 color{1.0, 1.0, 1.0};
    double intensity = 1.0;
};

struct Material {
    std::string name;
    Vec3 diffuse{1.0, 1.0, 1.0};
    std::optional<std::string> baseTexture;
    std::optional<std::string> detailTexture;
    Vec3 baseColor{1.0, 1.0, 1.0};
    Vec3 detailColor{1.0, 1.0, 1.0};
    double baseWeight = 1.0;
    double detailWeight = 1.0;
};

struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
    Vec4 color{1.0, 1.0, 1.0, 1.0};
};

struct Mesh {
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::size_t materialIndex = 0;
    Mat4 model = Mat4::identity();
    Mat4 normalMatrix = Mat4::identity();
};

struct Scene {
    int width = 640;
    int height = 480;
    Camera camera;
    std::vector<DirectionalLight> lights;
    std::vector<Material> materials;
    std::vector<Mesh> meshes;
    std::filesystem::path sourceDirectory;
};

Scene parseScene(const JsonValue& root, const std::filesystem::path& sourcePath);

}  // namespace raster

