#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "errors.hpp"
#include "math.hpp"

namespace orv {

struct Camera {
  Vec3 position;
  Vec3 target;
  Vec3 up;
  float fovYDegrees = 45.0f;
  float nearPlane = 0.1f;
  float farPlane = 100.0f;
};

struct Material {
  Vec3 baseColor{1.0f, 1.0f, 1.0f};
  float roughness = 0.5f;
  float metallic = 0.0f;
  float anisotropyStrength = 0.0f;
  float anisotropyRotationRadians = 0.0f;
};

struct Mesh {
  std::string name;
  Mat4 model = Mat4::identity();
  std::vector<Vec3> positions;
  std::vector<Vec3> normals;
  std::vector<Vec3> tangents;
  std::vector<Vec2> uvs;
  std::vector<std::uint32_t> indices;
  Material material;
};

struct DirectionalLight {
  Vec3 direction{-0.4f, -1.0f, -0.3f};
  Vec3 color{1.0f, 1.0f, 1.0f};
  float intensity = 1.0f;
};

struct PointLight {
  Vec3 position;
  Vec3 color{1.0f, 1.0f, 1.0f};
  float intensity = 1.0f;
  float range = 10.0f;
};

struct Scene {
  Camera camera;
  std::vector<Mesh> meshes;
  bool hasDirectionalLight = false;
  DirectionalLight directionalLight;
  std::vector<PointLight> pointLights;

  int lightCount() const {
    return (hasDirectionalLight ? 1 : 0) + static_cast<int>(pointLights.size());
  }
};

bool loadScene(const std::filesystem::path& path, Scene& scene, AppError& error);
bool parseSceneJson(const std::string& text, Scene& scene, AppError& error);

}  // namespace orv
