#pragma once

#include "error.h"
#include "math.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace softraster {

inline constexpr int kMaxShadowMapSize = 8192;
inline constexpr int kMaxImageDimension = 8192;
inline constexpr std::uint64_t kDefaultMemoryLimitBytes = 1024ULL * 1024ULL * 1024ULL;
inline constexpr std::size_t kMaxSceneFileBytes = 64ULL * 1024ULL * 1024ULL;
inline constexpr std::size_t kMaxVertices = 2'000'000ULL;
inline constexpr std::size_t kMaxIndices = 6'000'000ULL;
inline constexpr int kMaxPointLights = 8;

struct NormalTexture {
    int width = 0;
    int height = 0;
    float strength = 1.0F;
    std::vector<Vec3> pixels;
};

struct Material {
    std::string name;
    Vec3 baseColor{1.0F, 1.0F, 1.0F};
    Vec3 specularColor{1.0F, 1.0F, 1.0F};
    float shininess = 32.0F;
    float opacity = 1.0F;
    std::optional<NormalTexture> normalTexture;

    [[nodiscard]] bool isTransparent() const { return opacity < 1.0F; }
};

struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
};

struct Mesh {
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct SceneObject {
    std::string name;
    int meshIndex = 0;
    int materialIndex = 0;
    Mat4 model = Mat4::identity();
};

struct PointLight {
    Vec3 position;
    Vec3 color{1.0F, 1.0F, 1.0F};
    float intensity = 1.0F;
};

struct DirectionalLight {
    Vec3 directionToLight{0.0F, 1.0F, 0.0F};
    Vec3 color{1.0F, 1.0F, 1.0F};
    float intensity = 1.0F;
    bool shadowEnabled = true;
    int shadowMapSize = 512;
    float depthBias = 0.001F;
    float normalBias = 0.02F;
    int pcfRadius = 1;
};

struct Camera {
    Vec3 position{0.0F, 0.0F, 5.0F};
    Vec3 target{0.0F, 0.0F, 0.0F};
    Vec3 up{0.0F, 1.0F, 0.0F};
    float fovYDegrees = 45.0F;
    float nearPlane = 0.1F;
    float farPlane = 100.0F;
};

struct Viewport {
    int x = 0;
    int y = 0;
    int width = 640;
    int height = 480;
};

struct Scene {
    int imageWidth = 640;
    int imageHeight = 480;
    Viewport viewport;
    Camera camera;
    Vec3 ambientColor{0.02F, 0.02F, 0.02F};
    Vec3 backgroundColor{0.01F, 0.015F, 0.025F};
    DirectionalLight directionalLight;
    std::vector<PointLight> pointLights;
    std::vector<Material> materials;
    std::vector<Mesh> meshes;
    std::vector<SceneObject> objects;
    float transparentDepthWriteMaxOpacity = 0.0F;
    std::uint64_t maxMemoryBytes = kDefaultMemoryLimitBytes;
    std::size_t sourceBytes = 0;
};

struct SceneLoadResult {
    std::optional<Scene> scene;
    std::optional<RenderError> error;
};

[[nodiscard]] SceneLoadResult loadScene(const std::string& path);

}  // namespace softraster
