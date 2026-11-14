#pragma once

#include "math.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dr {

constexpr int kRenderWidth = 800;
constexpr int kRenderHeight = 600;
constexpr int kMaxPointLights = 96;
constexpr int kMaxDirectionalLights = 16;
constexpr std::uint32_t kSnapshotFormatVersion = 1;

enum class LightType : std::uint32_t {
    Point = 0,
    Directional = 1,
};

struct Light {
    std::string id;
    LightType type = LightType::Point;
    Vec3 positionOrDirection{};
    Vec3 color{1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    float radius = 1.0f;
};

struct Camera {
    Vec3 position{0.0f, 7.0f, 14.0f};
    Vec3 target{0.0f, 0.0f, 0.0f};
    Vec3 up{0.0f, 1.0f, 0.0f};
    float fovYDegrees = 45.0f;
    float nearPlane = 0.1f;
    float farPlane = 200.0f;
};

struct Material {
    Vec3 albedo{0.5f, 0.5f, 0.5f};
    float specular = 0.2f;
};

enum class PrimitiveType {
    Plane,
    SphereGrid,
};

struct MeshDefinition {
    std::string name;
    PrimitiveType primitive = PrimitiveType::Plane;
    Vec3 center{};
    Vec3 size{1.0f, 1.0f, 1.0f};
    int countX = 1;
    int countZ = 1;
    float spacing = 1.0f;
    float radius = 0.5f;
    int slices = 32;
    int stacks = 16;
    Material material;
};

struct Scene {
    std::string name;
    std::string sceneId;
    Camera camera;
    Vec3 ambient{0.02f, 0.025f, 0.035f};
    float exposure = 1.0f;
    std::vector<MeshDefinition> meshes;
    std::vector<Light> lights;
    std::uint64_t lightVersion = 1;
};

struct LightSet {
    std::vector<Light> lights;
    std::uint64_t version = 1;

    std::size_t pointCount() const;
    std::size_t directionalCount() const;
};

struct GBufferCapture {
    int width = 0;
    int height = 0;
    std::vector<float> world;
    std::vector<float> normal;
    std::vector<std::uint8_t> albedo;
    std::vector<float> depth;
    std::vector<std::uint8_t> resultRgbaTopDown;
};

struct SnapshotData {
    std::uint32_t formatVersion = kSnapshotFormatVersion;
    std::uint64_t lightVersion = 1;
    int width = 0;
    int height = 0;
    std::string sceneId;
    Vec3 cameraPosition{};
    Vec3 ambient{};
    float exposure = 1.0f;
    std::vector<Light> lights;
    GBufferCapture gbuffer;
};

struct SnapshotError {
    std::string code;
    std::string message;
    std::uint32_t formatVersion = 0;
    std::uint64_t lightVersion = 0;
    std::string path;
};

struct ImageDiff {
    int width = 0;
    int height = 0;
    int maxChannelDifference = 0;
    std::uint64_t differentPixels = 0;
    double meanAbsoluteDifference = 0.0;
};

struct FrameTiming {
    int frame = 0;
    std::string mode;
    double cpuMs = 0.0;
    double gpuMs = 0.0;
    std::size_t pointLights = 0;
    std::size_t directionalLights = 0;
    std::uint64_t lightVersion = 1;
};

} // namespace dr
