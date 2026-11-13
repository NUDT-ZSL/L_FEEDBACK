#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace lfb {

constexpr std::size_t kMaxMaterialLayers = 32;
constexpr std::size_t kMaxFrames = 4096;
constexpr int kMaxResolution = 8192;

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Color {
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
};

struct MaterialLayer {
    Color baseColor;
    double roughness = 0.0;
    double normalStrength = 0.0;
    Color emissionColor;
};

struct Light {
    Vec3 direction;
    Color color;
    double intensity = 1.0;
};

struct FrameDesc {
    std::uint32_t index = 0;
    Vec3 direction;
};

struct Scene {
    int width = 0;
    int height = 0;
    std::vector<MaterialLayer> layers;
    std::vector<Light> lights;
    std::vector<FrameDesc> frames;
};

Scene loadScene(const std::filesystem::path& scenePath);

std::string globalCacheKey(const Scene& scene);
std::string frameCacheKey(const Scene& scene, const FrameDesc& frame);

Vec3 normalized(Vec3 value);
double length(Vec3 value);

}  // namespace lfb
