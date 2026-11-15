#pragma once

#include "math.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vlr {

struct Camera {
    Vec3 position{};
    Vec3 target{};
    Vec3 up{};
    float fovYDegrees = 45.0f;
    float nearMeters = 0.1f;
    float farMeters = 40.0f;
};

struct OutputSettings {
    int width = 384;
    int height = 216;
    float exposure = 1.0f;
    float gain = 1.0f;
    std::string pngPath = "render.png";
    std::string statsPath = "stats.json";
    std::string referencePngPath = "render.reference.png";
    Vec3 background{0.01f, 0.02f, 0.035f};
};

struct DirectionalLight {
    Vec3 direction{0.0f, -1.0f, 0.0f};
    Vec3 color{1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
};

struct PointLight {
    Vec3 position{};
    Vec3 color{1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    float rangeMeters = 10.0f;
};

struct VolumeMedium {
    Vec3 boundsMin{};
    Vec3 boundsMax{};
    float density = 0.1f;
    float absorption = 0.05f;
    float scattering = 0.5f;
    float anisotropy = 0.0f;
    float stepMeters = 0.1f;
    int maxSteps = 128;
    float noiseFrequencyPerMeter = 0.15f;
};

struct ShadowSettings {
    int resolution = 512;
    int pcfKernel = 3;
    float directionalBias = 0.0015f;
    float pointBias = 0.0025f;
};

struct Ground {
    float height = 0.0f;
    float sizeMeters = 10.0f;
    Vec3 albedo{0.3f, 0.3f, 0.3f};
};

struct Sphere {
    Vec3 center{};
    float radius = 1.0f;
    Vec3 albedo{0.7f, 0.7f, 0.7f};
};

struct Scene {
    int schemaVersion = 0;
    std::string name;
    std::filesystem::path sourcePath;
    std::filesystem::path sourceDirectory;
    std::string sourceSha256;
    Camera camera;
    OutputSettings output;
    DirectionalLight directionalLight;
    PointLight pointLight;
    VolumeMedium volume;
    ShadowSettings shadows;
    Ground ground;
    std::vector<Sphere> spheres;
    std::filesystem::path noisePath;
    int noiseDimension = 32;
    std::uint32_t noiseSeed = 1337;
};

struct NoiseVolume {
    int dimension = 32;
    std::vector<float> values;
};

Scene loadScene(const std::filesystem::path& path);
NoiseVolume loadNoise(const std::filesystem::path& path, int expectedDimension);
NoiseVolume generateNoise(int dimension, std::uint32_t seed);
std::vector<std::uint8_t> generateNoiseRaw(int dimension, std::uint32_t seed);

std::filesystem::path resolveRelative(const std::filesystem::path& base,
                                      const std::filesystem::path& candidate);
std::vector<std::uint8_t> readFileBytes(const std::filesystem::path& path);

} // namespace vlr
