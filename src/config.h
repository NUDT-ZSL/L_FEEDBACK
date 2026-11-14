#pragma once

#include "density.h"
#include "math.h"
#include "renderer.h"

#include <cstdint>
#include <map>
#include <string>

namespace cloud {

struct ExplicitCamera {
    bool usesMatrices = false;
    Mat4 view = Mat4::identity();
    Mat4 projection = Mat4::identity();
    Vec3 position{};
    Vec3 target{};
    Vec3 up{0.0f, 1.0f, 0.0f};
    float fovDegrees = 50.0f;
};

struct CameraConfig {
    Vec3 position{1.75f, 1.15f, 1.75f};
    Vec3 target{0.5f, 0.45f, 0.5f};
    Vec3 up{0.0f, 1.0f, 0.0f};
    float fovDegrees = 50.0f;
    float nearPlane = 0.05f;
    float farPlane = 100.0f;
    bool orbitEnabled = true;
    Vec3 orbitCenter{0.5f, 0.47f, 0.5f};
    float orbitRadius = 2.35f;
    float orbitHeight = 0.95f;
    float orbitStartDegrees = 35.0f;
    float orbitDegreesPerFrame = 1.1f;
    std::map<std::uint32_t, ExplicitCamera> frameOverrides;

    Camera cameraAt(std::uint32_t frameIndex, int width, int height) const;
};

struct SunConfig {
    Vec3 direction{0.45f, 0.82f, 0.25f};
    Vec3 color{1.0f, 0.95f, 0.86f};
    float ambient = 0.16f;
    float degreesPerFrame = 0.0f;

    Light lightAt(std::uint32_t frameIndex) const;
};

struct SnapshotConfig {
    std::uint32_t version = 2;
    std::string loadPath;
    std::string savePath;
    std::uint32_t saveEveryFrames = 1;
    bool stopAfterSave = false;
};

struct MemoryConfig {
    double availableMB = 512.0;
};

struct QualityConfig {
    bool compareAgainstFullResolution = false;
    bool writeFullResolutionReference = false;
};

struct AppConfig {
    std::string configPath;
    std::string outputDirectory = "artifacts/render";
    std::string outputName = "frame";
    int width = 1280;
    int height = 720;
    std::uint32_t startFrame = 0;
    std::uint32_t endFrame = 1;
    std::uint64_t seed = 1337;
    CloudType cloudType = CloudType::Cumulus;
    int renderScale = 2;
    int maxSteps = 80;
    int lightSteps = 6;
    float densityExtinction = 8.0f;
    float lightExtinction = 10.0f;
    float exposure = 1.15f;
    int maxHistoryLength = 8;
    bool asynchronous = true;
    std::uint32_t workerThreads = 4;
    CameraConfig camera;
    SunConfig sun;
    SnapshotConfig snapshot;
    MemoryConfig memory;
    QualityConfig quality;

    RenderSettings renderSettings() const;
};

AppConfig loadConfig(const std::string& path);

}  // namespace cloud

