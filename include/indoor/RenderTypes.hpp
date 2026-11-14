#pragma once

#include "indoor/Scene.hpp"

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace indoor {

// Four G-Buffer reads plus eleven cube shadow maps stay within the OpenGL 4.5
// guaranteed minimum of sixteen fragment texture units.
constexpr int kMaxLightsPerBatch = 11;
constexpr int kMaterialTextureLayers = 8;
constexpr int kMaterialTextureSize = 256;

struct LightSnapshot {
    Light light{};
    std::uint64_t shadowVersion = 0;
    bool cacheResident = false;
};

struct FrameSnapshot {
    std::uint64_t frameId = 0;
    std::uint64_t sceneGeneration = 0;
    MaterialPresetId materialPreset = MaterialPresetId::FullLayered;
    Camera camera{};
    std::vector<LightSnapshot> lights;
    // Oldest to newest. Used to restore exactly the same resident shadow maps
    // after a context loss, including when cacheCapacity < lightCount.
    std::vector<std::uint32_t> cacheLruOrder;
};

struct ShadowFrameStats {
    std::uint32_t totalLights = 0;
    std::uint32_t residentShadowMaps = 0;
    std::uint32_t validShadowMaps = 0;
    std::uint32_t cacheCapacity = 0;
    std::uint64_t versionMigrations = 0;
    std::uint64_t versionMigrationsThisFrame = 0;
    std::uint32_t shadowMapsRenderedThisFrame = 0;
    std::uint32_t shadowFacesRenderedThisFrame = 0;
    std::uint32_t cacheEvictionsThisFrame = 0;
    std::uint64_t cacheBytes = 0;
    std::vector<std::uint32_t> dirtyLightIds;
};

enum class DebugView : int {
    Final = 0,
    WorldPosition = 1,
    ViewNormal = 2,
    Albedo = 3,
    RoughMetal = 4,
    ShadowVersions = 5,
    CacheOccupancy = 6,
    Count = 7
};

const char* debugViewName(DebugView view);

struct RenderSettings {
    int width = 1280;
    int height = 720;
    int shadowSize = 96;
    std::size_t shadowCacheCapacity = 320;
};

struct MaterialTextureData {
    int size = kMaterialTextureSize;
    int layers = kMaterialTextureLayers;
    std::vector<std::uint8_t> rgba;
};

MaterialTextureData generateMaterialTextures(std::uint64_t seed);

std::uint64_t pixelDigest(const std::vector<std::uint8_t>& pixels);
int countPixelDifferences(const std::vector<std::uint8_t>& a,
                          const std::vector<std::uint8_t>& b);

} // namespace indoor
