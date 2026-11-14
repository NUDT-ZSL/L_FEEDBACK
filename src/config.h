#pragma once

#include <string>

namespace df {

struct VolumetricConfig {
    int steps = 32;
    float densityScale = 0.12f;
    float maxDistance = 34.0f;
    float noiseScale = 0.19f;
    float absorption = 1.15f;
    float scattering = 0.85f;
    bool enabled = true;
};

struct TransparentConfig {
    bool sortingEnabled = true;
    int exactSortLimit = 1024;
    int bucketCount = 2048;
};

struct LightConfig {
    int maxLights = 8;
};

struct RenderConfig {
    int width = 800;
    int height = 450;
    int offscreenSamples = 1;
    int maxFrames = 0;
    VolumetricConfig volumetric;
    TransparentConfig transparent;
    LightConfig lights;
};

RenderConfig loadConfig(const std::string& path);
int effectiveFogSteps(const RenderConfig& config, bool offline);

} // namespace df

