#pragma once

#include "density.h"
#include "math.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cloud {

constexpr float kSkyDepth = 1.0e9f;

struct Light {
    Vec3 direction{0.45f, 0.82f, 0.25f};
    Vec3 color{1.0f, 0.95f, 0.86f};
    float ambient = 0.16f;
};

Light defaultLight();

struct RenderSettings {
    int width = 1280;
    int height = 720;
    int renderScale = 2;
    int maxSteps = 80;
    int lightSteps = 6;
    float densityExtinction = 8.0f;
    float lightExtinction = 10.0f;
    float exposure = 1.15f;
    int maxHistoryLength = 8;
};

struct SampleFrame {
    int width = 0;
    int height = 0;
    int renderScale = 1;
    std::vector<Vec3> color;
    std::vector<float> depth;
};

struct RenderState {
    bool hasHistory = false;
    std::uint32_t frameIndex = 0;
    Mat4 previousView = Mat4::identity();
    Mat4 previousProjection = Mat4::identity();
    Vec3 previousCameraPosition{};
    std::vector<Vec3> accumulatedColor;
    std::vector<float> depth;
    std::vector<std::uint32_t> counts;

    void configure(int width, int height);
};

struct FrameStats {
    std::uint32_t frameIndex = 0;
    std::uint64_t reprojectionAttempts = 0;
    std::uint64_t reprojectionHits = 0;
    std::uint64_t reprojectionMisses = 0;
    double reprojectionHitRate = 0.0;
    std::uint32_t minimumCount = 0;
    std::uint32_t maximumCount = 0;
    bool countsStrictlyMonotonic = true;
};

class Renderer {
public:
    Renderer(const RenderSettings& settings, const DensityField& density);

    SampleFrame renderSample(const Camera& camera, const Light& light, int scaleOverride = 0) const;
    FrameStats accumulate(RenderState& state, const SampleFrame& sample, const Camera& camera,
                          std::uint32_t frameIndex);

    std::vector<std::uint8_t> encodeRGBA(const std::vector<Vec3>& hdrColor) const;
    static Vec3 toneMap(const Vec3& color, float exposure);
    static double psnrRGB8(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b);

private:
    struct ScatterResult {
        Vec3 color;
        float depth = kSkyDepth;
    };

    ScatterResult traceRay(const Ray& ray, const Light& light) const;
    Vec3 sampleSky(const Ray& ray, const Light& light) const;
    SampleFrame renderLowResolution(const Camera& camera, const Light& light, int scale) const;
    SampleFrame upsampleDepthAware(const SampleFrame& low, int fullWidth, int fullHeight,
                                   int scale) const;

    RenderSettings settings_;
    const DensityField& density_;
};

}  // namespace cloud

