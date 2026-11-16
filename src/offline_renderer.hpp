#pragma once

#include "error.hpp"
#include "scene.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

#include <glm/glm.hpp>

namespace skel {

struct Camera {
    glm::vec3 eye{4.5F, 3.4F, 6.5F};
    glm::vec3 target{0.0F, 1.3F, 0.0F};
    glm::vec3 up{0.0F, 1.0F, 0.0F};
    float fieldOfViewYDegrees = 45.0F;
    float nearPlane = 0.05F;
    float farPlane = 100.0F;
};

struct RenderSettings {
    std::uint32_t width = 640;
    std::uint32_t height = 480;
    Camera camera;
};

struct OfflineImage {
    std::vector<float> depth;
    std::vector<std::uint8_t> rgb;
};

bool cameraMatrices(const Camera& camera, std::uint32_t width,
                    std::uint32_t height, glm::mat4& view,
                    glm::mat4& projection, Error& error);

// Deterministic CPU rasterizer. This is separate from platform/GPU behavior
// and feeds only the already-baked/shared vertex stream.
bool rasterizeOffline(const BakedFrame& baked, const RenderSettings& settings,
                      OfflineImage& image, Error& error);

bool writeOfflineImages(const OfflineImage& image,
                        const RenderSettings& settings,
                        const std::filesystem::path& depthPath,
                        const std::filesystem::path& colorPath,
                        Error& error);

} // namespace skel

