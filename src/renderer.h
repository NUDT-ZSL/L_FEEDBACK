#pragma once

#include "assets.h"
#include "image_io.h"
#include "raster.h"
#include "visibility.h"

#include <array>
#include <cstdint>
#include <vector>

namespace cv {

struct RendererConfig {
    int width = 960;
    int height = 640;
    int cascadeResolution = 1024;
    int pointShadowResolution = 256;
    float cascadeSplit = 7.0f;
    int pcfRadius = 1;
    bool shadows = true;
};

struct FrameStats {
    uint32_t totalTriangles = 0;
    uint32_t gridCandidateTriangles = 0;
    uint32_t gridVisibleTriangles = 0;
    uint32_t submittedTriangles = 0;
    uint32_t opaqueTriangles = 0;
    uint32_t transparentTriangles = 0;
    uint32_t visibleCells = 0;
    uint32_t totalCells = 0;
    uint32_t opaqueCommands = 0;
    uint32_t transparentCommands = 0;
    uint32_t wireCommands = 0;
    uint32_t stateSwitches = 0;
    uint32_t shadowMaps = 0;
    uint32_t shadowTriangles = 0;
    bool transparentOrderValid = true;
    float cullRatio = 1.0f;
    double cullMs = 0.0;
    double shadowMs = 0.0;
    double colorMs = 0.0;
    double totalMs = 0.0;
};

class Renderer {
public:
    Renderer(const Scene& scene, const Lighting& lighting, const RendererConfig& config);

    void resize(int width, int height);
    void render(const Scene& scene, const Lighting& lighting, const Camera& camera,
                uint32_t tick, bool wireframe);

    const Image& image() const { return image_; }
    const FrameStats& stats() const { return stats_; }
    int width() const { return config_.width; }
    int height() const { return config_.height; }
    const RendererConfig& config() const { return config_; }

private:
    struct Command {
        enum class Kind {
            Opaque,
            Transparent,
            Wire,
        } kind = Kind::Opaque;
        uint16_t material = 0;
        std::vector<FaceRef> faces;
    };

    struct PointShadow {
        std::array<DepthRasterizer, 6> faces;
        std::array<Mat4, 6> viewProjection{};
    };

    void configureShadowMatrices(const Lighting& lighting);
    void renderShadows(const Scene& scene,
                       const std::vector<std::vector<FrameVertex>>& frameVertices,
                       const Lighting& lighting);
    void buildCommands(const Scene& scene,
                       const std::vector<std::vector<FrameVertex>>& frameVertices,
                       const Camera& camera, const std::vector<FaceRef>& visibleFaces,
                       bool wireframe);
    void executeCommands(const Scene& scene,
                         const std::vector<std::vector<FrameVertex>>& frameVertices,
                         const Camera& camera, const Lighting& lighting);
    ShadeResult shadeFragment(const FragmentInput& fragment, const Material& material,
                              const Camera& camera, const Lighting& lighting) const;
    Vec3 sampleNormalMap(Vec2 uv) const;
    float directionalShadow(Vec3 position, Vec3 normal, Vec3 lightDirection,
                            float ndotl, const Camera& camera) const;
    float pointShadow(size_t lightIndex, const Light& light, Vec3 position, Vec3 normal,
                      float ndotl) const;
    float sampleShadow(const DepthRasterizer& map, Vec2 uv, float depth, float bias) const;
    void resolveImage();

    RendererConfig config_;
    Rasterizer raster_;
    UniformGrid grid_;
    NormalTexture normalMap_;
    std::array<DepthRasterizer, 2> cascades_;
    std::array<Mat4, 2> cascadeViewProjection_{};
    std::array<PointShadow, 2> pointShadowViews_{};
    std::vector<Command> commands_;
    Image image_;
    FrameStats stats_;
    int64_t shadowCacheTick_ = -1;
};

}  // namespace cv

