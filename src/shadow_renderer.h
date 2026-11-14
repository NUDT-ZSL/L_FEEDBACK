#pragma once

#include <array>
#include <cstddef>

#include <glm/glm.hpp>

#include "camera.h"
#include "config.h"
#include "gl_utils.h"
#include "scene.h"

struct CascadeData {
    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f};
    glm::mat4 viewProjection{1.0f};
    float splitNear = 0.0f;
    float splitFar = 0.0f;
    float radius = 1.0f;
    float farPlane = 1.0f;
    float texelWorld = 0.01f;
};

class ShadowRenderer {
public:
    ShadowRenderer() = default;
    ~ShadowRenderer();

    ShadowRenderer(const ShadowRenderer&) = delete;
    ShadowRenderer& operator=(const ShadowRenderer&) = delete;

    void initialize(int resolution);
    void shutdown();
    void resize(int resolution);

    void updateCascades(const OrbitCamera& camera,
                        const DirectionalLight& light,
                        float splitLambda,
                        float overlapFraction);
    void renderShadowMaps(const Scene& scene,
                          const LightManager& lights,
                          const OrbitCamera& camera);

    int resolution() const { return resolution_; }
    int cascadeCount() const { return cascadeCount_; }
    const std::array<CascadeData, kMaxCascades>& cascades() const { return cascades_; }
    const std::array<float, kMaxCascades>& splitDepths() const { return splitDepths_; }
    const std::array<float, kMaxPointLights>& pointFarPlanes() const { return pointFarPlanes_; }
    GLuint directionalTexture() const { return directionalTexture_; }
    GLuint pointTexture(int index) const;
    std::size_t estimatedBytes() const { return estimatedBytes_; }

private:
    void createResources(int resolution);
    void destroyResources();
    void renderDirectional(const Scene& scene, const LightManager& lights);
    void renderPointLight(const Scene& scene, const PointLight& light, int index);
    void setDepthPassUniforms();

    ShaderProgram depthShader_;
    int resolution_ = 1024;
    int cascadeCount_ = kMaxCascades;
    std::size_t estimatedBytes_ = 0;

    GLuint directionalFbo_ = 0;
    GLuint directionalTexture_ = 0;
    std::array<GLuint, kMaxPointLights> pointFbos_{};
    std::array<GLuint, kMaxPointLights> pointTextures_{};

    std::array<CascadeData, kMaxCascades> cascades_{};
    std::array<float, kMaxCascades> splitDepths_{};
    std::array<float, kMaxPointLights> pointFarPlanes_{};
};

