#pragma once

#include "common.hpp"
#include "config.hpp"
#include "gl_backend.hpp"
#include "gl_resources.hpp"
#include "logger.hpp"

#include <glm/glm.hpp>

#include <string>

namespace skin {

class Renderer {
public:
    Renderer(
        GlContext& context,
        const RenderConfig& config,
        Logger& logger
    );
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    RgbImage renderGroup(
        const std::string& groupName,
        const MaterialSpec& primary,
        const MaterialSpec& reference
    );

    void shutdown();

private:
    void initializePipeline();
    void generateLut();
    void drawSphere(
        Program& program,
        const MaterialSpec& material,
        const glm::mat4& model
    );
    void checkNoOpenGlErrors(const std::string& group, const char* stage);
    void drainOpenGlErrors(const std::string& group);

    GlContext& context_;
    const RenderConfig& config_;
    Logger& logger_;

    ResourceCounters counters_;
    FramebufferCache framebufferCache_;
    ProgramCache programCache_;
    SphereMesh sphereMesh_;
    FullscreenVao fullscreenVao_;

    Program* lutProgram_ = nullptr;
    Program* surfaceProgram_ = nullptr;
    GLuint lutTexture_ = 0;

    glm::mat4 viewProjection_{1.0F};
    glm::vec3 lightCenter_{0.0F};
    glm::vec3 lightRight_{1.0F, 0.0F, 0.0F};
    glm::vec3 lightUp_{0.0F, 1.0F, 0.0F};
    glm::vec3 lightNormal_{0.0F, 0.0F, -1.0F};
    glm::vec3 lightRadiance_{1.0F};

    bool initialized_ = false;
    bool shutDown_ = false;
};

}  // namespace skin

