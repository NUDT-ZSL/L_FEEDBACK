#include "renderer.hpp"

#include "shaders_embedded.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace skin {
namespace {

glm::mat4 makeViewProjection(const RenderConfig& config) {
    const float aspect =
        static_cast<float>(config.width) / static_cast<float>(config.height);
    const glm::mat4 projection = glm::perspective(
        glm::radians(config.camera.fovYDegrees),
        aspect,
        0.05F,
        100.0F
    );
    const glm::mat4 view = glm::lookAt(
        config.camera.position,
        config.camera.target,
        glm::normalize(config.camera.up)
    );
    return projection * view;
}

}  // namespace

Renderer::Renderer(
    GlContext& context,
    const RenderConfig& config,
    Logger& logger
)
    : context_(context),
      config_(config),
      logger_(logger),
      counters_(),
      framebufferCache_(counters_),
      programCache_(counters_),
      sphereMesh_(counters_),
      fullscreenVao_(counters_) {
    try {
        initializePipeline();
        initialized_ = true;
    } catch (...) {
        shutdown();
        throw;
    }
}

Renderer::~Renderer() {
    shutdown();
}

void Renderer::initializePipeline() {
    context_.makeCurrent();

    GLint maxTextureSize = 0;
    GLint maxRenderbufferSize = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);
    glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &maxRenderbufferSize);
    if (config_.width > maxTextureSize
        || config_.height > maxTextureSize
        || config_.width > maxRenderbufferSize
        || config_.height > maxRenderbufferSize
        || config_.lut.profileSamples > maxTextureSize
        || config_.lut.curvatureSamples > maxTextureSize) {
        throw std::runtime_error(
            "configured image/LUT dimensions exceed the OpenGL device limits"
        );
    }

    viewProjection_ = makeViewProjection(config_);

    lightCenter_ = config_.light.position;
    lightNormal_ = glm::normalize(
        config_.light.target - config_.light.position
    );
    const glm::vec3 worldUp(0.0F, 1.0F, 0.0F);
    glm::vec3 right = glm::cross(worldUp, lightNormal_);
    if (glm::length(right) < 0.0001F) {
        right = glm::cross(glm::vec3(1.0F, 0.0F, 0.0F), lightNormal_);
    }
    lightRight_ = glm::normalize(right);
    lightUp_ = glm::normalize(glm::cross(lightNormal_, lightRight_));
    lightRadiance_ = config_.light.color * config_.light.intensity;

    sphereMesh_.create();
    fullscreenVao_.create();
    lutProgram_ = &programCache_.get(
        "sss-lut",
        shaders::kLutVertex,
        shaders::kLutFragment
    );
    surfaceProgram_ = &programCache_.get(
        "surface",
        shaders::kSurfaceVertex,
        shaders::kSurfaceFragment
    );
    generateLut();

    logger_.info(
        "-",
        "OpenGL backend=" + context_.info().backend
            + " vendor=\"" + context_.info().vendor + "\""
            + " renderer=\"" + context_.info().renderer + "\""
            + " version=\"" + context_.info().version + "\""
    );
    logger_.info(
        "-",
        "SSS LUT generated profile_samples="
            + std::to_string(config_.lut.profileSamples)
            + " curvature_samples="
            + std::to_string(config_.lut.curvatureSamples)
            + " scattering_samples="
            + std::to_string(config_.lut.scatteringSamples)
            + " curvature_min="
            + std::to_string(config_.lut.curvatureMin)
            + " curvature_max="
            + std::to_string(config_.lut.curvatureMax)
    );
}

void Renderer::generateLut() {
    const FramebufferTarget& target = framebufferCache_.acquire(
        config_.lut.profileSamples,
        config_.lut.curvatureSamples,
        GL_RGBA16F,
        false
    );
    lutTexture_ = target.colorTexture;

    glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);
    glViewport(
        0,
        0,
        config_.lut.profileSamples,
        config_.lut.curvatureSamples
    );
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DITHER);
    glDisable(GL_FRAMEBUFFER_SRGB);
    glDisable(GL_MULTISAMPLE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glClearColor(0.0F, 0.0F, 0.0F, 0.0F);
    glClear(GL_COLOR_BUFFER_BIT);

    Program& program = *lutProgram_;
    glUseProgram(program.id());
    program.set(program.uniform("uProfileSamples"), config_.lut.profileSamples);
    program.set(
        program.uniform("uCurvatureSamples"),
        config_.lut.curvatureSamples
    );
    program.set(
        program.uniform("uScatteringSamples"),
        config_.lut.scatteringSamples
    );
    program.set(
        program.uniform("uCurvatureRange"),
        glm::vec2(config_.lut.curvatureMin, config_.lut.curvatureMax)
    );
    program.set(program.uniform("uScatterColor"), config_.lut.scatterColor);
    program.set(program.uniform("uScatterRadius"), config_.lut.scatterRadius);
    program.set(program.uniform("uWrap"), config_.lut.wrap);
    program.set(program.uniform("uProfilePower"), config_.lut.profilePower);
    fullscreenVao_.draw();
    glFinish();

    std::vector<GLenum> errors;
    for (GLenum error = glGetError(); error != GL_NO_ERROR; error = glGetError()) {
        errors.push_back(error);
    }
    if (!errors.empty()) {
        std::ostringstream message;
        message << "OpenGL error during SSS LUT generation:";
        for (GLenum error : errors) {
            message << ' ' << glErrorName(error);
        }
        throw std::runtime_error(message.str());
    }
}

void Renderer::drawSphere(
    Program& program,
    const MaterialSpec& material,
    const glm::mat4& model
) {
    program.set(program.uniform("uModel"), model);
    const glm::mat3 normalMatrix =
        glm::transpose(glm::inverse(glm::mat3(model)));
    program.set(program.uniform("uNormalMatrix"), normalMatrix);
    program.set(program.uniform("uAlbedo"), material.albedo);
    program.set(program.uniform("uRoughness"), material.roughness);
    program.set(program.uniform("uAnisotropy"), material.anisotropy);
    program.set(
        program.uniform("uAnisotropyRotation"),
        material.anisotropyRotation
    );
    program.set(program.uniform("uSssColor"), material.sssColor);
    program.set(program.uniform("uSssStrength"), material.sssStrength);
    program.set(program.uniform("uSssCurvature"), material.sssCurvature);
    program.set(program.uniform("uSpecularF0"), material.specularF0);
    sphereMesh_.draw();
}

RgbImage Renderer::renderGroup(
    const std::string& groupName,
    const MaterialSpec& primary,
    const MaterialSpec& reference
) {
    context_.makeCurrent();
    drainOpenGlErrors(groupName);

    FramebufferTarget& target = framebufferCache_.acquire(
        config_.width,
        config_.height,
        GL_RGBA8,
        true
    );
    glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);
    glViewport(0, 0, config_.width, config_.height);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DITHER);
    glDisable(GL_FRAMEBUFFER_SRGB);
    glDisable(GL_MULTISAMPLE);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_STENCIL_TEST);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glClearColor(
        config_.scene.background.r,
        config_.scene.background.g,
        config_.scene.background.b,
        1.0F
    );
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glClear(GL_DEPTH_BUFFER_BIT);

    Program& program = *surfaceProgram_;
    glUseProgram(program.id());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, lutTexture_);
    program.set(program.uniform("uSssLut"), 0);
    program.set(
        program.uniform("uLutCurvatureRange"),
        glm::vec2(config_.lut.curvatureMin, config_.lut.curvatureMax)
    );
    program.set(
        program.uniform("uCameraPosition"),
        config_.camera.position
    );
    program.set(program.uniform("uViewProjection"), viewProjection_);

    program.set(program.uniform("uLightCenter"), lightCenter_);
    program.set(program.uniform("uLightRight"), lightRight_);
    program.set(program.uniform("uLightUp"), lightUp_);
    program.set(program.uniform("uLightNormal"), lightNormal_);
    program.set(program.uniform("uLightColor"), lightRadiance_);
    program.set(
        program.uniform("uLightSize"),
        glm::vec2(config_.light.width, config_.light.height)
    );
    program.set(
        program.uniform("uLightSampleCount"),
        glm::ivec2(config_.light.samplesX, config_.light.samplesY)
    );

    program.set(
        program.uniform("uAmbientSky"),
        config_.environment.sky
    );
    program.set(
        program.uniform("uAmbientGround"),
        config_.environment.ground
    );
    program.set(
        program.uniform("uAmbientIntensity"),
        config_.environment.intensity
    );
    program.set(program.uniform("uExposure"), config_.tone.exposure);
    program.set(program.uniform("uGamma"), config_.tone.gamma);
    program.set(
        program.uniform("uToneMapMode"),
        static_cast<int>(config_.tone.mode)
    );
    program.set(
        program.uniform("uSeed"),
        static_cast<float>(config_.seed % 100000U)
    );

    const float halfSpacing = config_.scene.sphereSpacing * 0.5F;
    const glm::mat4 sphereScale = glm::scale(
        glm::mat4(1.0F),
        glm::vec3(config_.scene.sphereRadius)
    );
    drawSphere(
        program,
        primary,
        glm::translate(
            glm::mat4(1.0F),
            glm::vec3(-halfSpacing, 0.0F, 0.0F)
        ) * sphereScale
    );
    drawSphere(
        program,
        reference,
        glm::translate(
            glm::mat4(1.0F),
            glm::vec3(halfSpacing, 0.0F, 0.0F)
        ) * sphereScale
    );
    glFinish();
    checkNoOpenGlErrors(groupName, "surface draw");

    RgbImage image;
    image.width = config_.width;
    image.height = config_.height;
    image.pixels.resize(
        static_cast<std::size_t>(config_.width)
        * static_cast<std::size_t>(config_.height) * 3U
    );

    std::vector<std::uint8_t> rgba(
        static_cast<std::size_t>(config_.width)
        * static_cast<std::size_t>(config_.height) * 4U
    );
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(
        0,
        0,
        config_.width,
        config_.height,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        rgba.data()
    );
    checkNoOpenGlErrors(groupName, "framebuffer readback");

    for (int y = 0; y < config_.height; ++y) {
        const int sourceRow = config_.height - 1 - y;
        for (int x = 0; x < config_.width; ++x) {
            const std::size_t source =
                (static_cast<std::size_t>(sourceRow)
                    * static_cast<std::size_t>(config_.width)
                    + static_cast<std::size_t>(x)) * 4U;
            const std::size_t destination =
                (static_cast<std::size_t>(y)
                    * static_cast<std::size_t>(config_.width)
                    + static_cast<std::size_t>(x)) * 3U;
            image.pixels[destination] = rgba[source];
            image.pixels[destination + 1U] = rgba[source + 1U];
            image.pixels[destination + 2U] = rgba[source + 2U];
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
    glBindVertexArray(0);
    return image;
}

void Renderer::checkNoOpenGlErrors(
    const std::string& group,
    const char* stage
) {
    std::vector<GLenum> errors;
    for (GLenum error = glGetError(); error != GL_NO_ERROR; error = glGetError()) {
        errors.push_back(error);
    }
    if (errors.empty()) {
        return;
    }

    std::ostringstream message;
    message << "OpenGL error after " << stage
            << " for group \"" << group << "\":";
    for (GLenum error : errors) {
        message << ' ' << glErrorName(error);
    }
    throw GroupFailure(kErrorOpenGl, message.str());
}

void Renderer::drainOpenGlErrors(const std::string& group) {
    bool drainedAny = false;
    for (GLenum error = glGetError(); error != GL_NO_ERROR; error = glGetError()) {
        drainedAny = true;
        logger_.warning(
            group,
            kErrorOpenGl,
            "drained stale OpenGL error before group render: "
                + glErrorName(error)
        );
    }
    if (drainedAny) {
        glFinish();
    }
}

void Renderer::shutdown() {
    if (shutDown_) {
        return;
    }
    shutDown_ = true;

    context_.makeCurrent();
    fullscreenVao_.release();
    sphereMesh_.release();
    programCache_.releaseAll();
    framebufferCache_.releaseAll();
    lutTexture_ = 0;
    lutProgram_ = nullptr;
    surfaceProgram_ = nullptr;
    counters_.logSummary(logger_, "-");
}

}  // namespace skin

