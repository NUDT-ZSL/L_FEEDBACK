#include "renderer.hpp"

#include "error.hpp"
#include "post_process.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <limits>
#include <memory>
#include <sstream>

namespace scene {
namespace {

using Clock = std::chrono::steady_clock;

double elapsedMilliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string joinShaderPath(const std::string& directory, const std::string& file) {
    if (directory.empty()) {
        return file;
    }
    const char last = directory.back();
    return (last == '/' || last == '\\') ? directory + file
                                         : directory + "/" + file;
}

struct PipelineResources {
    bool hasGbuffer = false;
    bool hasShadow = false;
    bool hasDeferred = false;
    bool hasPostStage = false;
    bool hasEnabledPostEffects = false;
    int shadowResolution = 1024;

    GlFramebuffer gbufferFbo;
    GlTexture gPosition;
    GlTexture gNormal;
    GlTexture gMaterial;
    GlTexture gDepth;

    GlFramebuffer sceneFbo;
    GlTexture sceneColor;

    GlFramebuffer shadowFbo;
    GlTexture shadowDepth;
    GlTexture fallbackShadowDepth;

    GlFramebuffer finalFbo;
    GlTexture finalColor;

    GlFramebuffer postFboA;
    GlFramebuffer postFboB;
    GlTexture postTextureA;
    GlTexture postTextureB;

    ShaderProgram geometryProgram;
    ShaderProgram shadowProgram;
    ShaderProgram deferredProgram;
    ShaderProgram previewProgram;
    ShaderProgram presentProgram;

    std::vector<std::unique_ptr<PostEffect>> effects;
};

void configureTexture(GlTexture& texture, const std::string& label, int width,
                      int height, GLenum internalFormat, GLenum format,
                      GLenum type, GLenum filter) {
    texture.setLabel(label);
    texture.create2D(width, height, internalFormat, format, type, filter,
                     filter, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, nullptr);
}

} // namespace

struct Renderer::Impl {
    const Scene& scene;
    int width;
    int height;
    std::string shaderDirectory;
    PipelineConfig config;
    bool pipelineReady = false;
    GLuint fullscreenVao = 0;
    std::unique_ptr<PipelineResources> resources;

    Impl(const Scene& sceneValue, int widthValue, int heightValue,
         const std::string& shaderDirectoryValue)
        : scene(sceneValue), width(widthValue), height(heightValue),
          shaderDirectory(shaderDirectoryValue) {
        if (width < 16 || height < 16 || width > 8192 || height > 8192) {
            throw AppError(ErrorType::Cli,
                           "render resolution must be between 16 and 8192");
        }

        if (shaderDirectory.empty()) {
            shaderDirectory = SCENE_DEFAULT_SHADER_DIR;
        }
        const std::filesystem::path shaderDir(shaderDirectory);
        if (!std::filesystem::exists(shaderDir / "fullscreen.vert")) {
            throw AppError(ErrorType::FileIo,
                           "shader directory '" + shaderDirectory +
                           "' does not contain fullscreen.vert");
        }

        glGenVertexArrays(1, &fullscreenVao);
        if (fullscreenVao == 0) {
            throw AppError(ErrorType::GlRuntime,
                           "failed to create fullscreen VAO");
        }
        checkGlError("creating renderer state");
    }

    ~Impl() {
        resources.reset();
        if (fullscreenVao != 0) {
            glDeleteVertexArrays(1, &fullscreenVao);
            fullscreenVao = 0;
        }
    }

    std::string shaderFile(const std::string& name) const {
        return joinShaderPath(shaderDirectory, name);
    }

    std::unique_ptr<PipelineResources> buildResources(
        const PipelineConfig& nextConfig) const {
        auto built = std::make_unique<PipelineResources>();
        const bool geometryEnabled = nextConfig.hasEnabledStage(StageType::Geometry);
        const bool shadowEnabled = nextConfig.hasEnabledStage(StageType::Shadow);
        const bool deferredEnabled = nextConfig.hasEnabledStage(StageType::Deferred);
        const bool postEnabled = nextConfig.hasEnabledStage(StageType::PostProcess);

        built->hasShadow = shadowEnabled;
        built->hasDeferred = deferredEnabled;
        built->hasPostStage = postEnabled;
        built->shadowResolution = nextConfig.shadow.resolution;

        built->sceneColor.setLabel("scene-color");
        built->sceneColor.create2D(width, height, GL_RGBA16F, GL_RGBA, GL_FLOAT,
                                   GL_LINEAR, GL_LINEAR, GL_CLAMP_TO_EDGE,
                                   GL_CLAMP_TO_EDGE, nullptr);
        built->sceneFbo.setLabel("scene-color");
        built->sceneFbo.create();
        built->sceneFbo.attachColor(0, built->sceneColor);
        built->sceneFbo.checkComplete("scene-color");

        if (geometryEnabled || deferredEnabled) {
            built->hasGbuffer = true;
            configureTexture(built->gPosition, "g-buffer-position", width, height,
                             GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_NEAREST);
            configureTexture(built->gNormal, "g-buffer-normal", width, height,
                             GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_NEAREST);
            built->gMaterial.setLabel("g-buffer-material");
            built->gMaterial.create2D(width, height, GL_RGBA8, GL_RGBA,
                                      GL_UNSIGNED_BYTE, GL_NEAREST, GL_NEAREST,
                                      GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, nullptr);
            built->gDepth.setLabel("g-buffer-depth");
            built->gDepth.create2D(width, height, GL_DEPTH_COMPONENT24,
                                   GL_DEPTH_COMPONENT, GL_UNSIGNED_INT,
                                   GL_NEAREST, GL_NEAREST, GL_CLAMP_TO_EDGE,
                                   GL_CLAMP_TO_EDGE, nullptr);

            built->gbufferFbo.setLabel("g-buffer");
            built->gbufferFbo.create();
            built->gbufferFbo.attachColor(0, built->gPosition);
            built->gbufferFbo.attachColor(1, built->gNormal);
            built->gbufferFbo.attachColor(2, built->gMaterial);
            built->gbufferFbo.attachDepth(built->gDepth);
            glBindFramebuffer(GL_FRAMEBUFFER, built->gbufferFbo.id());
            const GLenum buffers[3] = {
                GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2
            };
            glDrawBuffers(3, buffers);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            built->gbufferFbo.checkComplete("g-buffer");
        }

        if (shadowEnabled) {
            built->shadowDepth.setLabel("shadow-depth");
            built->shadowDepth.create2D(
                nextConfig.shadow.resolution, nextConfig.shadow.resolution,
                GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT,
                GL_NEAREST, GL_NEAREST, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE,
                nullptr);
            built->shadowFbo.setLabel("shadow");
            built->shadowFbo.create();
            built->shadowFbo.attachDepth(built->shadowDepth);
            glBindFramebuffer(GL_FRAMEBUFFER, built->shadowFbo.id());
            glDrawBuffer(GL_NONE);
            glReadBuffer(GL_NONE);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            built->shadowFbo.checkComplete("shadow");
        }

        if (deferredEnabled) {
            const float clearDepth = 1.0f;
            built->fallbackShadowDepth.setLabel("fallback-shadow-depth");
            built->fallbackShadowDepth.create2D(
                1, 1, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT,
                GL_NEAREST, GL_NEAREST, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE,
                &clearDepth);
        }

        built->finalColor.setLabel("final-color");
        built->finalColor.create2D(width, height, GL_RGBA8, GL_RGBA,
                                   GL_UNSIGNED_BYTE, GL_NEAREST, GL_NEAREST,
                                   GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, nullptr);
        built->finalFbo.setLabel("final");
        built->finalFbo.create();
        built->finalFbo.attachColor(0, built->finalColor);
        built->finalFbo.checkComplete("final");

        for (const auto& effectConfig : nextConfig.postEffects) {
            if (effectConfig.enabled) {
                built->hasEnabledPostEffects = true;
                break;
            }
        }

        if (postEnabled && built->hasEnabledPostEffects) {
            configureTexture(built->postTextureA, "post-ping", width, height,
                             GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_LINEAR);
            configureTexture(built->postTextureB, "post-pong", width, height,
                             GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_LINEAR);
            built->postFboA.setLabel("post-ping");
            built->postFboA.create();
            built->postFboA.attachColor(0, built->postTextureA);
            built->postFboA.checkComplete("post-ping");
            built->postFboB.setLabel("post-pong");
            built->postFboB.create();
            built->postFboB.attachColor(0, built->postTextureB);
            built->postFboB.checkComplete("post-pong");
        }

        if (geometryEnabled) {
            built->geometryProgram = ShaderProgram::fromFiles(
                shaderFile("geometry.vert"), shaderFile("geometry.frag"));
            built->previewProgram = ShaderProgram::fromFiles(
                shaderFile("fullscreen.vert"), shaderFile("preview.frag"));
        }
        if (shadowEnabled) {
            built->shadowProgram = ShaderProgram::fromFiles(
                shaderFile("shadow.vert"), shaderFile("shadow.frag"));
        }
        if (deferredEnabled) {
            built->deferredProgram = ShaderProgram::fromFiles(
                shaderFile("fullscreen.vert"), shaderFile("deferred.frag"));
        }
        built->presentProgram = ShaderProgram::fromFiles(
            shaderFile("fullscreen.vert"), shaderFile("present.frag"));

        if (postEnabled) {
            for (const auto& effectConfig : nextConfig.postEffects) {
                if (!effectConfig.enabled) {
                    continue;
                }
                built->effects.push_back(createPostEffect(
                    effectConfig, shaderDirectory, width, height, fullscreenVao));
            }
        }

        checkGlError("building pipeline resources");
        return built;
    }

    void prepareFrame(const PipelineResources& target) {
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glDisable(GL_FRAMEBUFFER_SRGB);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask(GL_TRUE);
        glClearDepth(1.0);

        glBindFramebuffer(GL_FRAMEBUFFER, target.sceneFbo.id());
        glViewport(0, 0, width, height);
        glClearColor(config.clearColor.x, config.clearColor.y,
                     config.clearColor.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        if (target.hasGbuffer) {
            glBindFramebuffer(GL_FRAMEBUFFER, target.gbufferFbo.id());
            const GLenum buffers[3] = {
                GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2
            };
            glDrawBuffers(3, buffers);
            glViewport(0, 0, width, height);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }

        if (target.hasShadow) {
            glBindFramebuffer(GL_FRAMEBUFFER, target.shadowFbo.id());
            glViewport(0, 0, config.shadow.resolution, config.shadow.resolution);
            glClear(GL_DEPTH_BUFFER_BIT);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        checkGlError("preparing frame");
    }

    void executeGeometry(PipelineResources& target) {
        glBindFramebuffer(GL_FRAMEBUFFER, target.gbufferFbo.id());
        glViewport(0, 0, width, height);
        const GLenum buffers[3] = {
            GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2
        };
        glDrawBuffers(3, buffers);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);

        target.geometryProgram.use();
        target.geometryProgram.setMat4("uView", scene.view());
        target.geometryProgram.setMat4(
            "uProjection",
            scene.projectionForAspect(static_cast<float>(width) /
                                      static_cast<float>(height)));

        for (const SceneObject& object : scene.objects()) {
            target.geometryProgram.setMat4("uModel", object.model);
            target.geometryProgram.setMat3("uNormalMatrix", object.normalMatrix);
            target.geometryProgram.setVec3("uAlbedo", object.material.albedo);
            target.geometryProgram.setFloat("uRoughness", object.material.roughness);
            target.geometryProgram.setFloat("uMetallic", object.material.metallic);
            scene.meshes()[static_cast<size_t>(object.mesh)].draw();
        }

        glBindFramebuffer(GL_FRAMEBUFFER, target.sceneFbo.id());
        glViewport(0, 0, width, height);
        glDisable(GL_DEPTH_TEST);
        target.previewProgram.use();
        bindTexture(0, target.gPosition.id());
        bindTexture(1, target.gNormal.id());
        bindTexture(2, target.gMaterial.id());
        target.previewProgram.setInt("uPosition", 0);
        target.previewProgram.setInt("uNormal", 1);
        target.previewProgram.setInt("uMaterial", 2);
        target.previewProgram.setVec2("uResolution",
                                      static_cast<float>(width),
                                      static_cast<float>(height));
        target.previewProgram.setVec3("uClearColor", config.clearColor);
        drawFullscreenTriangle(fullscreenVao);
        checkGlError("rendering geometry stage");
    }

    void executeShadow(PipelineResources& target) {
        glBindFramebuffer(GL_FRAMEBUFFER, target.shadowFbo.id());
        glViewport(0, 0, config.shadow.resolution, config.shadow.resolution);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(2.0f, 4.0f);
        target.shadowProgram.use();
        target.shadowProgram.setMat4("uLightViewProjection",
                                     scene.lightViewProjection());
        target.shadowProgram.setMat4("uModel", Mat4::identity());
        for (const SceneObject& object : scene.objects()) {
            target.shadowProgram.setMat4("uModel", object.model);
            scene.meshes()[static_cast<size_t>(object.mesh)].draw();
        }
        glDisable(GL_POLYGON_OFFSET_FILL);
        checkGlError("rendering shadow stage");
    }

    void executeDeferred(PipelineResources& target, bool shadowProduced) {
        glBindFramebuffer(GL_FRAMEBUFFER, target.sceneFbo.id());
        glViewport(0, 0, width, height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);

        target.deferredProgram.use();
        bindTexture(0, target.gPosition.id());
        bindTexture(1, target.gNormal.id());
        bindTexture(2, target.gMaterial.id());
        bindTexture(3, shadowProduced ? target.shadowDepth.id()
                                      : target.fallbackShadowDepth.id());
        target.deferredProgram.setInt("uPosition", 0);
        target.deferredProgram.setInt("uNormal", 1);
        target.deferredProgram.setInt("uMaterial", 2);
        target.deferredProgram.setInt("uShadowMap", 3);
        target.deferredProgram.setVec2("uResolution",
                                       static_cast<float>(width),
                                       static_cast<float>(height));
        target.deferredProgram.setVec3("uClearColor", config.clearColor);
        target.deferredProgram.setVec3("uAmbient", config.deferred.ambient);
        target.deferredProgram.setMat4("uInvView", inverse(scene.view()));
        target.deferredProgram.setMat4("uLightViewProjection",
                                       scene.lightViewProjection());
        target.deferredProgram.setInt("uShadowEnabled", shadowProduced ? 1 : 0);
        target.deferredProgram.setFloat("uShadowBias", config.shadow.bias);
        target.deferredProgram.setFloat("uShadowStrength", config.shadow.strength);
        target.deferredProgram.setVec2(
            "uShadowTexel",
            config.shadow.resolution > 0
                ? 1.0f / static_cast<float>(config.shadow.resolution) : 0.0f,
            config.shadow.resolution > 0
                ? 1.0f / static_cast<float>(config.shadow.resolution) : 0.0f);

        constexpr size_t maxLights = 16;
        const size_t lightCount = std::min(scene.lights().size(), maxLights);
        std::vector<int> types;
        std::vector<Vec3> positions;
        std::vector<Vec3> directions;
        std::vector<Vec3> colors;
        std::vector<float> intensities;
        std::vector<float> cones;
        types.reserve(lightCount);
        positions.reserve(lightCount);
        directions.reserve(lightCount);
        colors.reserve(lightCount);
        intensities.reserve(lightCount);
        cones.reserve(lightCount * 2);

        for (size_t i = 0; i < lightCount; ++i) {
            const Light& light = scene.lights()[i];
            types.push_back(static_cast<int>(light.type));
            positions.push_back(transformPoint(scene.view(), light.position));
            directions.push_back(normalize(
                transformDirection(scene.view(), light.direction)));
            colors.push_back(light.color);
            intensities.push_back(light.intensity);
            cones.push_back(light.innerConeCosine);
            cones.push_back(light.outerConeCosine);
        }

        target.deferredProgram.setInt("uLightCount",
                                      static_cast<int>(lightCount));
        target.deferredProgram.setIntArray("uLightType", types);
        target.deferredProgram.setVec3Array("uLightPosition", positions);
        target.deferredProgram.setVec3Array("uLightDirection", directions);
        target.deferredProgram.setVec3Array("uLightColor", colors);
        target.deferredProgram.setFloatArray("uLightIntensity", intensities);
        target.deferredProgram.setVec2Array("uLightCone", cones);
        drawFullscreenTriangle(fullscreenVao);
        checkGlError("rendering deferred shading stage");
    }

    void blitColor(GLuint sourceFramebuffer, GLuint targetFramebuffer) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, sourceFramebuffer);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, targetFramebuffer);
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    }

    void executePost(PipelineResources& target,
                     std::vector<TimingEntry>& effectTimings) {
        if (!target.hasEnabledPostEffects || target.effects.empty()) {
            return;
        }

        blitColor(target.sceneFbo.id(), target.postFboA.id());
        GLuint currentFramebuffer = target.postFboA.id();
        GLuint currentTexture = target.postTextureA.id();
        GLuint nextFramebuffer = target.postFboB.id();
        GLuint nextTexture = target.postTextureB.id();

        const PostPassContext passContext{width, height, fullscreenVao};
        for (size_t i = 0; i < target.effects.size(); ++i) {
            const auto start = Clock::now();
            target.effects[i]->apply(passContext, currentTexture, nextFramebuffer);
            glFinish();
            const auto end = Clock::now();
            effectTimings.push_back(TimingEntry{
                target.effects[i]->type(), true,
                elapsedMilliseconds(start, end)
            });

            std::swap(currentFramebuffer, nextFramebuffer);
            std::swap(currentTexture, nextTexture);
        }

        blitColor(currentFramebuffer, target.sceneFbo.id());
        (void)currentTexture;
        checkGlError("rendering postprocess stage");
    }

    double present(PipelineResources& target, bool capturePixels,
                   std::vector<uint8_t>& pixels, double& readbackMilliseconds) {
        const auto start = Clock::now();
        glBindFramebuffer(GL_FRAMEBUFFER, target.finalFbo.id());
        glViewport(0, 0, width, height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        target.presentProgram.use();
        bindTexture(0, target.sceneColor.id());
        target.presentProgram.setInt("uInput", 0);
        drawFullscreenTriangle(fullscreenVao);
        glFinish();
        const auto afterDraw = Clock::now();
        const double drawMilliseconds = elapsedMilliseconds(start, afterDraw);

        if (capturePixels) {
            const auto readStart = Clock::now();
            pixels.resize(static_cast<size_t>(width) * height * 4);
            glBindFramebuffer(GL_FRAMEBUFFER, target.finalFbo.id());
            glReadBuffer(GL_COLOR_ATTACHMENT0);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE,
                         pixels.data());
            glFinish();
            readbackMilliseconds = elapsedMilliseconds(readStart, Clock::now());
            checkGlError("reading final image");
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        checkGlError("presenting final image");
        return drawMilliseconds;
    }
};

Renderer::Renderer(const Scene& scene, int width, int height,
                   const std::string& shaderDirectory)
    : impl_(std::make_unique<Impl>(scene, width, height, shaderDirectory)) {
}

Renderer::~Renderer() = default;

void Renderer::applyPipeline(const PipelineConfig& config) {
    auto replacement = impl_->buildResources(config);
    impl_->resources = std::move(replacement);
    impl_->config = config;
    impl_->pipelineReady = true;
}

RenderResult Renderer::renderFrame(bool capturePixels) {
    if (!impl_->pipelineReady || !impl_->resources) {
        throw AppError(ErrorType::Internal,
                       "renderFrame called before applyPipeline");
    }

    PipelineResources& target = *impl_->resources;
    RenderResult result;
    result.width = impl_->width;
    result.height = impl_->height;
    const auto totalStart = Clock::now();

    const auto prepareStart = Clock::now();
    impl_->prepareFrame(target);
    glFinish();
    result.prepareMilliseconds =
        elapsedMilliseconds(prepareStart, Clock::now());

    bool shadowProduced = false;
    for (const StageEntry& stage : impl_->config.stages) {
        const auto start = Clock::now();
        if (stage.enabled) {
            switch (stage.type) {
            case StageType::Geometry:
                impl_->executeGeometry(target);
                break;
            case StageType::Shadow:
                impl_->executeShadow(target);
                shadowProduced = true;
                break;
            case StageType::Deferred:
                impl_->executeDeferred(target, shadowProduced);
                break;
            case StageType::PostProcess:
                impl_->executePost(target, result.effectTimings);
                break;
            }
            glFinish();
        }
        result.stageTimings.push_back(TimingEntry{
            stageName(stage.type), stage.enabled,
            stage.enabled ? elapsedMilliseconds(start, Clock::now()) : 0.0
        });
    }

    result.presentMilliseconds = impl_->present(
        target, capturePixels, result.pixels, result.readbackMilliseconds);
    result.totalMilliseconds =
        elapsedMilliseconds(totalStart, Clock::now());
    return result;
}

const PipelineConfig& Renderer::pipeline() const {
    return impl_->config;
}

int Renderer::width() const noexcept {
    return impl_->width;
}

int Renderer::height() const noexcept {
    return impl_->height;
}

const std::string& Renderer::shaderDirectory() const noexcept {
    return impl_->shaderDirectory;
}

} // namespace scene

