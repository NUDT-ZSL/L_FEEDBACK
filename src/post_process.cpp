#include "post_process.hpp"

#include "error.hpp"

#include <algorithm>

namespace scene {
namespace {

std::string shaderPath(const std::string& directory, const std::string& file) {
    if (directory.empty()) {
        return file;
    }
    if (directory.back() == '/' || directory.back() == '\\') {
        return directory + file;
    }
    return directory + "/" + file;
}

template <typename T>
T parameter(const EffectConfig& config, const char* name) {
    return config.params.at(name).get<T>();
}

class BloomEffect final : public PostEffect {
public:
    BloomEffect(const EffectConfig& config, const std::string& shaderDirectory,
                int width, int height, GLuint fullscreenVao)
        : type_(config.type),
          threshold_(parameter<float>(config, "threshold")),
          intensity_(parameter<float>(config, "intensity")),
          radius_(parameter<float>(config, "radius")),
          knee_(parameter<float>(config, "knee")),
          fullscreenVao_(fullscreenVao) {
        thresholdProgram_ = ShaderProgram::fromFiles(
            shaderPath(shaderDirectory, "fullscreen.vert"),
            shaderPath(shaderDirectory, "post_bloom_threshold.frag"));
        blurProgram_ = ShaderProgram::fromFiles(
            shaderPath(shaderDirectory, "fullscreen.vert"),
            shaderPath(shaderDirectory, "post_bloom_blur.frag"));
        compositeProgram_ = ShaderProgram::fromFiles(
            shaderPath(shaderDirectory, "fullscreen.vert"),
            shaderPath(shaderDirectory, "post_bloom_composite.frag"));

        halfWidth_ = std::max(1, width / 2);
        halfHeight_ = std::max(1, height / 2);
        halfA_.setLabel("bloom-half-a");
        halfA_.create2D(halfWidth_, halfHeight_, GL_RGBA16F, GL_RGBA, GL_FLOAT,
                        GL_LINEAR, GL_LINEAR, GL_CLAMP_TO_EDGE,
                        GL_CLAMP_TO_EDGE, nullptr);
        halfB_.setLabel("bloom-half-b");
        halfB_.create2D(halfWidth_, halfHeight_, GL_RGBA16F, GL_RGBA, GL_FLOAT,
                        GL_LINEAR, GL_LINEAR, GL_CLAMP_TO_EDGE,
                        GL_CLAMP_TO_EDGE, nullptr);

        fboA_.setLabel("bloom-half-a");
        fboA_.create();
        fboA_.attachColor(0, halfA_);
        fboA_.checkComplete("bloom-half-a");
        fboB_.setLabel("bloom-half-b");
        fboB_.create();
        fboB_.attachColor(0, halfB_);
        fboB_.checkComplete("bloom-half-b");
    }

    const std::string& type() const override { return type_; }

    void apply(const PostPassContext& context, GLuint inputTexture,
               GLuint outputFramebuffer) override {
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);

        glBindFramebuffer(GL_FRAMEBUFFER, fboA_.id());
        glViewport(0, 0, halfWidth_, halfHeight_);
        thresholdProgram_.use();
        bindTexture(0, inputTexture);
        thresholdProgram_.setInt("uInput", 0);
        thresholdProgram_.setFloat("uThreshold", threshold_);
        thresholdProgram_.setFloat("uKnee", knee_);
        drawFullscreenTriangle(fullscreenVao_);

        glBindFramebuffer(GL_FRAMEBUFFER, fboB_.id());
        glViewport(0, 0, halfWidth_, halfHeight_);
        blurProgram_.use();
        bindTexture(0, halfA_.id());
        blurProgram_.setInt("uInput", 0);
        blurProgram_.setVec2("uDirection", radius_ / static_cast<float>(halfWidth_), 0.0f);
        drawFullscreenTriangle(fullscreenVao_);

        glBindFramebuffer(GL_FRAMEBUFFER, fboA_.id());
        glViewport(0, 0, halfWidth_, halfHeight_);
        blurProgram_.use();
        bindTexture(0, halfB_.id());
        blurProgram_.setInt("uInput", 0);
        blurProgram_.setVec2("uDirection", 0.0f, radius_ / static_cast<float>(halfHeight_));
        drawFullscreenTriangle(fullscreenVao_);

        glBindFramebuffer(GL_FRAMEBUFFER, outputFramebuffer);
        glViewport(0, 0, context.width, context.height);
        compositeProgram_.use();
        bindTexture(0, inputTexture);
        bindTexture(1, halfA_.id());
        compositeProgram_.setInt("uScene", 0);
        compositeProgram_.setInt("uBloom", 1);
        compositeProgram_.setFloat("uIntensity", intensity_);
        drawFullscreenTriangle(fullscreenVao_);
        checkGlError("applying bloom");
    }

private:
    std::string type_;
    float threshold_ = 0.9f;
    float intensity_ = 0.65f;
    float radius_ = 1.0f;
    float knee_ = 0.35f;
    GLuint fullscreenVao_ = 0;
    int halfWidth_ = 1;
    int halfHeight_ = 1;
    ShaderProgram thresholdProgram_;
    ShaderProgram blurProgram_;
    ShaderProgram compositeProgram_;
    GlTexture halfA_;
    GlTexture halfB_;
    GlFramebuffer fboA_;
    GlFramebuffer fboB_;
};

class TonemapEffect final : public PostEffect {
public:
    TonemapEffect(const EffectConfig& config, const std::string& shaderDirectory,
                  GLuint fullscreenVao)
        : type_(config.type),
          exposure_(parameter<float>(config, "exposure")),
          mode_(parameter<std::string>(config, "mode") == "reinhard" ? 1 : 0),
          fullscreenVao_(fullscreenVao) {
        program_ = ShaderProgram::fromFiles(
            shaderPath(shaderDirectory, "fullscreen.vert"),
            shaderPath(shaderDirectory, "post_tonemap.frag"));
    }

    const std::string& type() const override { return type_; }

    void apply(const PostPassContext& context, GLuint inputTexture,
               GLuint outputFramebuffer) override {
        glBindFramebuffer(GL_FRAMEBUFFER, outputFramebuffer);
        glViewport(0, 0, context.width, context.height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        program_.use();
        bindTexture(0, inputTexture);
        program_.setInt("uInput", 0);
        program_.setFloat("uExposure", exposure_);
        program_.setInt("uMode", mode_);
        drawFullscreenTriangle(fullscreenVao_);
        checkGlError("applying tonemap");
    }

private:
    std::string type_;
    float exposure_ = 1.0f;
    int mode_ = 0;
    GLuint fullscreenVao_ = 0;
    ShaderProgram program_;
};

class FxaaEffect final : public PostEffect {
public:
    FxaaEffect(const EffectConfig& config, const std::string& shaderDirectory,
               GLuint fullscreenVao)
        : type_(config.type),
          edgeThreshold_(parameter<float>(config, "edge_threshold")),
          fullscreenVao_(fullscreenVao) {
        program_ = ShaderProgram::fromFiles(
            shaderPath(shaderDirectory, "fullscreen.vert"),
            shaderPath(shaderDirectory, "post_fxaa.frag"));
    }

    const std::string& type() const override { return type_; }

    void apply(const PostPassContext& context, GLuint inputTexture,
               GLuint outputFramebuffer) override {
        glBindFramebuffer(GL_FRAMEBUFFER, outputFramebuffer);
        glViewport(0, 0, context.width, context.height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        program_.use();
        bindTexture(0, inputTexture);
        program_.setInt("uInput", 0);
        program_.setVec2("uInvResolution",
                         1.0f / static_cast<float>(context.width),
                         1.0f / static_cast<float>(context.height));
        program_.setFloat("uEdgeThreshold", edgeThreshold_);
        drawFullscreenTriangle(fullscreenVao_);
        checkGlError("applying fxaa");
    }

private:
    std::string type_;
    float edgeThreshold_ = 0.08f;
    GLuint fullscreenVao_ = 0;
    ShaderProgram program_;
};

class ColorGradeEffect final : public PostEffect {
public:
    ColorGradeEffect(const EffectConfig& config,
                     const std::string& shaderDirectory, GLuint fullscreenVao)
        : type_(config.type), fullscreenVao_(fullscreenVao) {
        const auto tint = config.params.at("tint");
        const auto lift = config.params.at("lift");
        tint_ = {tint[0].get<float>(), tint[1].get<float>(),
                 tint[2].get<float>()};
        lift_ = {lift[0].get<float>(), lift[1].get<float>(),
                 lift[2].get<float>()};
        contrast_ = parameter<float>(config, "contrast");
        saturation_ = parameter<float>(config, "saturation");
        program_ = ShaderProgram::fromFiles(
            shaderPath(shaderDirectory, "fullscreen.vert"),
            shaderPath(shaderDirectory, "post_color_grade.frag"));
    }

    const std::string& type() const override { return type_; }

    void apply(const PostPassContext& context, GLuint inputTexture,
               GLuint outputFramebuffer) override {
        glBindFramebuffer(GL_FRAMEBUFFER, outputFramebuffer);
        glViewport(0, 0, context.width, context.height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        program_.use();
        bindTexture(0, inputTexture);
        program_.setInt("uInput", 0);
        program_.setFloat("uContrast", contrast_);
        program_.setFloat("uSaturation", saturation_);
        program_.setVec3("uTint", tint_);
        program_.setVec3("uLift", lift_);
        drawFullscreenTriangle(fullscreenVao_);
        checkGlError("applying color grade");
    }

private:
    std::string type_;
    float contrast_ = 1.0f;
    float saturation_ = 1.0f;
    Vec3 tint_{1.0f, 1.0f, 1.0f};
    Vec3 lift_{0.0f, 0.0f, 0.0f};
    GLuint fullscreenVao_ = 0;
    ShaderProgram program_;
};

} // namespace

std::unique_ptr<PostEffect> createPostEffect(const EffectConfig& config,
                                             const std::string& shaderDirectory,
                                             int width, int height,
                                             GLuint fullscreenVao) {
    if (config.type == "bloom") {
        return std::make_unique<BloomEffect>(config, shaderDirectory, width,
                                             height, fullscreenVao);
    }
    if (config.type == "tonemap") {
        return std::make_unique<TonemapEffect>(config, shaderDirectory,
                                               fullscreenVao);
    }
    if (config.type == "fxaa") {
        return std::make_unique<FxaaEffect>(config, shaderDirectory,
                                            fullscreenVao);
    }
    if (config.type == "color_grade") {
        return std::make_unique<ColorGradeEffect>(config, shaderDirectory,
                                                  fullscreenVao);
    }
    throw AppError(ErrorType::UnknownEffect,
                   "cannot create unknown post effect '" + config.type + "'");
}

} // namespace scene

