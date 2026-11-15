#pragma once

#include "gl.hpp"
#include "pipeline.hpp"

#include <memory>
#include <string>

namespace scene {

struct PostPassContext {
    int width = 0;
    int height = 0;
    GLuint fullscreenVao = 0;
};

class PostEffect {
public:
    virtual ~PostEffect() = default;
    virtual const std::string& type() const = 0;
    virtual void apply(const PostPassContext& context, GLuint inputTexture,
                       GLuint outputFramebuffer) = 0;
};

std::unique_ptr<PostEffect> createPostEffect(const EffectConfig& config,
                                             const std::string& shaderDirectory,
                                             int width, int height,
                                             GLuint fullscreenVao);

} // namespace scene

