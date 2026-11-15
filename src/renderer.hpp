#pragma once

#include "frame_data.hpp"
#include "gl_loader.hpp"
#include "scene.hpp"

#include <string>
#include <vector>

namespace demo {

enum class ShaderMode {
    CustomLighting,
    SafeFallback
};

struct RendererOptions {
    bool injectShaderCompileFailure = false;
    bool injectShaderLinkFailure = false;
};

class Renderer {
public:
    Renderer() = default;
    ~Renderer();

    Renderer(const Renderer &) = delete;
    Renderer &operator=(const Renderer &) = delete;

    void initialize(const MeshData &mesh, const RendererOptions &options);
    void updateFrame(const FrameUniforms &frame);
    void updateMeshVertices(const std::vector<Vertex> &vertices);
    void draw();
    FrameUniforms readBackFrame();
    void destroy();

    ShaderMode shaderMode() const noexcept {
        return shaderMode_;
    }

    std::size_t frameUploadCount() const noexcept {
        return frameUploadCount_;
    }

    std::size_t meshUpdateCount() const noexcept {
        return meshUpdateCount_;
    }

    std::size_t headerUploadCount() const noexcept {
        return headerUploadCount_;
    }

private:
    GLuint program_ = 0;
    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    GLuint ebo_ = 0;
    GLuint ubo_ = 0;
    GLsizei indexCount_ = 0;
    std::size_t vertexCount_ = 0;
    std::size_t frameUploadCount_ = 0;
    std::size_t meshUpdateCount_ = 0;
    std::size_t headerUploadCount_ = 0;
    bool headerUploaded_ = false;
    glm::mat4 lastViewProjection_{};
    glm::mat4 lastModel_{};
    ShaderMode shaderMode_ = ShaderMode::CustomLighting;
};

} // namespace demo
