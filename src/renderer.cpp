#include "renderer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <sstream>
#include <string>

#include <glad/gl.h>

#include "error.h"
#include "gl_context.h"
#include "shaders.h"

namespace sr {
namespace {

struct alignas(16) BoneBlock {
    glm::mat4 bones[4];
};

struct alignas(16) LightBlock {
    glm::vec4 directionalDirection;
    glm::vec4 directionalColorIntensity;
    glm::vec4 point0PositionRange;
    glm::vec4 point0ColorIntensity;
    glm::vec4 point1PositionRange;
    glm::vec4 point1ColorIntensity;
    glm::vec4 ambientColorIntensity;
};

struct alignas(16) ShadowBlock {
    glm::mat4 matrices[4];
    glm::vec4 splits;
    glm::vec4 parameters;
};

static_assert(sizeof(BoneBlock) == 4 * 64, "BoneBlock std140 size mismatch");
static_assert(sizeof(LightBlock) == 7 * 16, "LightBlock std140 size mismatch");
static_assert(sizeof(ShadowBlock) == 4 * 64 + 32,
              "ShadowBlock std140 size mismatch");

using Clock = std::chrono::steady_clock;

double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

const char* framebufferStatusName(GLenum status) {
    switch (status) {
    case GL_FRAMEBUFFER_COMPLETE:
        return "GL_FRAMEBUFFER_COMPLETE";
    case GL_FRAMEBUFFER_UNDEFINED:
        return "GL_FRAMEBUFFER_UNDEFINED";
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        return "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        return "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
        return "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
        return "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
    case GL_FRAMEBUFFER_UNSUPPORTED:
        return "GL_FRAMEBUFFER_UNSUPPORTED";
    case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
        return "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
    case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
        return "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS";
    default:
        return "unknown framebuffer status";
    }
}

void flipRows(std::vector<std::uint8_t>& pixels, int width, int height, int channels) {
    const std::size_t rowBytes =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
    std::vector<std::uint8_t> row(rowBytes);
    for (int y = 0; y < height / 2; ++y) {
        std::uint8_t* top = pixels.data() + static_cast<std::size_t>(y) * rowBytes;
        std::uint8_t* bottom = pixels.data() +
            static_cast<std::size_t>(height - 1 - y) * rowBytes;
        std::memcpy(row.data(), top, rowBytes);
        std::memcpy(top, bottom, rowBytes);
        std::memcpy(bottom, row.data(), rowBytes);
    }
}

void flipRows(std::vector<float>& pixels, int width, int height) {
    const std::size_t rowSize = static_cast<std::size_t>(width);
    std::vector<float> row(rowSize);
    for (int y = 0; y < height / 2; ++y) {
        float* top = pixels.data() + static_cast<std::size_t>(y) * rowSize;
        float* bottom = pixels.data() +
            static_cast<std::size_t>(height - 1 - y) * rowSize;
        std::copy(top, top + rowSize, row.begin());
        std::copy(bottom, bottom + rowSize, top);
        std::copy(row.begin(), row.end(), bottom);
    }
}

void setMat4(unsigned int program, const char* name, const glm::mat4& matrix) {
    const int location = glGetUniformLocation(program, name);
    if (location >= 0) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &matrix[0][0]);
    }
}

void setVec3(unsigned int program, const char* name, const glm::dvec3& value) {
    const int location = glGetUniformLocation(program, name);
    if (location >= 0) {
        glUniform3f(location, static_cast<float>(value.x),
                    static_cast<float>(value.y),
                    static_cast<float>(value.z));
    }
}

void setVec4(unsigned int program, const char* name, const glm::vec4& value) {
    const int location = glGetUniformLocation(program, name);
    if (location >= 0) {
        glUniform4fv(location, 1, &value.x);
    }
}

void setInt(unsigned int program, const char* name, int value) {
    const int location = glGetUniformLocation(program, name);
    if (location >= 0) {
        glUniform1i(location, value);
    }
}

} // namespace

Renderer::Renderer(GLContext& context, RendererSettings settings)
    : context_(context), settings_(settings) {
    context_.makeCurrent();
    try {
        initialize();
    } catch (...) {
        shutdown();
        throw;
    }
}

Renderer::~Renderer() {
    shutdown();
}

void Renderer::initialize() {
    if (settings_.cascadeCount < 2 || settings_.cascadeCount > 4 ||
        settings_.shadowSize < 256 || settings_.shadowSize > 4096 ||
        (settings_.shadowSize & (settings_.shadowSize - 1)) != 0 ||
        settings_.width < 1 || settings_.height < 1) {
        fail(ExitCode::Cascade, "E_CASCADE_CONFIG",
             "renderer received an invalid framebuffer or cascade configuration");
    }
    initialized_ = true;

    const auto* versionString =
        reinterpret_cast<const char*>(glGetString(GL_VERSION));
    const auto* rendererString =
        reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    glVersion_ = versionString != nullptr ? versionString : "unknown";
    glRenderer_ = rendererString != nullptr ? rendererString : "unknown";

    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DITHER);
    glDisable(GL_MULTISAMPLE);
    glDisable(GL_SCISSOR_TEST);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glDepthRange(0.0, 1.0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glClearDepth(1.0);

    createPrograms();
    createUniformBuffers();
    createTargets();
    checkGl("renderer initialization");
}

void Renderer::shutdown() {
    if (!initialized_) {
        return;
    }
    context_.makeCurrent();

    destroyGpuMesh(character_);
    destroyGpuMesh(occluders_);
    destroyGpuMesh(ground_);

    const unsigned int programs[] = {
        skinnedForward_, skinnedDepth_, staticForward_,
        staticDepth_, skinnedShadow_, staticShadow_,
    };
    for (unsigned int program : programs) {
        if (program != 0) {
            glDeleteProgram(program);
        }
    }
    skinnedForward_ = 0;
    skinnedDepth_ = 0;
    staticForward_ = 0;
    staticDepth_ = 0;
    skinnedShadow_ = 0;
    staticShadow_ = 0;

    const unsigned int buffers[] = {boneUbo_, lightUbo_, shadowUbo_};
    for (unsigned int buffer : buffers) {
        if (buffer != 0) {
            glDeleteBuffers(1, &buffer);
        }
    }
    boneUbo_ = 0;
    lightUbo_ = 0;
    shadowUbo_ = 0;

    if (colorTexture_ != 0) {
        glDeleteTextures(1, &colorTexture_);
    }
    if (depthTexture_ != 0) {
        glDeleteTextures(1, &depthTexture_);
    }
    colorTexture_ = 0;
    depthTexture_ = 0;
    if (mainFbo_ != 0) {
        glDeleteFramebuffers(1, &mainFbo_);
    }
    mainFbo_ = 0;

    for (unsigned int& texture : shadowTextures_) {
        if (texture != 0) {
            glDeleteTextures(1, &texture);
        }
        texture = 0;
    }
    for (unsigned int& framebuffer : shadowFbos_) {
        if (framebuffer != 0) {
            glDeleteFramebuffers(1, &framebuffer);
        }
        framebuffer = 0;
    }
    initialized_ = false;
}

unsigned int Renderer::compileShader(unsigned int type,
                                     const char* source,
                                     const char* debugName) {
    const unsigned int shader = glCreateShader(type);
    if (shader == 0) {
        fail(ExitCode::Shader, "E_SHADER_CREATE",
             std::string("glCreateShader failed for ") + debugName);
    }
    std::string shaderSource = source;
    if (settings_.forceShaderError) {
        shaderSource += "\n#error forced shader compilation failure\n";
    }
    const char* sourcePointer = shaderSource.c_str();
    glShaderSource(shader, 1, &sourcePointer, nullptr);
    glCompileShader(shader);

    GLint status = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        GLint logLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        std::string log(static_cast<std::size_t>(std::max(logLength, 1)), '\0');
        glGetShaderInfoLog(shader, logLength, nullptr, log.data());
        glDeleteShader(shader);
        fail(ExitCode::Shader, "E_SHADER_COMPILE",
             std::string("shader compilation failed for ") + debugName + ": " + log);
    }
    return shader;
}

unsigned int Renderer::compileProgram(const char* vertexSource,
                                      const char* fragmentSource,
                                      const char* debugName) {
    const unsigned int vertex =
        compileShader(GL_VERTEX_SHADER, vertexSource, debugName);
    unsigned int fragment = 0;
    try {
        fragment = compileShader(GL_FRAGMENT_SHADER, fragmentSource, debugName);
    } catch (...) {
        glDeleteShader(vertex);
        throw;
    }

    const unsigned int program = glCreateProgram();
    if (program == 0) {
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        fail(ExitCode::Shader, "E_SHADER_LINK",
             std::string("glCreateProgram failed for ") + debugName);
    }
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    GLint status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        GLint logLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
        std::string log(static_cast<std::size_t>(std::max(logLength, 1)), '\0');
        glGetProgramInfoLog(program, logLength, nullptr, log.data());
        glDeleteProgram(program);
        fail(ExitCode::Shader, "E_SHADER_LINK",
             std::string("program link failed for ") + debugName + ": " + log);
    }
    return program;
}

void Renderer::createPrograms() {
    skinnedForward_ = compileProgram(shaders::kSkinnedSceneVertex,
                                     shaders::kForwardFragment,
                                     "skinned_forward");
    skinnedDepth_ = compileProgram(shaders::kSkinnedSceneVertex,
                                   shaders::kDepthFragment,
                                   "skinned_depth");
    staticForward_ = compileProgram(shaders::kStaticSceneVertex,
                                    shaders::kForwardFragment,
                                    "static_forward");
    staticDepth_ = compileProgram(shaders::kStaticSceneVertex,
                                  shaders::kDepthFragment,
                                  "static_depth");
    skinnedShadow_ = compileProgram(shaders::kSkinnedShadowVertex,
                                    shaders::kDepthFragment,
                                    "skinned_shadow");
    staticShadow_ = compileProgram(shaders::kStaticShadowVertex,
                                   shaders::kDepthFragment,
                                   "static_shadow");

    const unsigned int forwardPrograms[] = {skinnedForward_, staticForward_};
    for (unsigned int program : forwardPrograms) {
        glUseProgram(program);
        setInt(program, "uShadowMap0", 2);
        setInt(program, "uShadowMap1", 3);
        setInt(program, "uShadowMap2", 4);
        setInt(program, "uShadowMap3", 5);
    }
    glUseProgram(0);
}

void Renderer::createUniformBuffers() {
    auto createBuffer = [](unsigned int& buffer, std::size_t size, unsigned int binding) {
        glGenBuffers(1, &buffer);
        glBindBuffer(GL_UNIFORM_BUFFER, buffer);
        glBufferData(GL_UNIFORM_BUFFER, static_cast<GLsizeiptr>(size),
                     nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, binding, buffer);
    };
    createBuffer(boneUbo_, sizeof(BoneBlock), 0);
    createBuffer(shadowUbo_, sizeof(ShadowBlock), 1);
    createBuffer(lightUbo_, sizeof(LightBlock), 2);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Renderer::createTargets() {
    glGenFramebuffers(1, &mainFbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, mainFbo_);

    glGenTextures(1, &colorTexture_);
    glBindTexture(GL_TEXTURE_2D, colorTexture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, settings_.width, settings_.height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, colorTexture_, 0);

    glGenTextures(1, &depthTexture_);
    glBindTexture(GL_TEXTURE_2D, depthTexture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F,
                 settings_.width, settings_.height, 0,
                 GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                           GL_TEXTURE_2D, depthTexture_, 0);
    const GLenum drawBuffer = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &drawBuffer);

    unsigned int forcedIncompleteRenderbuffer = 0;
    if (settings_.forceFramebufferError) {
        glGenRenderbuffers(1, &forcedIncompleteRenderbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, forcedIncompleteRenderbuffer);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_RENDERBUFFER, forcedIncompleteRenderbuffer);
    }

    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        if (forcedIncompleteRenderbuffer != 0) {
            glDeleteRenderbuffers(1, &forcedIncompleteRenderbuffer);
        }
        std::ostringstream message;
        message << "main color/depth framebuffer is incomplete: 0x"
                << std::hex << status << " (" << framebufferStatusName(status) << ")";
        fail(ExitCode::Framebuffer, "E_FRAMEBUFFER", message.str());
    }

    for (int index = 0; index < settings_.cascadeCount; ++index) {
        unsigned int& texture = shadowTextures_[static_cast<std::size_t>(index)];
        unsigned int& framebuffer = shadowFbos_[static_cast<std::size_t>(index)];

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F,
                     settings_.shadowSize, settings_.shadowSize, 0,
                     GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        const GLfloat borderColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);

        glGenFramebuffers(1, &framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_2D, texture, 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        const GLenum shadowStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (shadowStatus != GL_FRAMEBUFFER_COMPLETE) {
            std::ostringstream message;
            message << "cascade " << index
                    << " shadow framebuffer is incomplete: 0x"
                    << std::hex << shadowStatus << " ("
                    << framebufferStatusName(shadowStatus) << ")";
            fail(ExitCode::Framebuffer, "E_FRAMEBUFFER", message.str());
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::createGpuMesh(GpuMesh& out, const MeshData& mesh, bool dynamic) {
    glGenVertexArrays(1, &out.vao);
    glGenBuffers(1, &out.vbo);
    glGenBuffers(1, &out.ebo);
    glBindVertexArray(out.vao);
    glBindBuffer(GL_ARRAY_BUFFER, out.vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(mesh.vertices.size() * sizeof(SkinnedVertex)),
                 mesh.vertices.data(), dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, out.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(mesh.indices.size() * sizeof(std::uint32_t)),
                 mesh.indices.data(), dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);

    constexpr GLsizei stride = sizeof(SkinnedVertex);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
                          reinterpret_cast<const void*>(offsetof(SkinnedVertex, px)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                          reinterpret_cast<const void*>(offsetof(SkinnedVertex, nx)));
    glEnableVertexAttribArray(2);
    glVertexAttribIPointer(2, 4, GL_UNSIGNED_INT, stride,
                           reinterpret_cast<const void*>(offsetof(SkinnedVertex, boneIndices)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride,
                          reinterpret_cast<const void*>(offsetof(SkinnedVertex, boneWeights)));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    out.indexCount = static_cast<int>(mesh.indices.size());
}

void Renderer::updateGpuMesh(GpuMesh& out, const MeshData& mesh) {
    if (out.vao == 0) {
        createGpuMesh(out, mesh, true);
        return;
    }
    glBindVertexArray(out.vao);
    glBindBuffer(GL_ARRAY_BUFFER, out.vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(mesh.vertices.size() * sizeof(SkinnedVertex)),
                 mesh.vertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, out.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(mesh.indices.size() * sizeof(std::uint32_t)),
                 mesh.indices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    out.indexCount = static_cast<int>(mesh.indices.size());
}

void Renderer::destroyGpuMesh(GpuMesh& mesh) {
    if (mesh.ebo != 0) {
        glDeleteBuffers(1, &mesh.ebo);
    }
    if (mesh.vbo != 0) {
        glDeleteBuffers(1, &mesh.vbo);
    }
    if (mesh.vao != 0) {
        glDeleteVertexArrays(1, &mesh.vao);
    }
    mesh = {};
}

void Renderer::uploadCharacter(const MeshData& mesh) {
    updateGpuMesh(character_, mesh);
}

void Renderer::uploadStaticMeshes(const MeshData& occluders, const MeshData& ground) {
    if (occluders_.vao == 0) {
        createGpuMesh(occluders_, occluders, false);
    }
    if (ground_.vao == 0) {
        createGpuMesh(ground_, ground, false);
    }
}

void Renderer::drawMesh(const GpuMesh& mesh) const {
    if (mesh.indexCount <= 0) {
        return;
    }
    glBindVertexArray(mesh.vao);
    glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, nullptr);
}

void Renderer::uploadBoneBlock(const PoseResult& pose) {
    BoneBlock block{};
    for (int i = 0; i < 4; ++i) {
        block.bones[i] = glm::mat4(1.0f);
    }
    const std::size_t count =
        std::min<std::size_t>(pose.skinMatrices.size(), 4);
    for (std::size_t i = 0; i < count; ++i) {
        block.bones[i] = toMat4(pose.skinMatrices[i]);
    }
    glBindBuffer(GL_UNIFORM_BUFFER, boneUbo_);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(block), &block);
}

void Renderer::uploadLightBlock(const Lights& lights) {
    LightBlock block{};
    block.directionalDirection = toVec4(
        lights.directional.directionTowardScene, 0.0);
    block.directionalColorIntensity = toVec4(
        lights.directional.color, lights.directional.intensity);
    block.point0PositionRange = toVec4(
        lights.point0.position, lights.point0.range);
    block.point0ColorIntensity = toVec4(
        lights.point0.color, lights.point0.intensity);
    block.point1PositionRange = toVec4(
        lights.point1.position, lights.point1.range);
    block.point1ColorIntensity = toVec4(
        lights.point1.color, lights.point1.intensity);
    block.ambientColorIntensity = toVec4(
        lights.ambientColor, lights.ambientIntensity);
    glBindBuffer(GL_UNIFORM_BUFFER, lightUbo_);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(block), &block);
}

void Renderer::uploadShadowBlock(const CascadeSet& cascades) {
    ShadowBlock block{};
    const double lastSplit = cascades.cascades.empty()
        ? 1.0
        : cascades.cascades.back().splitFar;
    for (int i = 0; i < 4; ++i) {
        block.matrices[i] = glm::mat4(1.0f);
    }
    for (std::size_t i = 0; i < cascades.cascades.size() && i < 4; ++i) {
        block.matrices[i] = toMat4(cascades.cascades[i].lightViewProjection);
    }
    block.splits = glm::vec4(
        cascades.cascades.size() > 0 ? static_cast<float>(cascades.cascades[0].splitFar) : static_cast<float>(lastSplit),
        cascades.cascades.size() > 1 ? static_cast<float>(cascades.cascades[1].splitFar) : static_cast<float>(lastSplit),
        cascades.cascades.size() > 2 ? static_cast<float>(cascades.cascades[2].splitFar) : static_cast<float>(lastSplit),
        cascades.cascades.size() > 3 ? static_cast<float>(cascades.cascades[3].splitFar) : static_cast<float>(lastSplit));
    block.parameters = glm::vec4(
        static_cast<float>(cascades.cascades.size()),
        static_cast<float>(settings_.shadowSize),
        0.00045f,
        0.0028f);
    glBindBuffer(GL_UNIFORM_BUFFER, shadowUbo_);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(block), &block);
}

void Renderer::drawShadowPass(const CascadeSet& cascades,
                              std::vector<double>& shadowCoverage) {
    const glm::mat4 identity(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(2.0f, 4.0f);

    for (std::size_t index = 0;
         index < cascades.cascades.size() &&
         index < static_cast<std::size_t>(settings_.cascadeCount);
         ++index) {
        glBindFramebuffer(GL_FRAMEBUFFER,
                          shadowFbos_[static_cast<std::size_t>(index)]);
        glViewport(0, 0, settings_.shadowSize, settings_.shadowSize);
        glClear(GL_DEPTH_BUFFER_BIT);

        glUseProgram(skinnedShadow_);
        setMat4(skinnedShadow_, "uModel", identity);
        setInt(skinnedShadow_, "uCascadeIndex", static_cast<int>(index));
        drawMesh(character_);

        glUseProgram(staticShadow_);
        setMat4(staticShadow_, "uModel", identity);
        setInt(staticShadow_, "uCascadeIndex", static_cast<int>(index));
        drawMesh(occluders_);
    }

    glDisable(GL_POLYGON_OFFSET_FILL);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glFinish();
    shadowCoverage.clear();
    shadowCoverage.reserve(cascades.cascades.size());
    std::vector<float> pixels(
        static_cast<std::size_t>(settings_.shadowSize) *
        static_cast<std::size_t>(settings_.shadowSize));
    for (std::size_t index = 0;
         index < cascades.cascades.size() &&
         index < static_cast<std::size_t>(settings_.cascadeCount);
         ++index) {
        glBindTexture(GL_TEXTURE_2D, shadowTextures_[index]);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                      pixels.data());
        const std::size_t occupied = static_cast<std::size_t>(std::count_if(
            pixels.begin(), pixels.end(),
            [](float depth) { return depth < 0.999999f; }));
        shadowCoverage.push_back(
            static_cast<double>(occupied) /
            static_cast<double>(pixels.size()));
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::drawMainPass(const PoseResult& pose,
                            const Camera& camera,
                            const CascadeSet& cascades,
                            const Lights& lights,
                            const Material& characterMaterial,
                            const Material& occluderMaterial,
                            const Material& groundMaterial,
                            double& depthPrepassMs,
                            double& colorPassMs) {
    (void)pose;
    (void)lights;
    const glm::mat4 identity(1.0f);
    const glm::mat4 viewProjection = toMat4(camera.viewProjection);
    const glm::mat4 view = toMat4(camera.view);

    glBindFramebuffer(GL_FRAMEBUFFER, mainFbo_);
    glViewport(0, 0, settings_.width, settings_.height);
    glClearColor(0.025f, 0.032f, 0.052f, 1.0f);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glDepthFunc(GL_LESS);
    auto phaseStart = Clock::now();
    glUseProgram(staticDepth_);
    setMat4(staticDepth_, "uViewProjection", viewProjection);
    setMat4(staticDepth_, "uModel", identity);
    drawMesh(ground_);
    drawMesh(occluders_);
    glUseProgram(skinnedDepth_);
    setMat4(skinnedDepth_, "uViewProjection", viewProjection);
    setMat4(skinnedDepth_, "uModel", identity);
    drawMesh(character_);
    glFinish();
    depthPrepassMs = elapsedMs(phaseStart, Clock::now());
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_FALSE);
    phaseStart = Clock::now();
    for (int index = 0; index < 4; ++index) {
        const std::size_t sourceIndex =
            std::min<std::size_t>(static_cast<std::size_t>(index),
                                  cascades.cascades.size() - 1);
        glActiveTexture(GL_TEXTURE0 + 2 + index);
        glBindTexture(GL_TEXTURE_2D,
                      shadowTextures_[sourceIndex]);
    }

    auto setCommonUniforms = [&](unsigned int program) {
        setMat4(program, "uViewProjection", viewProjection);
        setMat4(program, "uView", view);
        setVec3(program, "uCameraPosition", camera.eye);
    };

    auto drawLitMesh = [&](unsigned int program,
                           const GpuMesh& mesh,
                           const Material& material) {
        setMat4(program, "uModel", identity);
        setVec3(program, "uBaseColor", material.baseColor);
        setVec4(program, "uMaterial", glm::vec4(
            static_cast<float>(material.roughness),
            static_cast<float>(material.specular),
            static_cast<float>(material.checkerScale),
            0.0f));
        drawMesh(mesh);
    };

    glUseProgram(staticForward_);
    setCommonUniforms(staticForward_);
    drawLitMesh(staticForward_, ground_, groundMaterial);
    drawLitMesh(staticForward_, occluders_, occluderMaterial);

    glUseProgram(skinnedForward_);
    setCommonUniforms(skinnedForward_);
    drawLitMesh(skinnedForward_, character_, characterMaterial);
    glFinish();
    colorPassMs = elapsedMs(phaseStart, Clock::now());
    glDepthMask(GL_TRUE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

FrameRenderResult Renderer::renderFrame(const PoseResult& pose,
                                        const Camera& camera,
                                        const CascadeSet& cascades,
                                        const Lights& lights,
                                        const Material& characterMaterial,
                                        const Material& occluderMaterial,
                                        const Material& groundMaterial) {
    if (cascades.cascades.empty()) {
        fail(ExitCode::Cascade, "E_CASCADE_CONFIG", "no cascades were provided");
    }
    const auto frameStart = Clock::now();
    FrameRenderResult result;

    uploadBoneBlock(pose);
    uploadLightBlock(lights);
    uploadShadowBlock(cascades);
    context_.makeCurrent();

    auto phaseStart = Clock::now();
    drawShadowPass(cascades, result.shadowCoverage);
    glFinish();
    auto phaseEnd = Clock::now();
    const double shadowMs = elapsedMs(phaseStart, phaseEnd);

    double depthPrepassMs = 0.0;
    double colorPassMs = 0.0;
    drawMainPass(pose, camera, cascades, lights,
                 characterMaterial, occluderMaterial, groundMaterial,
                 depthPrepassMs, colorPassMs);
    glFinish();

    phaseStart = Clock::now();
    result.shadowPassMs = shadowMs;
    result.depthPrepassMs = depthPrepassMs;
    result.colorPassMs = colorPassMs;
    result.shadowMatrices.reserve(cascades.cascades.size());
    for (const Cascade& cascade : cascades.cascades) {
        result.shadowMatrices.push_back(cascade.lightViewProjection);
    }

    const std::size_t colorBytes =
        static_cast<std::size_t>(settings_.width) *
        static_cast<std::size_t>(settings_.height) * 4U;
    result.colorRgbaTopDown.resize(colorBytes);
    result.depthTopDown.resize(
        static_cast<std::size_t>(settings_.width) *
        static_cast<std::size_t>(settings_.height));

    glBindFramebuffer(GL_READ_FRAMEBUFFER, mainFbo_);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, settings_.width, settings_.height,
                 GL_RGBA, GL_UNSIGNED_BYTE, result.colorRgbaTopDown.data());
    glReadPixels(0, 0, settings_.width, settings_.height,
                 GL_DEPTH_COMPONENT, GL_FLOAT, result.depthTopDown.data());
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    flipRows(result.colorRgbaTopDown, settings_.width, settings_.height, 4);
    flipRows(result.depthTopDown, settings_.width, settings_.height);
    phaseEnd = Clock::now();
    result.readbackMs = elapsedMs(phaseStart, phaseEnd);
    result.totalMs = elapsedMs(frameStart, Clock::now());
    checkGl("frame render");
    return result;
}

void Renderer::checkGl(const char* operation) {
    GLenum error = glGetError();
    if (error == GL_NO_ERROR) {
        return;
    }
    std::ostringstream message;
    message << "OpenGL error after " << operation << ": 0x"
            << std::hex << error;
    while (error != GL_NO_ERROR) {
        error = glGetError();
    }
    fail(ExitCode::Internal, "E_GL_STATE", message.str());
}

} // namespace sr
