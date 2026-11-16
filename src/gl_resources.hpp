#pragma once

#include "logger.hpp"

#include <glad/gl.h>
#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace skin {

struct ResourceCounters {
    int framebuffersCreated = 0;
    int framebuffersReleased = 0;
    int texturesCreated = 0;
    int texturesReleased = 0;
    int renderbuffersCreated = 0;
    int renderbuffersReleased = 0;
    int programsCreated = 0;
    int programsReleased = 0;
    int shaderObjectsCreated = 0;
    int shaderObjectsReleased = 0;
    int vertexArraysCreated = 0;
    int vertexArraysReleased = 0;
    int buffersCreated = 0;
    int buffersReleased = 0;

    void logSummary(Logger& logger, const std::string& group = "-") const;
};

struct FramebufferTarget {
    GLuint framebuffer = 0;
    GLuint colorTexture = 0;
    GLuint depthRenderbuffer = 0;
    int width = 0;
    int height = 0;
    GLenum colorFormat = GL_RGBA8;
    bool depth = false;
};

class FramebufferCache {
public:
    explicit FramebufferCache(ResourceCounters& counters);
    ~FramebufferCache();

    FramebufferCache(const FramebufferCache&) = delete;
    FramebufferCache& operator=(const FramebufferCache&) = delete;

    FramebufferTarget& acquire(
        int width,
        int height,
        GLenum colorFormat,
        bool withDepth
    );
    void releaseAll();

private:
    ResourceCounters& counters_;
    std::map<std::string, FramebufferTarget> targets_;
};

class Program {
public:
    Program() = default;
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;

    GLuint id() const { return program_; }
    GLint uniform(const std::string& name);

    void set(GLint location, float value) const;
    void set(GLint location, int value) const;
    void set(GLint location, const glm::vec2& value) const;
    void set(GLint location, const glm::ivec2& value) const;
    void set(GLint location, const glm::vec3& value) const;
    void set(GLint location, const glm::mat3& value) const;
    void set(GLint location, const glm::mat4& value) const;

private:
    friend class ProgramCache;
    GLuint program_ = 0;
    std::unordered_map<std::string, GLint> uniformLocations_;
};

class ProgramCache {
public:
    explicit ProgramCache(ResourceCounters& counters);
    ~ProgramCache();

    ProgramCache(const ProgramCache&) = delete;
    ProgramCache& operator=(const ProgramCache&) = delete;

    Program& get(
        const std::string& key,
        const char* vertexSource,
        const char* fragmentSource
    );
    void releaseAll();

private:
    ResourceCounters& counters_;
    std::map<std::string, std::unique_ptr<Program>> programs_;
};

class SphereMesh {
public:
    explicit SphereMesh(ResourceCounters& counters);
    ~SphereMesh();

    SphereMesh(const SphereMesh&) = delete;
    SphereMesh& operator=(const SphereMesh&) = delete;

    void create(int slices = 128, int stacks = 64);
    void draw() const;
    void release();

private:
    ResourceCounters& counters_;
    GLuint vertexArray_ = 0;
    GLuint vertexBuffer_ = 0;
    GLuint indexBuffer_ = 0;
    GLsizei indexCount_ = 0;
};

class FullscreenVao {
public:
    explicit FullscreenVao(ResourceCounters& counters);
    ~FullscreenVao();

    FullscreenVao(const FullscreenVao&) = delete;
    FullscreenVao& operator=(const FullscreenVao&) = delete;

    void create();
    void draw() const;
    void release();

private:
    ResourceCounters& counters_;
    GLuint vertexArray_ = 0;
};

std::string glErrorName(GLenum error);

}  // namespace skin

