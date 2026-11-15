#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "error.hpp"
#include "math.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace scene {

class GlTexture {
public:
    GlTexture() = default;
    ~GlTexture();
    GlTexture(const GlTexture&) = delete;
    GlTexture& operator=(const GlTexture&) = delete;
    GlTexture(GlTexture&& other) noexcept;
    GlTexture& operator=(GlTexture&& other) noexcept;

    void create2D(int width, int height, GLenum internalFormat,
                  GLenum format, GLenum type, GLenum minFilter,
                  GLenum magFilter, GLenum wrapS, GLenum wrapT,
                  const void* pixels = nullptr);
    void setLabel(const std::string& label) { label_ = label; }

    GLuint id() const noexcept { return id_; }
    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }

private:
    void release();
    GLuint id_ = 0;
    int width_ = 0;
    int height_ = 0;
    std::string label_;
};

class GlFramebuffer {
public:
    GlFramebuffer() = default;
    ~GlFramebuffer();
    GlFramebuffer(const GlFramebuffer&) = delete;
    GlFramebuffer& operator=(const GlFramebuffer&) = delete;
    GlFramebuffer(GlFramebuffer&& other) noexcept;
    GlFramebuffer& operator=(GlFramebuffer&& other) noexcept;

    void create();
    void attachColor(size_t index, const GlTexture& texture);
    void attachDepth(const GlTexture& texture);
    void checkComplete(const std::string& label) const;
    void setLabel(const std::string& label) { label_ = label; }

    GLuint id() const noexcept { return id_; }

private:
    void release();
    GLuint id_ = 0;
    std::string label_;
};

class ShaderProgram {
public:
    ShaderProgram() = default;
    ~ShaderProgram();
    ShaderProgram(const ShaderProgram&) = delete;
    ShaderProgram& operator=(const ShaderProgram&) = delete;
    ShaderProgram(ShaderProgram&& other) noexcept;
    ShaderProgram& operator=(ShaderProgram&& other) noexcept;

    static ShaderProgram fromFiles(const std::string& vertexPath,
                                   const std::string& fragmentPath);
    static ShaderProgram fromSources(const std::string& vertexSource,
                                     const std::string& fragmentSource,
                                     const std::string& label);

    void use() const;
    GLuint id() const noexcept { return id_; }
    GLint location(const char* name);

    void setInt(const char* name, int value);
    void setFloat(const char* name, float value);
    void setVec2(const char* name, float x, float y);
    void setVec3(const char* name, Vec3 value);
    void setMat3(const char* name, const Mat3& value);
    void setMat4(const char* name, const Mat4& value);
    void setIntArray(const char* name, const std::vector<int>& values);
    void setVec2Array(const char* name, const std::vector<float>& values);
    void setVec3Array(const char* name, const std::vector<Vec3>& values);
    void setFloatArray(const char* name, const std::vector<float>& values);

private:
    void release();
    GLuint id_ = 0;
    std::string label_;
    std::unordered_map<std::string, GLint> uniforms_;
};

std::string readTextFile(const std::string& path);
void checkGlError(const std::string& context);
void drawFullscreenTriangle(GLuint vao);
void bindTexture(GLuint unit, GLuint texture);

} // namespace scene

