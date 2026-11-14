#pragma once

#include <glad/glad.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace dr {

class GlTexture {
public:
    GlTexture() = default;
    ~GlTexture();
    GlTexture(const GlTexture&) = delete;
    GlTexture& operator=(const GlTexture&) = delete;
    GlTexture(GlTexture&& other) noexcept;
    GlTexture& operator=(GlTexture&& other) noexcept;

    void create2D(int width, int height, GLint internalFormat, GLenum format, GLenum type,
                  const void* pixels = nullptr, GLint filter = GL_NEAREST);
    void destroy();
    GLuint id() const { return id_; }
    bool valid() const { return id_ != 0; }

private:
    GLuint id_ = 0;
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
    void destroy();
    void attachColor(GLuint texture, GLuint attachment);
    void attachDepth(GLuint texture);
    void setDrawBuffers(const GLuint* attachments, std::size_t count);
    bool complete(std::string& error) const;
    GLuint id() const { return id_; }

private:
    GLuint id_ = 0;
};

class GlBuffer {
public:
    GlBuffer() = default;
    ~GlBuffer();
    GlBuffer(const GlBuffer&) = delete;
    GlBuffer& operator=(const GlBuffer&) = delete;
    GlBuffer(GlBuffer&& other) noexcept;
    GlBuffer& operator=(GlBuffer&& other) noexcept;

    void create(GLenum target);
    void destroy();
    void upload(const void* data, std::size_t bytes, GLenum usage);
    void bind() const;
    GLuint id() const { return id_; }
    GLenum target() const { return target_; }

private:
    GLuint id_ = 0;
    GLenum target_ = 0;
};

class GlVertexArray {
public:
    GlVertexArray() = default;
    ~GlVertexArray();
    GlVertexArray(const GlVertexArray&) = delete;
    GlVertexArray& operator=(const GlVertexArray&) = delete;
    GlVertexArray(GlVertexArray&& other) noexcept;
    GlVertexArray& operator=(GlVertexArray&& other) noexcept;

    void create();
    void destroy();
    void bind() const;
    GLuint id() const { return id_; }

private:
    GLuint id_ = 0;
};

class GlQuery {
public:
    GlQuery() = default;
    ~GlQuery();
    GlQuery(const GlQuery&) = delete;
    GlQuery& operator=(const GlQuery&) = delete;
    GlQuery(GlQuery&& other) noexcept;
    GlQuery& operator=(GlQuery&& other) noexcept;

    void create(GLenum target);
    void destroy();
    void begin() const;
    void end() const;
    std::uint64_t result64() const;
    GLuint id() const { return id_; }

private:
    GLuint id_ = 0;
    GLenum target_ = 0;
};

class GlProgram {
public:
    GlProgram() = default;
    ~GlProgram();
    GlProgram(const GlProgram&) = delete;
    GlProgram& operator=(const GlProgram&) = delete;
    GlProgram(GlProgram&& other) noexcept;
    GlProgram& operator=(GlProgram&& other) noexcept;

    bool compile(const char* vertexSource, const char* fragmentSource, std::string& error);
    void destroy();
    void use() const;
    GLint uniform(const char* name) const;
    GLuint id() const { return id_; }

private:
    GLuint id_ = 0;
};

} // namespace dr
