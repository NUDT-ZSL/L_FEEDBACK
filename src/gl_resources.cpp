#include "gl_resources.h"

#include <algorithm>
#include <iostream>
#include <vector>

namespace dr {

GlTexture::~GlTexture() {
    destroy();
}

GlTexture::GlTexture(GlTexture&& other) noexcept : id_(other.id_) {
    other.id_ = 0;
}

GlTexture& GlTexture::operator=(GlTexture&& other) noexcept {
    if (this != &other) {
        destroy();
        id_ = other.id_;
        other.id_ = 0;
    }
    return *this;
}

void GlTexture::create2D(int width, int height, GLint internalFormat, GLenum format, GLenum type,
                         const void* pixels, GLint filter) {
    destroy();
    glGenTextures(1, &id_);
    glBindTexture(GL_TEXTURE_2D, id_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, pixels);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void GlTexture::destroy() {
    if (id_ != 0) {
        glDeleteTextures(1, &id_);
        id_ = 0;
    }
}

GlFramebuffer::~GlFramebuffer() {
    destroy();
}

GlFramebuffer::GlFramebuffer(GlFramebuffer&& other) noexcept : id_(other.id_) {
    other.id_ = 0;
}

GlFramebuffer& GlFramebuffer::operator=(GlFramebuffer&& other) noexcept {
    if (this != &other) {
        destroy();
        id_ = other.id_;
        other.id_ = 0;
    }
    return *this;
}

void GlFramebuffer::create() {
    destroy();
    glGenFramebuffers(1, &id_);
}

void GlFramebuffer::destroy() {
    if (id_ != 0) {
        glDeleteFramebuffers(1, &id_);
        id_ = 0;
    }
}

void GlFramebuffer::attachColor(GLuint texture, GLuint attachment) {
    glBindFramebuffer(GL_FRAMEBUFFER, id_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture, 0);
}

void GlFramebuffer::attachDepth(GLuint texture) {
    glBindFramebuffer(GL_FRAMEBUFFER, id_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture, 0);
}

void GlFramebuffer::setDrawBuffers(const GLuint* attachments, std::size_t count) {
    glBindFramebuffer(GL_FRAMEBUFFER, id_);
    glDrawBuffers(static_cast<GLsizei>(count), attachments);
}

bool GlFramebuffer::complete(std::string& error) const {
    glBindFramebuffer(GL_FRAMEBUFFER, id_);
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status == GL_FRAMEBUFFER_COMPLETE) {
        return true;
    }
    error = "framebuffer status 0x" + std::to_string(static_cast<unsigned int>(status));
    return false;
}

GlBuffer::~GlBuffer() {
    destroy();
}

GlBuffer::GlBuffer(GlBuffer&& other) noexcept : id_(other.id_), target_(other.target_) {
    other.id_ = 0;
    other.target_ = 0;
}

GlBuffer& GlBuffer::operator=(GlBuffer&& other) noexcept {
    if (this != &other) {
        destroy();
        id_ = other.id_;
        target_ = other.target_;
        other.id_ = 0;
        other.target_ = 0;
    }
    return *this;
}

void GlBuffer::create(GLenum target) {
    destroy();
    target_ = target;
    glGenBuffers(1, &id_);
}

void GlBuffer::destroy() {
    if (id_ != 0) {
        glDeleteBuffers(1, &id_);
        id_ = 0;
        target_ = 0;
    }
}

void GlBuffer::upload(const void* data, std::size_t bytes, GLenum usage) {
    glBindBuffer(target_, id_);
    glBufferData(target_, static_cast<GLsizeiptr>(bytes), data, usage);
}

void GlBuffer::bind() const {
    glBindBuffer(target_, id_);
}

GlVertexArray::~GlVertexArray() {
    destroy();
}

GlVertexArray::GlVertexArray(GlVertexArray&& other) noexcept : id_(other.id_) {
    other.id_ = 0;
}

GlVertexArray& GlVertexArray::operator=(GlVertexArray&& other) noexcept {
    if (this != &other) {
        destroy();
        id_ = other.id_;
        other.id_ = 0;
    }
    return *this;
}

void GlVertexArray::create() {
    destroy();
    glGenVertexArrays(1, &id_);
}

void GlVertexArray::destroy() {
    if (id_ != 0) {
        glDeleteVertexArrays(1, &id_);
        id_ = 0;
    }
}

void GlVertexArray::bind() const {
    glBindVertexArray(id_);
}

GlQuery::~GlQuery() {
    destroy();
}

GlQuery::GlQuery(GlQuery&& other) noexcept : id_(other.id_), target_(other.target_) {
    other.id_ = 0;
    other.target_ = 0;
}

GlQuery& GlQuery::operator=(GlQuery&& other) noexcept {
    if (this != &other) {
        destroy();
        id_ = other.id_;
        target_ = other.target_;
        other.id_ = 0;
        other.target_ = 0;
    }
    return *this;
}

void GlQuery::create(GLenum target) {
    destroy();
    target_ = target;
    glGenQueries(1, &id_);
}

void GlQuery::destroy() {
    if (id_ != 0) {
        glDeleteQueries(1, &id_);
        id_ = 0;
    }
}

void GlQuery::begin() const {
    glBeginQuery(target_, id_);
}

void GlQuery::end() const {
    glEndQuery(target_);
}

std::uint64_t GlQuery::result64() const {
    GLuint64 result = 0;
    glGetQueryObjectui64v(id_, GL_QUERY_RESULT, &result);
    return static_cast<std::uint64_t>(result);
}

GlProgram::~GlProgram() {
    destroy();
}

GlProgram::GlProgram(GlProgram&& other) noexcept : id_(other.id_) {
    other.id_ = 0;
}

GlProgram& GlProgram::operator=(GlProgram&& other) noexcept {
    if (this != &other) {
        destroy();
        id_ = other.id_;
        other.id_ = 0;
    }
    return *this;
}

bool GlProgram::compile(const char* vertexSource, const char* fragmentSource, std::string& error) {
    destroy();

    const GLuint vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vertexSource, nullptr);
    glCompileShader(vertex);

    GLint successful = GL_FALSE;
    glGetShaderiv(vertex, GL_COMPILE_STATUS, &successful);
    if (successful != GL_TRUE) {
        GLint length = 0;
        glGetShaderiv(vertex, GL_INFO_LOG_LENGTH, &length);
        std::vector<char> log(static_cast<std::size_t>(std::max(length, 1)));
        glGetShaderInfoLog(vertex, length, nullptr, log.data());
        error = "vertex shader: " + std::string(log.data());
        glDeleteShader(vertex);
        return false;
    }

    const GLuint fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fragmentSource, nullptr);
    glCompileShader(fragment);
    glGetShaderiv(fragment, GL_COMPILE_STATUS, &successful);
    if (successful != GL_TRUE) {
        GLint length = 0;
        glGetShaderiv(fragment, GL_INFO_LOG_LENGTH, &length);
        std::vector<char> log(static_cast<std::size_t>(std::max(length, 1)));
        glGetShaderInfoLog(fragment, length, nullptr, log.data());
        error = "fragment shader: " + std::string(log.data());
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        return false;
    }

    id_ = glCreateProgram();
    glAttachShader(id_, vertex);
    glAttachShader(id_, fragment);
    glLinkProgram(id_);
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    glGetProgramiv(id_, GL_LINK_STATUS, &successful);
    if (successful != GL_TRUE) {
        GLint length = 0;
        glGetProgramiv(id_, GL_INFO_LOG_LENGTH, &length);
        std::vector<char> log(static_cast<std::size_t>(std::max(length, 1)));
        glGetProgramInfoLog(id_, length, nullptr, log.data());
        error = "program link: " + std::string(log.data());
        destroy();
        return false;
    }
    return true;
}

void GlProgram::destroy() {
    if (id_ != 0) {
        glDeleteProgram(id_);
        id_ = 0;
    }
}

void GlProgram::use() const {
    glUseProgram(id_);
}

GLint GlProgram::uniform(const char* name) const {
    return glGetUniformLocation(id_, name);
}

} // namespace dr
