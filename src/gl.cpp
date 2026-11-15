#include "gl.hpp"

#include <fstream>
#include <sstream>

namespace scene {
namespace {

std::string shaderLog(GLuint object, bool program) {
    GLint length = 0;
    if (program) {
        glGetProgramiv(object, GL_INFO_LOG_LENGTH, &length);
    } else {
        glGetShaderiv(object, GL_INFO_LOG_LENGTH, &length);
    }
    if (length <= 0) {
        return "<no shader log>";
    }
    std::string log(static_cast<size_t>(length), '\0');
    if (program) {
        glGetProgramInfoLog(object, length, nullptr, log.data());
    } else {
        glGetShaderInfoLog(object, length, nullptr, log.data());
    }
    while (!log.empty() && (log.back() == '\0' || log.back() == '\n')) {
        log.pop_back();
    }
    return log.empty() ? "<empty shader log>" : log;
}

GLuint compileStage(GLenum stage, const std::string& source,
                    const std::string& label) {
    const GLuint shader = glCreateShader(stage);
    const char* sourcePointer = source.c_str();
    glShaderSource(shader, 1, &sourcePointer, nullptr);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled != GL_TRUE) {
        const std::string log = shaderLog(shader, false);
        glDeleteShader(shader);
        throw AppError(ErrorType::ShaderCompile,
                       label + " compilation failed:\n" + log);
    }
    return shader;
}

} // namespace

GlTexture::~GlTexture() {
    release();
}

GlTexture::GlTexture(GlTexture&& other) noexcept
    : id_(other.id_), width_(other.width_), height_(other.height_),
      label_(std::move(other.label_)) {
    other.id_ = 0;
    other.width_ = 0;
    other.height_ = 0;
}

GlTexture& GlTexture::operator=(GlTexture&& other) noexcept {
    if (this != &other) {
        release();
        id_ = other.id_;
        width_ = other.width_;
        height_ = other.height_;
        label_ = std::move(other.label_);
        other.id_ = 0;
        other.width_ = 0;
        other.height_ = 0;
    }
    return *this;
}

void GlTexture::create2D(int width, int height, GLenum internalFormat,
                         GLenum format, GLenum type, GLenum minFilter,
                         GLenum magFilter, GLenum wrapS, GLenum wrapT,
                         const void* pixels) {
    release();
    width_ = width;
    height_ = height;
    glGenTextures(1, &id_);
    if (id_ == 0) {
        throw AppError(ErrorType::GlRuntime, "glGenTextures returned 0");
    }
    glBindTexture(GL_TEXTURE_2D, id_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    static_cast<GLint>(minFilter));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    static_cast<GLint>(magFilter));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, static_cast<GLint>(wrapS));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, static_cast<GLint>(wrapT));
    glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(internalFormat),
                 width, height, 0, format, type, pixels);
    glBindTexture(GL_TEXTURE_2D, 0);
    checkGlError("creating texture " + label_);
}

void GlTexture::release() {
    if (id_ != 0) {
        glDeleteTextures(1, &id_);
        id_ = 0;
    }
    width_ = 0;
    height_ = 0;
}

GlFramebuffer::~GlFramebuffer() {
    release();
}

GlFramebuffer::GlFramebuffer(GlFramebuffer&& other) noexcept
    : id_(other.id_), label_(std::move(other.label_)) {
    other.id_ = 0;
}

GlFramebuffer& GlFramebuffer::operator=(GlFramebuffer&& other) noexcept {
    if (this != &other) {
        release();
        id_ = other.id_;
        label_ = std::move(other.label_);
        other.id_ = 0;
    }
    return *this;
}

void GlFramebuffer::create() {
    release();
    glGenFramebuffers(1, &id_);
    if (id_ == 0) {
        throw AppError(ErrorType::GlRuntime, "glGenFramebuffers returned 0");
    }
}

void GlFramebuffer::attachColor(size_t index, const GlTexture& texture) {
    glBindFramebuffer(GL_FRAMEBUFFER, id_);
    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           static_cast<GLenum>(GL_COLOR_ATTACHMENT0 + index),
                           GL_TEXTURE_2D, texture.id(), 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GlFramebuffer::attachDepth(const GlTexture& texture) {
    glBindFramebuffer(GL_FRAMEBUFFER, id_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                           GL_TEXTURE_2D, texture.id(), 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GlFramebuffer::checkComplete(const std::string& label) const {
    glBindFramebuffer(GL_FRAMEBUFFER, id_);
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::ostringstream message;
        message << label << " framebuffer is incomplete (status 0x"
                << std::hex << status << ")";
        throw AppError(ErrorType::FramebufferIncomplete, message.str());
    }
}

void GlFramebuffer::release() {
    if (id_ != 0) {
        glDeleteFramebuffers(1, &id_);
        id_ = 0;
    }
}

ShaderProgram::~ShaderProgram() {
    release();
}

ShaderProgram::ShaderProgram(ShaderProgram&& other) noexcept
    : id_(other.id_), label_(std::move(other.label_)),
      uniforms_(std::move(other.uniforms_)) {
    other.id_ = 0;
}

ShaderProgram& ShaderProgram::operator=(ShaderProgram&& other) noexcept {
    if (this != &other) {
        release();
        id_ = other.id_;
        label_ = std::move(other.label_);
        uniforms_ = std::move(other.uniforms_);
        other.id_ = 0;
    }
    return *this;
}

ShaderProgram ShaderProgram::fromFiles(const std::string& vertexPath,
                                       const std::string& fragmentPath) {
    return fromSources(readTextFile(vertexPath), readTextFile(fragmentPath),
                       vertexPath + " + " + fragmentPath);
}

ShaderProgram ShaderProgram::fromSources(const std::string& vertexSource,
                                         const std::string& fragmentSource,
                                         const std::string& label) {
    ShaderProgram program;
    program.label_ = label;
    const GLuint vertex = compileStage(GL_VERTEX_SHADER, vertexSource,
                                       label + " (vertex)");
    GLuint fragment = 0;
    try {
        fragment = compileStage(GL_FRAGMENT_SHADER, fragmentSource,
                                label + " (fragment)");
    } catch (...) {
        glDeleteShader(vertex);
        throw;
    }

    program.id_ = glCreateProgram();
    if (program.id_ == 0) {
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        throw AppError(ErrorType::ShaderCompile,
                       label + ": glCreateProgram returned 0");
    }
    glAttachShader(program.id_, vertex);
    glAttachShader(program.id_, fragment);
    glLinkProgram(program.id_);
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    GLint linked = GL_FALSE;
    glGetProgramiv(program.id_, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE) {
        const std::string log = shaderLog(program.id_, true);
        program.release();
        throw AppError(ErrorType::ShaderCompile,
                       label + " link failed:\n" + log);
    }
    checkGlError("linking shader program " + label);
    return program;
}

void ShaderProgram::use() const {
    glUseProgram(id_);
}

GLint ShaderProgram::location(const char* name) {
    const auto found = uniforms_.find(name);
    if (found != uniforms_.end()) {
        return found->second;
    }
    const GLint result = glGetUniformLocation(id_, name);
    uniforms_.emplace(name, result);
    return result;
}

void ShaderProgram::setInt(const char* name, int value) {
    const GLint loc = location(name);
    if (loc >= 0) glUniform1i(loc, value);
}

void ShaderProgram::setFloat(const char* name, float value) {
    const GLint loc = location(name);
    if (loc >= 0) glUniform1f(loc, value);
}

void ShaderProgram::setVec2(const char* name, float x, float y) {
    const GLint loc = location(name);
    if (loc >= 0) glUniform2f(loc, x, y);
}

void ShaderProgram::setVec3(const char* name, Vec3 value) {
    const GLint loc = location(name);
    if (loc >= 0) glUniform3f(loc, value.x, value.y, value.z);
}

void ShaderProgram::setMat3(const char* name, const Mat3& value) {
    const GLint loc = location(name);
    if (loc >= 0) glUniformMatrix3fv(loc, 1, GL_FALSE, value.m.data());
}

void ShaderProgram::setMat4(const char* name, const Mat4& value) {
    const GLint loc = location(name);
    if (loc >= 0) glUniformMatrix4fv(loc, 1, GL_FALSE, value.m.data());
}

void ShaderProgram::setIntArray(const char* name, const std::vector<int>& values) {
    const GLint loc = location(name);
    if (loc >= 0 && !values.empty()) {
        glUniform1iv(loc, static_cast<GLsizei>(values.size()), values.data());
    }
}

void ShaderProgram::setVec2Array(const char* name,
                                 const std::vector<float>& values) {
    const GLint loc = location(name);
    if (loc >= 0 && !values.empty()) {
        glUniform2fv(loc, static_cast<GLsizei>(values.size() / 2), values.data());
    }
}

void ShaderProgram::setVec3Array(const char* name,
                                 const std::vector<Vec3>& values) {
    const GLint loc = location(name);
    if (loc >= 0 && !values.empty()) {
        glUniform3fv(loc, static_cast<GLsizei>(values.size()),
                     reinterpret_cast<const float*>(values.data()));
    }
}

void ShaderProgram::setFloatArray(const char* name,
                                  const std::vector<float>& values) {
    const GLint loc = location(name);
    if (loc >= 0 && !values.empty()) {
        glUniform1fv(loc, static_cast<GLsizei>(values.size()), values.data());
    }
}

void ShaderProgram::release() {
    if (id_ != 0) {
        glDeleteProgram(id_);
        id_ = 0;
    }
    uniforms_.clear();
}

std::string readTextFile(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw AppError(ErrorType::FileIo, "cannot open shader file '" + path + "'");
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw AppError(ErrorType::FileIo,
                       "failed while reading shader file '" + path + "'");
    }
    return buffer.str();
}

void checkGlError(const std::string& context) {
    const GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::ostringstream message;
        message << "OpenGL error 0x" << std::hex << error
                << " while " << context;
        throw AppError(ErrorType::GlRuntime, message.str());
    }
}

void drawFullscreenTriangle(GLuint vao) {
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}

void bindTexture(GLuint unit, GLuint texture) {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, texture);
}

} // namespace scene

