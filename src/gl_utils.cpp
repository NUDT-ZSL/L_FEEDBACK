#include "gl_utils.h"

#include <iostream>
#include <stdexcept>
#include <vector>

#include <glm/gtc/type_ptr.hpp>

namespace {
int gShaderProgramBuildCount = 0;
}

int shaderProgramBuildCount() {
    return gShaderProgramBuildCount;
}

ShaderProgram::~ShaderProgram() {
    destroy();
}

GLuint compileShader(GLenum type, const std::string& source, const std::string& debugName) {
    GLuint shader = glCreateShader(type);
    const char* text = source.c_str();
    glShaderSource(shader, 1, &text, nullptr);
    glCompileShader(shader);

    GLint success = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success != GL_TRUE) {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::vector<char> log(static_cast<size_t>(length > 1 ? length : 1));
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        glDeleteShader(shader);
        throw std::runtime_error("Shader compile failed (" + debugName + "):\n" + log.data());
    }
    return shader;
}

bool ShaderProgram::build(const std::string& vertexSource,
                          const std::string& fragmentSource,
                          const std::string& debugName) {
    destroy();

    GLuint vertex = compileShader(GL_VERTEX_SHADER, vertexSource, debugName + ".vert");
    GLuint fragment = compileShader(GL_FRAGMENT_SHADER, fragmentSource, debugName + ".frag");

    program_ = glCreateProgram();
    glAttachShader(program_, vertex);
    glAttachShader(program_, fragment);
    glLinkProgram(program_);
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    GLint success = GL_FALSE;
    glGetProgramiv(program_, GL_LINK_STATUS, &success);
    if (success != GL_TRUE) {
        GLint length = 0;
        glGetProgramiv(program_, GL_INFO_LOG_LENGTH, &length);
        std::vector<char> log(static_cast<size_t>(length > 1 ? length : 1));
        glGetProgramInfoLog(program_, length, nullptr, log.data());
        destroy();
        throw std::runtime_error("Program link failed (" + debugName + "):\n" + log.data());
    }

    locations_.clear();
    ++gShaderProgramBuildCount;
    std::cout << "[SHADER] compiled " << debugName << "\n";
    return true;
}

void ShaderProgram::destroy() {
    if (program_ != 0) {
        glDeleteProgram(program_);
        program_ = 0;
    }
    locations_.clear();
}

void ShaderProgram::use() const {
    glUseProgram(program_);
}

GLint ShaderProgram::location(const char* name) const {
    auto it = locations_.find(name);
    if (it != locations_.end()) {
        return it->second;
    }
    GLint loc = glGetUniformLocation(program_, name);
    locations_.emplace(name, loc);
    return loc;
}

void ShaderProgram::setInt(const char* name, int value) const {
    glUniform1i(location(name), value);
}

void ShaderProgram::setFloat(const char* name, float value) const {
    glUniform1f(location(name), value);
}

void ShaderProgram::setVec2(const char* name, const glm::vec2& value) const {
    glUniform2fv(location(name), 1, glm::value_ptr(value));
}

void ShaderProgram::setVec3(const char* name, const glm::vec3& value) const {
    glUniform3fv(location(name), 1, glm::value_ptr(value));
}

void ShaderProgram::setVec4(const char* name, const glm::vec4& value) const {
    glUniform4fv(location(name), 1, glm::value_ptr(value));
}

void ShaderProgram::setMat3(const char* name, const glm::mat3& value) const {
    glUniformMatrix3fv(location(name), 1, GL_FALSE, glm::value_ptr(value));
}

void ShaderProgram::setMat4(const char* name, const glm::mat4& value) const {
    glUniformMatrix4fv(location(name), 1, GL_FALSE, glm::value_ptr(value));
}

void ShaderProgram::setMat4Array(const char* name, const glm::mat4* values, int count) const {
    glUniformMatrix4fv(location(name), count, GL_FALSE, reinterpret_cast<const float*>(values));
}

void ShaderProgram::setIntArray(const char* name, const int* values, int count) const {
    glUniform1iv(location(name), count, values);
}

void ShaderProgram::setFloatArray(const char* name, const float* values, int count) const {
    glUniform1fv(location(name), count, values);
}

void checkGLError(const char* context, bool throwOnError) {
    GLenum error = glGetError();
    while (error != GL_NO_ERROR) {
        std::cerr << "[GL_ERROR] " << context << ": 0x" << std::hex << error << std::dec << "\n";
        if (throwOnError) {
            throw std::runtime_error(std::string("OpenGL error at ") + context);
        }
        error = glGetError();
    }
}

