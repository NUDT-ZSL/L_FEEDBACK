#pragma once

#include <glad/gl.h>

#include <array>
#include <string>
#include <unordered_map>

#include <glm/glm.hpp>

class ShaderProgram {
public:
    ShaderProgram() = default;
    ~ShaderProgram();

    ShaderProgram(const ShaderProgram&) = delete;
    ShaderProgram& operator=(const ShaderProgram&) = delete;

    bool build(const std::string& vertexSource,
               const std::string& fragmentSource,
               const std::string& debugName);
    void destroy();
    void use() const;

    GLuint id() const { return program_; }

    void setInt(const char* name, int value) const;
    void setFloat(const char* name, float value) const;
    void setVec2(const char* name, const glm::vec2& value) const;
    void setVec3(const char* name, const glm::vec3& value) const;
    void setVec4(const char* name, const glm::vec4& value) const;
    void setMat3(const char* name, const glm::mat3& value) const;
    void setMat4(const char* name, const glm::mat4& value) const;
    void setMat4Array(const char* name, const glm::mat4* values, int count) const;
    void setIntArray(const char* name, const int* values, int count) const;
    void setFloatArray(const char* name, const float* values, int count) const;

private:
    GLint location(const char* name) const;

    GLuint program_ = 0;
    mutable std::unordered_map<std::string, GLint> locations_;
};

GLuint compileShader(GLenum type, const std::string& source, const std::string& debugName);
int shaderProgramBuildCount();

void checkGLError(const char* context, bool throwOnError = true);

