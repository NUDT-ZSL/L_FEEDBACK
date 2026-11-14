#pragma once

#include <glm/glm.hpp>
#include <string>
#include <unordered_map>

namespace indoor {

class ShaderProgram {
public:
    ShaderProgram() = default;
    ~ShaderProgram();

    ShaderProgram(const ShaderProgram&) = delete;
    ShaderProgram& operator=(const ShaderProgram&) = delete;

    void loadFromFiles(const std::string& vertexPath,
                       const std::string& fragmentPath,
                       const std::string& includeDirectory);
    void use() const;
    void abandon();

    [[nodiscard]] unsigned int id() const { return program_; }
    [[nodiscard]] bool valid() const { return program_ != 0; }

    void setInt(const char* name, int value);
    void setFloat(const char* name, float value);
    void setVec2(const char* name, const glm::vec2& value);
    void setVec3(const char* name, const glm::vec3& value);
    void setVec4(const char* name, const glm::vec4& value);
    void setMat4(const char* name, const glm::mat4& value);

private:
    int uniformLocation(const char* name);
    unsigned int program_ = 0;
    std::unordered_map<std::string, int> uniformCache_;
};

} // namespace indoor
