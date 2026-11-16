#pragma once

#include <glad/glad.h>

#include <string>
#include <unordered_map>

namespace lcr {

class ShaderProgram {
 public:
  ShaderProgram(
      const std::string& vertexSource,
      const std::string& fragmentSource,
      bool forceCompileFailure);
  ~ShaderProgram();

  ShaderProgram(const ShaderProgram&) = delete;
  ShaderProgram& operator=(const ShaderProgram&) = delete;

  GLuint id() const noexcept { return program_; }
  GLint uniform(const std::string& name);

  void setInt(const std::string& name, int value);
  void setFloat(const std::string& name, float value);
  void setVec3(const std::string& name, float x, float y, float z);
  void setVec4(const std::string& name, float x, float y, float z, float w);
  void setMat4(const std::string& name, const float* values);

 private:
  GLuint compile(GLenum type, const char* source, const char* stage);
  GLuint program_ = 0;
  std::unordered_map<std::string, GLint> uniformCache_;
};

}  // namespace lcr

