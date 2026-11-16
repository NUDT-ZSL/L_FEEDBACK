#include "render/ShaderProgram.hpp"

#include <algorithm>
#include <sstream>
#include <utility>

#include "core/Error.hpp"
#include "render/ShaderSources.hpp"

namespace lcr {
namespace {

std::string shaderLog(const GLuint shader) {
  GLint length = 0;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
  if (length <= 0) {
    return "<no shader log>";
  }
  std::string log(static_cast<std::size_t>(length), '\0');
  GLsizei written = 0;
  glGetShaderInfoLog(shader, length, &written, log.data());
  log.resize(static_cast<std::size_t>(std::max(0, written)));
  return log;
}

std::string programLog(const GLuint program) {
  GLint length = 0;
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
  if (length <= 0) {
    return "<no program log>";
  }
  std::string log(static_cast<std::size_t>(length), '\0');
  GLsizei written = 0;
  glGetProgramInfoLog(program, length, &written, log.data());
  log.resize(static_cast<std::size_t>(std::max(0, written)));
  return log;
}

}  // namespace

ShaderProgram::ShaderProgram(
    const std::string& vertexSource,
    const std::string& fragmentSource,
    const bool forceCompileFailure)
    : program_(glCreateProgram()) {
  if (program_ == 0U) {
    fail(ErrorCode::GpuInit, "shader.createProgram", "glCreateProgram returned zero");
  }
  const std::string vertex =
      forceCompileFailure ? injectShaderFailure(vertexSource) : vertexSource;
  const GLuint vertexShader = compile(
      GL_VERTEX_SHADER, vertex.c_str(), "character.vertex");
  GLuint fragmentShader = 0;
  try {
    const std::string fragment =
        forceCompileFailure ? injectShaderFailure(fragmentSource) : fragmentSource;
    fragmentShader = compile(
        GL_FRAGMENT_SHADER, fragment.c_str(), "character.fragment");
  } catch (...) {
    glDeleteShader(vertexShader);
    glDeleteProgram(program_);
    program_ = 0;
    throw;
  }

  glAttachShader(program_, vertexShader);
  glAttachShader(program_, fragmentShader);
  glLinkProgram(program_);
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  GLint linked = GL_FALSE;
  glGetProgramiv(program_, GL_LINK_STATUS, &linked);
  if (linked != GL_TRUE) {
    const std::string log = programLog(program_);
    glDeleteProgram(program_);
    program_ = 0;
    fail(ErrorCode::ShaderLink, "shader.link", "program link failed:\n" + log);
  }
}

ShaderProgram::~ShaderProgram() {
  if (program_ != 0U) {
    glDeleteProgram(program_);
  }
}

GLuint ShaderProgram::compile(
    const GLenum type,
    const char* source,
    const char* stage) {
  const GLuint shader = glCreateShader(type);
  if (shader == 0U) {
    fail(ErrorCode::GpuInit, std::string("shader.create.") + stage, "glCreateShader failed");
  }
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  GLint compiled = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (compiled != GL_TRUE) {
    const std::string log = shaderLog(shader);
    glDeleteShader(shader);
    fail(
        ErrorCode::ShaderCompile,
        std::string("shader.compile.") + stage,
        "shader compilation failed:\n" + log);
  }
  return shader;
}

GLint ShaderProgram::uniform(const std::string& name) {
  const auto found = uniformCache_.find(name);
  if (found != uniformCache_.end()) {
    return found->second;
  }
  const GLint location = glGetUniformLocation(program_, name.c_str());
  if (location < 0) {
    fail(
        ErrorCode::InternalInvariant,
        "shader.uniform",
        "required uniform is missing or optimized out: " + name);
  }
  uniformCache_.emplace(name, location);
  return location;
}

void ShaderProgram::setInt(const std::string& name, const int value) {
  glUniform1i(uniform(name), value);
}

void ShaderProgram::setFloat(const std::string& name, const float value) {
  glUniform1f(uniform(name), value);
}

void ShaderProgram::setVec3(
    const std::string& name,
    const float x,
    const float y,
    const float z) {
  glUniform3f(uniform(name), x, y, z);
}

void ShaderProgram::setVec4(
    const std::string& name,
    const float x,
    const float y,
    const float z,
    const float w) {
  glUniform4f(uniform(name), x, y, z, w);
}

void ShaderProgram::setMat4(const std::string& name, const float* values) {
  glUniformMatrix4fv(uniform(name), 1, GL_FALSE, values);
}

}  // namespace lcr

