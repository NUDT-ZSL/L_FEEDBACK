#include "indoor/ShaderProgram.hpp"

#include "indoor/GLUtil.hpp"

#include <glad/gl.h>

#include <fstream>
#include <algorithm>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace indoor {
namespace {

std::string readFile(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Unable to open shader file: " + path);
    }
    std::ostringstream stream;
    stream << input.rdbuf();
    return stream.str();
}

std::string expandIncludes(const std::string& source, const std::string& directory,
                           int depth) {
    if (depth > 8) {
        throw std::runtime_error("Shader include depth exceeded");
    }
    const std::regex includePattern(
        "^\\s*#include\\s+\"([^\"]+)\"\\s*$");
    std::istringstream input(source);
    std::ostringstream output;
    std::string line;
    while (std::getline(input, line)) {
        std::smatch match;
        if (std::regex_match(line, match, includePattern)) {
            const std::string includePath = directory + "/" + match[1].str();
            output << expandIncludes(readFile(includePath), directory, depth + 1)
                   << '\n';
        } else {
            output << line << '\n';
        }
    }
    return output.str();
}

unsigned int compile(GLenum type, const std::string& source,
                     const std::string& label) {
    const unsigned int shader = glCreateShader(type);
    const char* sourcePointer = source.c_str();
    glShaderSource(shader, 1, &sourcePointer, nullptr);
    glCompileShader(shader);
    int success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE) {
        int logLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        std::vector<char> log(static_cast<std::size_t>(std::max(logLength, 1)));
        glGetShaderInfoLog(shader, logLength, nullptr, log.data());
        glDeleteShader(shader);
        throw std::runtime_error("Shader compile failed (" + label + "):\n" +
                                 std::string(log.data(), log.size()));
    }
    return shader;
}

} // namespace

ShaderProgram::~ShaderProgram() {
    if (program_ != 0) {
        glDeleteProgram(program_);
    }
}

void ShaderProgram::loadFromFiles(const std::string& vertexPath,
                                  const std::string& fragmentPath,
                                  const std::string& includeDirectory) {
    RenderThreadGuard::requireRenderThread("ShaderProgram::loadFromFiles");
    const std::string vertexSource =
        expandIncludes(readFile(vertexPath), includeDirectory, 0);
    const std::string fragmentSource =
        expandIncludes(readFile(fragmentPath), includeDirectory, 0);

    const unsigned int vertex = compile(GL_VERTEX_SHADER, vertexSource, vertexPath);
    unsigned int fragment = 0;
    try {
        fragment = compile(GL_FRAGMENT_SHADER, fragmentSource, fragmentPath);
    } catch (...) {
        glDeleteShader(vertex);
        throw;
    }

    const unsigned int program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    int success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success == GL_FALSE) {
        int logLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
        std::vector<char> log(static_cast<std::size_t>(std::max(logLength, 1)));
        glGetProgramInfoLog(program, logLength, nullptr, log.data());
        glDeleteProgram(program);
        throw std::runtime_error("Shader link failed:\n" +
                                 std::string(log.data(), log.size()));
    }

    if (program_ != 0) {
        glDeleteProgram(program_);
    }
    program_ = program;
    uniformCache_.clear();
}

void ShaderProgram::use() const {
    RenderThreadGuard::requireRenderThread("ShaderProgram::use");
    glUseProgram(program_);
}

void ShaderProgram::abandon() {
    program_ = 0;
    uniformCache_.clear();
}

int ShaderProgram::uniformLocation(const char* name) {
    const auto found = uniformCache_.find(name);
    if (found != uniformCache_.end()) {
        return found->second;
    }
    const int location = glGetUniformLocation(program_, name);
    uniformCache_.emplace(name, location);
    return location;
}

void ShaderProgram::setInt(const char* name, int value) {
    glUniform1i(uniformLocation(name), value);
}

void ShaderProgram::setFloat(const char* name, float value) {
    glUniform1f(uniformLocation(name), value);
}

void ShaderProgram::setVec2(const char* name, const glm::vec2& value) {
    glUniform2fv(uniformLocation(name), 1, &value.x);
}

void ShaderProgram::setVec3(const char* name, const glm::vec3& value) {
    glUniform3fv(uniformLocation(name), 1, &value.x);
}

void ShaderProgram::setVec4(const char* name, const glm::vec4& value) {
    glUniform4fv(uniformLocation(name), 1, &value.x);
}

void ShaderProgram::setMat4(const char* name, const glm::mat4& value) {
    glUniformMatrix4fv(uniformLocation(name), 1, GL_FALSE, &value[0][0]);
}

} // namespace indoor
