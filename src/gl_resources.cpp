#include "gl_resources.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace skin {
namespace {

GLuint compileShader(
    GLenum type,
    const char* source,
    ResourceCounters& counters
) {
    const GLuint shader = glCreateShader(type);
    if (shader == 0) {
        throw std::runtime_error("glCreateShader returned 0");
    }
    ++counters.shaderObjectsCreated;

    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled != GL_TRUE) {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        glDeleteShader(shader);
        ++counters.shaderObjectsReleased;
        throw std::runtime_error("shader compilation failed: " + log);
    }
    return shader;
}

void deleteShaderObject(GLuint shader, ResourceCounters& counters) {
    if (shader != 0) {
        glDeleteShader(shader);
        ++counters.shaderObjectsReleased;
    }
}

std::string targetKey(
    int width,
    int height,
    GLenum format,
    bool withDepth
) {
    std::ostringstream key;
    key << width << 'x' << height << ':' << static_cast<unsigned int>(format)
        << ':' << (withDepth ? "depth" : "nodepth");
    return key.str();
}

}  // namespace

void ResourceCounters::logSummary(
    Logger& logger,
    const std::string& group
) const {
    auto line = [&logger, &group](
                    const std::string& label,
                    int created,
                    int released
                ) {
        logger.debug(
            group,
            "resource " + label
                + " created=" + std::to_string(created)
                + " released=" + std::to_string(released)
                + " balanced=" + (created == released ? "yes" : "no")
        );
    };
    line("framebuffers", framebuffersCreated, framebuffersReleased);
    line("textures", texturesCreated, texturesReleased);
    line("renderbuffers", renderbuffersCreated, renderbuffersReleased);
    line("programs", programsCreated, programsReleased);
    line("shader_objects", shaderObjectsCreated, shaderObjectsReleased);
    line("vertex_arrays", vertexArraysCreated, vertexArraysReleased);
    line("buffers", buffersCreated, buffersReleased);
}

FramebufferCache::FramebufferCache(ResourceCounters& counters)
    : counters_(counters) {}

FramebufferCache::~FramebufferCache() {
    releaseAll();
}

FramebufferTarget& FramebufferCache::acquire(
    int width,
    int height,
    GLenum colorFormat,
    bool withDepth
) {
    const std::string key = targetKey(width, height, colorFormat, withDepth);
    const auto existing = targets_.find(key);
    if (existing != targets_.end()) {
        return existing->second;
    }

    FramebufferTarget target;
    target.width = width;
    target.height = height;
    target.colorFormat = colorFormat;
    target.depth = withDepth;

    try {
        glGenTextures(1, &target.colorTexture);
        if (target.colorTexture == 0) {
            throw std::runtime_error("glGenTextures returned 0 for framebuffer");
        }
        ++counters_.texturesCreated;
        glBindTexture(GL_TEXTURE_2D, target.colorTexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        const GLenum externalFormat = (colorFormat == GL_RGB8)
            ? GL_RGB
            : GL_RGBA;
        const GLenum type = (colorFormat == GL_RGBA16F)
            ? GL_FLOAT
            : GL_UNSIGNED_BYTE;
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            static_cast<GLint>(colorFormat),
            width,
            height,
            0,
            externalFormat,
            type,
            nullptr
        );

        glGenFramebuffers(1, &target.framebuffer);
        if (target.framebuffer == 0) {
            throw std::runtime_error("glGenFramebuffers returned 0");
        }
        ++counters_.framebuffersCreated;
        glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            target.colorTexture,
            0
        );

        if (withDepth) {
            glGenRenderbuffers(1, &target.depthRenderbuffer);
            if (target.depthRenderbuffer == 0) {
                throw std::runtime_error("glGenRenderbuffers returned 0");
            }
            ++counters_.renderbuffersCreated;
            glBindRenderbuffer(GL_RENDERBUFFER, target.depthRenderbuffer);
            glRenderbufferStorage(
                GL_RENDERBUFFER,
                GL_DEPTH_COMPONENT24,
                width,
                height
            );
            glFramebufferRenderbuffer(
                GL_FRAMEBUFFER,
                GL_DEPTH_ATTACHMENT,
                GL_RENDERBUFFER,
                target.depthRenderbuffer
            );
        }

        const GLenum drawBuffers[] = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, drawBuffers);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error(
                "incomplete framebuffer, status=" + glErrorName(status)
            );
        }
    } catch (...) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        if (target.depthRenderbuffer != 0) {
            glDeleteRenderbuffers(1, &target.depthRenderbuffer);
            ++counters_.renderbuffersReleased;
        }
        if (target.framebuffer != 0) {
            glDeleteFramebuffers(1, &target.framebuffer);
            ++counters_.framebuffersReleased;
        }
        if (target.colorTexture != 0) {
            glDeleteTextures(1, &target.colorTexture);
            ++counters_.texturesReleased;
        }
        throw;
    }

    targets_.emplace(key, target);
    return targets_.at(key);
}

void FramebufferCache::releaseAll() {
    for (auto& [key, target] : targets_) {
        (void)key;
        if (target.depthRenderbuffer != 0) {
            glDeleteRenderbuffers(1, &target.depthRenderbuffer);
            ++counters_.renderbuffersReleased;
            target.depthRenderbuffer = 0;
        }
        if (target.framebuffer != 0) {
            glDeleteFramebuffers(1, &target.framebuffer);
            ++counters_.framebuffersReleased;
            target.framebuffer = 0;
        }
        if (target.colorTexture != 0) {
            glDeleteTextures(1, &target.colorTexture);
            ++counters_.texturesReleased;
            target.colorTexture = 0;
        }
    }
    targets_.clear();
}

GLint Program::uniform(const std::string& name) {
    const auto found = uniformLocations_.find(name);
    if (found != uniformLocations_.end()) {
        return found->second;
    }
    const GLint location = glGetUniformLocation(program_, name.c_str());
    uniformLocations_.emplace(name, location);
    return location;
}

void Program::set(GLint location, float value) const {
    if (location >= 0) {
        glUniform1f(location, value);
    }
}

void Program::set(GLint location, int value) const {
    if (location >= 0) {
        glUniform1i(location, value);
    }
}

void Program::set(GLint location, const glm::vec2& value) const {
    if (location >= 0) {
        glUniform2f(location, value.x, value.y);
    }
}

void Program::set(GLint location, const glm::ivec2& value) const {
    if (location >= 0) {
        glUniform2i(location, value.x, value.y);
    }
}

void Program::set(GLint location, const glm::vec3& value) const {
    if (location >= 0) {
        glUniform3f(location, value.x, value.y, value.z);
    }
}

void Program::set(GLint location, const glm::mat3& value) const {
    if (location >= 0) {
        glUniformMatrix3fv(location, 1, GL_FALSE, &value[0][0]);
    }
}

void Program::set(GLint location, const glm::mat4& value) const {
    if (location >= 0) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &value[0][0]);
    }
}

ProgramCache::ProgramCache(ResourceCounters& counters)
    : counters_(counters) {}

ProgramCache::~ProgramCache() {
    releaseAll();
}

Program& ProgramCache::get(
    const std::string& key,
    const char* vertexSource,
    const char* fragmentSource
) {
    const auto existing = programs_.find(key);
    if (existing != programs_.end()) {
        return *existing->second;
    }

    GLuint vertexShader = 0;
    GLuint fragmentShader = 0;
    GLuint programObject = 0;
    try {
        vertexShader = compileShader(
            GL_VERTEX_SHADER,
            vertexSource,
            counters_
        );
        fragmentShader = compileShader(
            GL_FRAGMENT_SHADER,
            fragmentSource,
            counters_
        );
        programObject = glCreateProgram();
        if (programObject == 0) {
            throw std::runtime_error("glCreateProgram returned 0");
        }
        ++counters_.programsCreated;
        glAttachShader(programObject, vertexShader);
        glAttachShader(programObject, fragmentShader);
        glLinkProgram(programObject);

        GLint linked = GL_FALSE;
        glGetProgramiv(programObject, GL_LINK_STATUS, &linked);
        if (linked != GL_TRUE) {
            GLint length = 0;
            glGetProgramiv(programObject, GL_INFO_LOG_LENGTH, &length);
            std::string log(
                static_cast<std::size_t>(std::max(length, 1)),
                '\0'
            );
            glGetProgramInfoLog(programObject, length, nullptr, log.data());
            throw std::runtime_error("program link failed: " + log);
        }
    } catch (...) {
        deleteShaderObject(vertexShader, counters_);
        deleteShaderObject(fragmentShader, counters_);
        if (programObject != 0) {
            glDeleteProgram(programObject);
            ++counters_.programsReleased;
        }
        throw;
    }

    glDetachShader(programObject, vertexShader);
    glDetachShader(programObject, fragmentShader);
    deleteShaderObject(vertexShader, counters_);
    deleteShaderObject(fragmentShader, counters_);

    auto program = std::make_unique<Program>();
    program->program_ = programObject;
    Program& reference = *program;
    programs_.emplace(key, std::move(program));
    return reference;
}

void ProgramCache::releaseAll() {
    for (auto& [key, program] : programs_) {
        (void)key;
        if (program->program_ != 0) {
            glDeleteProgram(program->program_);
            ++counters_.programsReleased;
            program->program_ = 0;
        }
    }
    programs_.clear();
}

SphereMesh::SphereMesh(ResourceCounters& counters)
    : counters_(counters) {}

SphereMesh::~SphereMesh() {
    release();
}

void SphereMesh::create(int slices, int stacks) {
    if (vertexArray_ != 0) {
        return;
    }
    slices = std::max(slices, 8);
    stacks = std::max(stacks, 4);

    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
    };
    std::vector<Vertex> vertices;
    vertices.reserve(
        static_cast<std::size_t>(slices + 1)
        * static_cast<std::size_t>(stacks + 1)
    );

    for (int y = 0; y <= stacks; ++y) {
        const float v = static_cast<float>(y) / static_cast<float>(stacks);
        const float phi = v * 3.14159265358979323846F;
        const float sinPhi = std::sin(phi);
        const float cosPhi = std::cos(phi);
        for (int x = 0; x <= slices; ++x) {
            const float u = static_cast<float>(x) / static_cast<float>(slices);
            const float theta = u * 2.0F * 3.14159265358979323846F;
            const glm::vec3 normal(
                sinPhi * std::cos(theta),
                cosPhi,
                sinPhi * std::sin(theta)
            );
            vertices.push_back({normal, normal, glm::vec2(u, v)});
        }
    }

    std::vector<std::uint32_t> indices;
    indices.reserve(
        static_cast<std::size_t>(slices)
        * static_cast<std::size_t>(stacks) * 6U
    );
    const int row = slices + 1;
    for (int y = 0; y < stacks; ++y) {
        for (int x = 0; x < slices; ++x) {
            const std::uint32_t i0 =
                static_cast<std::uint32_t>(y * row + x);
            const std::uint32_t i1 = i0 + 1U;
            const std::uint32_t i2 =
                static_cast<std::uint32_t>((y + 1) * row + x);
            const std::uint32_t i3 = i2 + 1U;
            indices.push_back(i0);
            indices.push_back(i2);
            indices.push_back(i1);
            indices.push_back(i1);
            indices.push_back(i2);
            indices.push_back(i3);
        }
    }

    glGenVertexArrays(1, &vertexArray_);
    glGenBuffers(1, &vertexBuffer_);
    glGenBuffers(1, &indexBuffer_);
    if (vertexArray_ != 0) {
        ++counters_.vertexArraysCreated;
    }
    if (vertexBuffer_ != 0) {
        ++counters_.buffersCreated;
    }
    if (indexBuffer_ != 0) {
        ++counters_.buffersCreated;
    }
    if (vertexArray_ == 0 || vertexBuffer_ == 0 || indexBuffer_ == 0) {
        throw std::runtime_error("failed to create sphere mesh GL objects");
    }

    glBindVertexArray(vertexArray_);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer_);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)),
        vertices.data(),
        GL_STATIC_DRAW
    );
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer_);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(indices.size() * sizeof(std::uint32_t)),
        indices.data(),
        GL_STATIC_DRAW
    );

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        static_cast<GLsizei>(sizeof(Vertex)),
        reinterpret_cast<const void*>(offsetof(Vertex, position))
    );
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1,
        3,
        GL_FLOAT,
        GL_FALSE,
        static_cast<GLsizei>(sizeof(Vertex)),
        reinterpret_cast<const void*>(offsetof(Vertex, normal))
    );
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(
        2,
        2,
        GL_FLOAT,
        GL_FALSE,
        static_cast<GLsizei>(sizeof(Vertex)),
        reinterpret_cast<const void*>(offsetof(Vertex, uv))
    );
    glBindVertexArray(0);
    indexCount_ = static_cast<GLsizei>(indices.size());
}

void SphereMesh::draw() const {
    if (vertexArray_ != 0) {
        glBindVertexArray(vertexArray_);
        glDrawElements(
            GL_TRIANGLES,
            indexCount_,
            GL_UNSIGNED_INT,
            nullptr
        );
    }
}

void SphereMesh::release() {
    if (indexBuffer_ != 0) {
        glDeleteBuffers(1, &indexBuffer_);
        ++counters_.buffersReleased;
        indexBuffer_ = 0;
    }
    if (vertexBuffer_ != 0) {
        glDeleteBuffers(1, &vertexBuffer_);
        ++counters_.buffersReleased;
        vertexBuffer_ = 0;
    }
    if (vertexArray_ != 0) {
        glDeleteVertexArrays(1, &vertexArray_);
        ++counters_.vertexArraysReleased;
        vertexArray_ = 0;
    }
    indexCount_ = 0;
}

FullscreenVao::FullscreenVao(ResourceCounters& counters)
    : counters_(counters) {}

FullscreenVao::~FullscreenVao() {
    release();
}

void FullscreenVao::create() {
    if (vertexArray_ != 0) {
        return;
    }
    glGenVertexArrays(1, &vertexArray_);
    if (vertexArray_ == 0) {
        throw std::runtime_error("failed to create fullscreen VAO");
    }
    ++counters_.vertexArraysCreated;
}

void FullscreenVao::draw() const {
    if (vertexArray_ != 0) {
        glBindVertexArray(vertexArray_);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
}

void FullscreenVao::release() {
    if (vertexArray_ != 0) {
        glDeleteVertexArrays(1, &vertexArray_);
        ++counters_.vertexArraysReleased;
        vertexArray_ = 0;
    }
}

std::string glErrorName(GLenum error) {
    switch (error) {
        case GL_NO_ERROR:
            return "GL_NO_ERROR";
        case GL_INVALID_ENUM:
            return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE:
            return "GL_INVALID_VALUE";
        case GL_INVALID_OPERATION:
            return "GL_INVALID_OPERATION";
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            return "GL_INVALID_FRAMEBUFFER_OPERATION";
        case GL_OUT_OF_MEMORY:
            return "GL_OUT_OF_MEMORY";
        default:
            return "GL_ERROR_" + std::to_string(error);
    }
}

}  // namespace skin

