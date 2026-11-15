#include "renderer.hpp"

#include "error.hpp"
#include "gl_loader.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

namespace demo {

namespace {

constexpr GLuint kFrameUboBinding = 0;

constexpr const char *kCustomVertexShader = R"glsl(
#version 450 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in float aJointIndex;

layout(std140, binding = 0) uniform FrameData {
    mat4 uViewProjection;
    mat4 uModel;
    vec4 uDiffuseColor;
    vec4 uMaterialParams;
    vec4 uLightDirection;
    vec4 uViewPosition;
    mat4 uJointWorld[8];
};

out vec3 vWorldPosition;
out vec3 vWorldNormal;
out vec2 vTexCoord;

void main() {
    int jointIndex = int(clamp(aJointIndex, 0.0, 7.0));
    mat4 worldTransform = uModel * uJointWorld[jointIndex];
    vec4 worldPosition = worldTransform * vec4(aPosition, 1.0);

    vWorldPosition = worldPosition.xyz;
    vWorldNormal = mat3(transpose(inverse(worldTransform))) * aNormal;
    vTexCoord = aTexCoord;
    gl_Position = uViewProjection * worldPosition;
}
)glsl";

constexpr const char *kCustomFragmentShader = R"glsl(
#version 450 core

layout(std140, binding = 0) uniform FrameData {
    mat4 uViewProjection;
    mat4 uModel;
    vec4 uDiffuseColor;
    vec4 uMaterialParams;
    vec4 uLightDirection;
    vec4 uViewPosition;
    mat4 uJointWorld[8];
};

in vec3 vWorldPosition;
in vec3 vWorldNormal;
in vec2 vTexCoord;

layout(location = 0) out vec4 FragColor;

void main() {
    vec3 normal = normalize(vWorldNormal);
    vec3 light = normalize(uLightDirection.xyz);
    vec3 view = normalize(uViewPosition.xyz - vWorldPosition);
    vec3 halfVector = normalize(light + view);

    float roughness = clamp(uMaterialParams.x, 0.04, 1.0);
    float emissive = max(uMaterialParams.y, 0.0);
    float nDotL = max(dot(normal, light), 0.0);
    float nDotH = max(dot(normal, halfVector), 0.0);
    float shininess = mix(128.0, 6.0, roughness);
    float specular = pow(nDotH, shininess) * (1.0 - roughness);

    float microPattern = 0.97 + 0.03 * sin(vTexCoord.x * 31.0) *
                                   sin(vTexCoord.y * 31.0);
    vec3 albedo = uDiffuseColor.rgb * microPattern;
    vec3 ambient = albedo * vec3(0.15, 0.17, 0.20);
    vec3 direct = (albedo * nDotL + vec3(0.32 * specular)) *
                  vec3(1.0, 0.96, 0.90);
    vec3 emission = albedo * emissive * 2.0;
    vec3 color = ambient + direct + emission;

    FragColor = vec4(color, 1.0);
}
)glsl";

constexpr const char *kSafeVertexShader = R"glsl(
#version 450 core

layout(location = 0) in vec3 aPosition;
layout(location = 3) in float aJointIndex;

layout(std140, binding = 0) uniform FrameData {
    mat4 uViewProjection;
    mat4 uModel;
    vec4 uDiffuseColor;
    vec4 uMaterialParams;
    vec4 uLightDirection;
    vec4 uViewPosition;
    mat4 uJointWorld[8];
};

void main() {
    int jointIndex = int(clamp(aJointIndex, 0.0, 7.0));
    gl_Position = uViewProjection * uModel * uJointWorld[jointIndex] *
                  vec4(aPosition, 1.0);
}
)glsl";

constexpr const char *kSafeFragmentShader = R"glsl(
#version 450 core

layout(std140, binding = 0) uniform FrameData {
    mat4 uViewProjection;
    mat4 uModel;
    vec4 uDiffuseColor;
    vec4 uMaterialParams;
    vec4 uLightDirection;
    vec4 uViewPosition;
    mat4 uJointWorld[8];
};

layout(location = 0) out vec4 FragColor;

void main() {
    FragColor = vec4(uDiffuseColor.rgb, 1.0);
}
)glsl";

constexpr const char *kLinkMismatchVertexShader = R"glsl(
#version 450 core
layout(location = 0) in vec3 aPosition;
layout(location = 0) out vec3 vInjectedMismatch;
void main() {
    vInjectedMismatch = vec3(1.0);
    gl_Position = vec4(aPosition, 1.0);
}
)glsl";

constexpr const char *kLinkMismatchFragmentShader = R"glsl(
#version 450 core
layout(location = 0) in vec4 vInjectedMismatch;
layout(location = 0) out vec4 FragColor;
void main() {
    FragColor = vInjectedMismatch;
}
)glsl";

void throwOnGlError(ErrorCode code, const char *operation) {
    GLenum error = glGetError();
    if (error == GL_NO_ERROR) {
        return;
    }

    std::ostringstream message;
    message << operation << " failed with " << glErrorName(error);
    while ((error = glGetError()) != GL_NO_ERROR) {
        message << ", " << glErrorName(error);
    }
    throw AppError(code, message.str());
}

std::string shaderInfoLog(GLuint shader) {
    GLint length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    if (length <= 1) {
        return "<no shader log>";
    }
    std::vector<GLchar> buffer(static_cast<std::size_t>(length));
    GLsizei written = 0;
    glGetShaderInfoLog(shader, length, &written, buffer.data());
    return std::string(buffer.data(), static_cast<std::size_t>(written));
}

std::string programInfoLog(GLuint program) {
    GLint length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
    if (length <= 1) {
        return "<no program log>";
    }
    std::vector<GLchar> buffer(static_cast<std::size_t>(length));
    GLsizei written = 0;
    glGetProgramInfoLog(program, length, &written, buffer.data());
    return std::string(buffer.data(), static_cast<std::size_t>(written));
}

GLuint compileStage(GLenum type, const std::string &source, std::string &log) {
    const GLuint shader = glCreateShader(type);
    if (shader == 0) {
        log = "glCreateShader returned 0";
        return 0;
    }

    const GLchar *sourcePointer = source.c_str();
    glShaderSource(shader, 1, &sourcePointer, nullptr);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled != GL_TRUE) {
        log = shaderInfoLog(shader);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

struct ProgramAttempt {
    GLuint program = 0;
    ErrorCode failure = ErrorCode::Ok;
    std::string log;
};

ProgramAttempt buildProgram(const char *vertexSource, const char *fragmentSource,
                            bool injectCompileFailure) {
    std::string vertex = vertexSource;
    std::string fragment = fragmentSource;
    if (injectCompileFailure) {
        vertex += "\n#error DEMO_INJECTED_SHADER_COMPILE_FAILURE\n";
    }

    std::string vertexLog;
    const GLuint vertexShader = compileStage(GL_VERTEX_SHADER, vertex, vertexLog);
    if (vertexShader == 0) {
        return ProgramAttempt{0, ErrorCode::ShaderCompileFallback,
                              "Vertex shader compilation failed:\n" + vertexLog};
    }

    std::string fragmentLog;
    const GLuint fragmentShader =
        compileStage(GL_FRAGMENT_SHADER, fragment, fragmentLog);
    if (fragmentShader == 0) {
        glDeleteShader(vertexShader);
        return ProgramAttempt{0, ErrorCode::ShaderCompileFallback,
                              "Fragment shader compilation failed:\n" + fragmentLog};
    }

    const GLuint program = glCreateProgram();
    if (program == 0) {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return ProgramAttempt{0, ErrorCode::ShaderLinkFallback,
                              "glCreateProgram returned 0"};
    }

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE) {
        const std::string linkLog = programInfoLog(program);
        glDeleteProgram(program);
        return ProgramAttempt{0, ErrorCode::ShaderLinkFallback,
                              "Program link failed:\n" + linkLog};
    }

    return ProgramAttempt{program, ErrorCode::Ok, {}};
}

void bindFrameDataBlock(GLuint program) {
    const GLuint block = glGetUniformBlockIndex(program, "FrameData");
    if (block == GL_INVALID_INDEX) {
        throw AppError(ErrorCode::ShaderInterfaceMismatch,
                       "Shader program has no FrameData uniform block");
    }
    glUniformBlockBinding(program, block, kFrameUboBinding);

    GLint blockSize = 0;
    GLint blockBinding = -1;
    glGetActiveUniformBlockiv(program, block, GL_UNIFORM_BLOCK_DATA_SIZE,
                              &blockSize);
    glGetActiveUniformBlockiv(program, block, GL_UNIFORM_BLOCK_BINDING,
                              &blockBinding);
    if (blockSize != static_cast<GLint>(sizeof(FrameUniforms)) ||
        blockBinding != static_cast<GLint>(kFrameUboBinding)) {
        std::ostringstream message;
        message << "FrameData std140 mismatch: shader size=" << blockSize
                << " binding=" << blockBinding
                << ", CPU size=" << sizeof(FrameUniforms)
                << " expected binding=" << kFrameUboBinding;
        throw AppError(ErrorCode::ShaderInterfaceMismatch, message.str());
    }
}

} // namespace

Renderer::~Renderer() {
    destroy();
}

void Renderer::initialize(const MeshData &mesh, const RendererOptions &options) {
    while (glGetError() != GL_NO_ERROR) {
    }

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_MULTISAMPLE);
    glClearColor(0.035f, 0.045f, 0.070f, 1.0f);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    throwOnGlError(ErrorCode::GpuBufferUploadFailed,
                   "Initial OpenGL render state setup");

    ProgramAttempt custom;
    if (options.injectShaderLinkFailure) {
        custom = buildProgram(kLinkMismatchVertexShader,
                              kLinkMismatchFragmentShader, false);
    } else {
        custom = buildProgram(kCustomVertexShader, kCustomFragmentShader,
                              options.injectShaderCompileFailure);
    }

    if (custom.program != 0) {
        program_ = custom.program;
        shaderMode_ = ShaderMode::CustomLighting;
    } else {
        std::fprintf(stderr,
                     "[WARN][E%04d/%s] Custom shader failed; using built-in safe "
                     "solid-color shader.\n%s\n",
                     exitCode(custom.failure), errorCodeName(custom.failure),
                     custom.log.c_str());

        const ProgramAttempt safe =
            buildProgram(kSafeVertexShader, kSafeFragmentShader, false);
        if (safe.program == 0) {
            std::fprintf(stderr, "[ERROR][E%04d/%s] Safe shader failed.\n%s\n",
                         exitCode(ErrorCode::SafeShaderFailed),
                         errorCodeName(ErrorCode::SafeShaderFailed),
                         safe.log.c_str());
            throw AppError(ErrorCode::SafeShaderFailed,
                           "Both custom and safe shader programs failed");
        }
        program_ = safe.program;
        shaderMode_ = ShaderMode::SafeFallback;
    }

    bindFrameDataBlock(program_);
    throwOnGlError(ErrorCode::ShaderInterfaceMismatch,
                   "FrameData uniform block binding");

    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);

    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(mesh.vertices.size() * sizeof(Vertex)),
                 mesh.vertices.data(), GL_DYNAMIC_DRAW);

    glGenBuffers(1, &ebo_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(mesh.indices.size() *
                                         sizeof(std::uint32_t)),
                 mesh.indices.data(), GL_STATIC_DRAW);

    constexpr GLsizei stride = static_cast<GLsizei>(sizeof(Vertex));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0, 3, GL_FLOAT, GL_FALSE, stride,
        reinterpret_cast<const void *>(static_cast<std::uintptr_t>(
            offsetof(Vertex, position))));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 3, GL_FLOAT, GL_FALSE, stride,
        reinterpret_cast<const void *>(static_cast<std::uintptr_t>(
            offsetof(Vertex, normal))));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(
        2, 2, GL_FLOAT, GL_FALSE, stride,
        reinterpret_cast<const void *>(static_cast<std::uintptr_t>(
            offsetof(Vertex, texCoord))));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(
        3, 1, GL_FLOAT, GL_FALSE, stride,
        reinterpret_cast<const void *>(static_cast<std::uintptr_t>(
            offsetof(Vertex, jointIndex))));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // Keep the EBO attached to the VAO; the index binding is VAO state.
    throwOnGlError(ErrorCode::GpuBufferUploadFailed,
                   "Initial VAO/VBO/EBO upload");

    if (vao_ == 0 || vbo_ == 0 || ebo_ == 0) {
        throw AppError(ErrorCode::GpuBufferUploadFailed,
                       "OpenGL returned a zero VAO/VBO/EBO handle");
    }
    if (mesh.indices.size() > static_cast<std::size_t>(
                                  std::numeric_limits<GLsizei>::max())) {
        throw AppError(ErrorCode::GpuBufferUploadFailed,
                       "Index count exceeds GLsizei range");
    }

    glGenBuffers(1, &ubo_);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo_);
    glBufferData(GL_UNIFORM_BUFFER, static_cast<GLsizeiptr>(sizeof(FrameUniforms)),
                 nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, kFrameUboBinding, ubo_);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    throwOnGlError(ErrorCode::GpuBufferUploadFailed,
                   "FrameData UBO allocation and indexed binding");

    if (ubo_ == 0) {
        throw AppError(ErrorCode::GpuBufferUploadFailed,
                       "OpenGL returned a zero UBO handle");
    }

    indexCount_ = static_cast<GLsizei>(mesh.indices.size());
    vertexCount_ = mesh.vertices.size();
}

void Renderer::updateFrame(const FrameUniforms &frame) {
    glBindBuffer(GL_UNIFORM_BUFFER, ubo_);

    constexpr std::size_t headerSize = offsetof(FrameUniforms, diffuseColor);
    static_assert(headerSize == 128);
    if (!headerUploaded_ ||
        std::memcmp(&lastViewProjection_, &frame.viewProjection,
                    sizeof(glm::mat4)) != 0 ||
        std::memcmp(&lastModel_, &frame.model, sizeof(glm::mat4)) != 0) {
        glBufferSubData(GL_UNIFORM_BUFFER, 0,
                        static_cast<GLsizeiptr>(headerSize), &frame);
        lastViewProjection_ = frame.viewProjection;
        lastModel_ = frame.model;
        headerUploaded_ = true;
        ++headerUploadCount_;
    }

    constexpr std::size_t dynamicOffset = offsetof(FrameUniforms, diffuseColor);
    constexpr std::size_t dynamicSize = sizeof(FrameUniforms) - dynamicOffset;
    static_assert(dynamicOffset == 128);
    static_assert(dynamicSize == 576);
    glBufferSubData(
        GL_UNIFORM_BUFFER, static_cast<GLintptr>(dynamicOffset),
        static_cast<GLsizeiptr>(dynamicSize),
        reinterpret_cast<const unsigned char *>(&frame) + dynamicOffset);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    throwOnGlError(ErrorCode::UniformBufferSyncFailed,
                   "glBufferSubData partial FrameData UBO sync");
    ++frameUploadCount_;
}

void Renderer::updateMeshVertices(const std::vector<Vertex> &vertices) {
    if (vertices.empty() || vertices.size() != vertexCount_) {
        throw AppError(ErrorCode::MeshDynamicUpdateFailed,
                       "Dynamic vertex update must preserve the original vertex count");
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)),
                    vertices.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    throwOnGlError(ErrorCode::MeshDynamicUpdateFailed,
                   "glBufferSubData dynamic VBO update");
    ++meshUpdateCount_;
}

void Renderer::draw() {
    glUseProgram(program_);
    glBindVertexArray(vao_);
    glDrawElements(GL_TRIANGLES, indexCount_, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
    glUseProgram(0);
}

FrameUniforms Renderer::readBackFrame() {
    FrameUniforms frame{};
    glBindBuffer(GL_UNIFORM_BUFFER, ubo_);
    glGetBufferSubData(GL_UNIFORM_BUFFER, 0,
                       static_cast<GLsizeiptr>(sizeof(FrameUniforms)), &frame);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    throwOnGlError(ErrorCode::UniformBufferSyncFailed,
                   "FrameData UBO readback");
    return frame;
}

void Renderer::destroy() {
    if (glDeleteProgram != nullptr && program_ != 0) {
        glDeleteProgram(program_);
    }
    if (glDeleteVertexArrays != nullptr && vao_ != 0) {
        glDeleteVertexArrays(1, &vao_);
    }
    if (glDeleteBuffers != nullptr && vbo_ != 0) {
        glDeleteBuffers(1, &vbo_);
    }
    if (glDeleteBuffers != nullptr && ebo_ != 0) {
        glDeleteBuffers(1, &ebo_);
    }
    if (glDeleteBuffers != nullptr && ubo_ != 0) {
        glDeleteBuffers(1, &ubo_);
    }

    program_ = 0;
    vao_ = 0;
    vbo_ = 0;
    ebo_ = 0;
    ubo_ = 0;
    indexCount_ = 0;
    vertexCount_ = 0;
}

} // namespace demo
