#include "lightcluster/renderer.hpp"

#ifndef GLEW_NO_GLU
#define GLEW_NO_GLU
#endif
#include <GL/glew.h>
#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lightcluster {

namespace {

constexpr int kInitialWidth = 960;
constexpr int kInitialHeight = 540;

const char* kGeometryVertexShader = R"(
#version 450 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aAlbedo;

uniform mat4 uModel;
uniform mat4 uViewProjection;
uniform mat3 uNormalMatrix;
uniform vec3 uTint;

out vec3 vWorldPosition;
out vec3 vNormal;
out vec3 vAlbedo;

void main() {
    vec4 world = uModel * vec4(aPosition, 1.0);
    vWorldPosition = world.xyz;
    vNormal = uNormalMatrix * aNormal;
    vAlbedo = aAlbedo * uTint;
    gl_Position = uViewProjection * world;
}
)";

const char* kGeometryFragmentShader = R"(
#version 450 core
in vec3 vWorldPosition;
in vec3 vNormal;
in vec3 vAlbedo;

layout(location = 0) out vec4 gPosition;
layout(location = 1) out vec4 gNormal;
layout(location = 2) out vec4 gAlbedo;

void main() {
    gPosition = vec4(vWorldPosition, 1.0);
    gNormal = vec4(normalize(vNormal), 1.0);
    gAlbedo = vec4(vAlbedo, 1.0);
}
)";

const char* kFullscreenVertexShader = R"(
#version 450 core
out vec2 vUv;

void main() {
    const vec2 positions[4] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0,  1.0)
    );
    vec2 position = positions[gl_VertexID];
    vUv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
)";

const char* kLightingFragmentShader = R"(
#version 450 core
in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uGPosition;
uniform sampler2D uGNormal;
uniform sampler2D uGAlbedo;
uniform vec3 uCameraPosition;

struct GpuLight {
    vec4 positionType;
    vec4 directionRange;
    vec4 colorIntensity;
};

layout(std430, binding = 0) readonly buffer LightBuffer {
    uint header[8];
    GpuLight lights[];
};

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 position = texelFetch(uGPosition, pixel, 0);
    if (position.w < 0.5) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 worldPosition = position.xyz;
    vec3 normal = normalize(texelFetch(uGNormal, pixel, 0).xyz);
    vec3 albedo = texelFetch(uGAlbedo, pixel, 0).rgb;
    vec3 viewDirection = normalize(uCameraPosition - worldPosition);
    vec3 accumulated = albedo * 0.035;

    uint lightCount = header[0];
    for (uint i = 0; i < lightCount; ++i) {
        GpuLight light = lights[i];
        vec3 lightDirection;
        float attenuation = 1.0;

        if (light.positionType.w > 0.5) {
            lightDirection = normalize(-light.directionRange.xyz);
        } else {
            vec3 toLight = light.positionType.xyz - worldPosition;
            float distanceToLight = max(length(toLight), 1.0e-4);
            lightDirection = toLight / distanceToLight;
            float lightRange = max(light.directionRange.w, 0.01);
            float ratio = clamp(distanceToLight / lightRange, 0.0, 1.0);
            float rangeFade = 1.0 - ratio * ratio * ratio * ratio;
            rangeFade *= rangeFade;
            float inverseSquare =
                1.0 / (1.0 + 0.09 * distanceToLight +
                       0.032 * distanceToLight * distanceToLight);
            attenuation = inverseSquare * rangeFade;
        }

        float diffuse = max(dot(normal, lightDirection), 0.0);
        if (diffuse <= 0.0) {
            continue;
        }

        vec3 halfVector = normalize(lightDirection + viewDirection);
        float specular = pow(max(dot(normal, halfVector), 0.0), 64.0);
        vec3 radiance =
            light.colorIntensity.rgb * light.colorIntensity.w * attenuation;
        accumulated += radiance *
            (albedo * diffuse + vec3(0.18) * specular);
    }

    fragColor = vec4(accumulated, 1.0);
}
)";

const char* kCompositeFragmentShader = R"(
#version 450 core
in vec2 vUv;
out vec4 fragColor;
uniform sampler2D uHdrColor;

void main() {
    vec3 color = texture(uHdrColor, vUv).rgb;
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    fragColor = vec4(color, 1.0);
}
)";

struct Vertex {
    std::array<float, 3> position;
    std::array<float, 3> normal;
    std::array<float, 3> albedo;
};

struct Mesh {
    GLuint vertexArray = 0;
    GLuint vertexBuffer = 0;
    GLuint indexBuffer = 0;
    GLsizei indexCount = 0;
};

GLuint compileShader(GLenum type, const char* source, const char* stage) {
    const GLuint shader = glCreateShader(type);
    if (shader == 0) {
        throw std::runtime_error("glCreateShader failed");
    }
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint status = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        glDeleteShader(shader);
        throw std::runtime_error(
            std::string("shader compile failed (") + stage + "): " + log);
    }
    return shader;
}

GLuint linkProgram(
    const char* vertexSource,
    const char* fragmentSource,
    const char* label) {
    const GLuint vertex = compileShader(
        GL_VERTEX_SHADER, vertexSource, (std::string(label) + " vertex").c_str());
    const GLuint fragment = compileShader(
        GL_FRAGMENT_SHADER,
        fragmentSource,
        (std::string(label) + " fragment").c_str());
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    GLint status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        GLint length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
        glGetProgramInfoLog(program, length, nullptr, log.data());
        glDeleteProgram(program);
        throw std::runtime_error(
            std::string("program link failed (") + label + "): " + log);
    }
    return program;
}

GLuint makeTexture(GLenum internalFormat, GLenum format, GLenum type) {
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        static_cast<GLint>(internalFormat),
        kInitialWidth,
        kInitialHeight,
        0,
        format,
        type,
        nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return texture;
}

Mesh makeMesh(
    const std::vector<Vertex>& vertices,
    const std::vector<std::uint32_t>& indices) {
    Mesh mesh;
    glGenVertexArrays(1, &mesh.vertexArray);
    glGenBuffers(1, &mesh.vertexBuffer);
    glGenBuffers(1, &mesh.indexBuffer);

    glBindVertexArray(mesh.vertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vertexBuffer);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)),
        vertices.data(),
        GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.indexBuffer);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(indices.size() * sizeof(std::uint32_t)),
        indices.data(),
        GL_STATIC_DRAW);

    const GLsizei stride = static_cast<GLsizei>(sizeof(Vertex));
    glVertexAttribPointer(
        0, 3, GL_FLOAT, GL_FALSE, stride,
        reinterpret_cast<const void*>(offsetof(Vertex, position)));
    glVertexAttribPointer(
        1, 3, GL_FLOAT, GL_FALSE, stride,
        reinterpret_cast<const void*>(offsetof(Vertex, normal)));
    glVertexAttribPointer(
        2, 3, GL_FLOAT, GL_FALSE, stride,
        reinterpret_cast<const void*>(offsetof(Vertex, albedo)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);
    mesh.indexCount = static_cast<GLsizei>(indices.size());
    return mesh;
}

Mesh makeFloor() {
    const std::vector<Vertex> vertices = {
        {{-1.0F, 0.0F, -1.0F}, {0.0F, 1.0F, 0.0F}, {0.32F, 0.36F, 0.42F}},
        {{1.0F, 0.0F, -1.0F}, {0.0F, 1.0F, 0.0F}, {0.32F, 0.36F, 0.42F}},
        {{1.0F, 0.0F, 1.0F}, {0.0F, 1.0F, 0.0F}, {0.32F, 0.36F, 0.42F}},
        {{-1.0F, 0.0F, 1.0F}, {0.0F, 1.0F, 0.0F}, {0.32F, 0.36F, 0.42F}},
    };
    const std::vector<std::uint32_t> indices = {0, 2, 1, 0, 3, 2};
    return makeMesh(vertices, indices);
}

Mesh makeCube() {
    const std::array<std::array<float, 3>, 6> faces = {{
        {{0.0F, 0.0F, 1.0F}},
        {{0.0F, 0.0F, -1.0F}},
        {{1.0F, 0.0F, 0.0F}},
        {{-1.0F, 0.0F, 0.0F}},
        {{0.0F, 1.0F, 0.0F}},
        {{0.0F, -1.0F, 0.0F}},
    }};
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    vertices.reserve(24);
    indices.reserve(36);

    for (std::size_t face = 0; face < faces.size(); ++face) {
        const auto& normal = faces[face];
        std::array<float, 3> tangent =
            std::fabs(normal[1]) > 0.5F
                ? std::array<float, 3>{1.0F, 0.0F, 0.0F}
                : std::array<float, 3>{0.0F, 1.0F, 0.0F};
        std::array<float, 3> bitangent = {
            normal[1] * tangent[2] - normal[2] * tangent[1],
            normal[2] * tangent[0] - normal[0] * tangent[2],
            normal[0] * tangent[1] - normal[1] * tangent[0],
        };

        const std::uint32_t base =
            static_cast<std::uint32_t>(vertices.size());
        for (int corner = 0; corner < 4; ++corner) {
            const float u = (corner == 1 || corner == 2) ? 1.0F : -1.0F;
            const float v = (corner >= 2) ? 1.0F : -1.0F;
            Vertex vertex{};
            for (int axis = 0; axis < 3; ++axis) {
                vertex.position[axis] =
                    normal[axis] * 0.5F + tangent[axis] * u * 0.5F +
                    bitangent[axis] * v * 0.5F;
                vertex.normal[axis] = normal[axis];
            }
            vertex.albedo = {0.74F, 0.68F, 0.58F};
            vertices.push_back(vertex);
        }
        indices.push_back(base + 0);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 0);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
    }
    return makeMesh(vertices, indices);
}

void destroyMesh(Mesh& mesh) {
    if (mesh.indexBuffer != 0) {
        glDeleteBuffers(1, &mesh.indexBuffer);
    }
    if (mesh.vertexBuffer != 0) {
        glDeleteBuffers(1, &mesh.vertexBuffer);
    }
    if (mesh.vertexArray != 0) {
        glDeleteVertexArrays(1, &mesh.vertexArray);
    }
    mesh = {};
}

GLuint uniformLocation(GLuint program, const char* name) {
    const GLint location = glGetUniformLocation(program, name);
    if (location < 0) {
        throw std::runtime_error(
            std::string("missing shader uniform: ") + name);
    }
    return static_cast<GLuint>(location);
}

void drawMesh(const Mesh& mesh) {
    glBindVertexArray(mesh.vertexArray);
    glDrawElements(
        GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, nullptr);
}

}  // namespace

struct Renderer::Impl {
    GLFWwindow* window = nullptr;
    bool visible = true;

    GLuint gBufferFramebuffer = 0;
    GLuint positionTexture = 0;
    GLuint normalTexture = 0;
    GLuint albedoTexture = 0;
    GLuint depthRenderbuffer = 0;

    GLuint hdrFramebuffer = 0;
    GLuint hdrTexture = 0;
    GLuint fullscreenVertexArray = 0;

    GLuint geometryProgram = 0;
    GLuint lightingProgram = 0;
    GLuint compositeProgram = 0;

    GLuint lightBuffer = 0;
    std::uint32_t allocatedCapacityLights = 0;
    std::uint32_t allocatedBytes = 0;
    std::uint32_t usedBytes = 0;

    Mesh floor;
    Mesh cube;
    std::vector<glm::mat4> cubeModels;
    glm::mat4 viewProjection{1.0F};
    glm::vec3 cameraPosition{0.0F, 2.2F, 8.0F};

    ~Impl() {
        if (window != nullptr) {
            glfwDestroyWindow(window);
        }
    }
};

Renderer::Renderer() : impl_(std::make_unique<Impl>()) {}

Renderer::~Renderer() {
    if (impl_ == nullptr) {
        return;
    }
    if (impl_->window != nullptr) {
        glfwMakeContextCurrent(impl_->window);
    }
    destroyMesh(impl_->floor);
    destroyMesh(impl_->cube);
    if (impl_->lightBuffer != 0) {
        glDeleteBuffers(1, &impl_->lightBuffer);
    }
    if (impl_->geometryProgram != 0) {
        glDeleteProgram(impl_->geometryProgram);
    }
    if (impl_->lightingProgram != 0) {
        glDeleteProgram(impl_->lightingProgram);
    }
    if (impl_->compositeProgram != 0) {
        glDeleteProgram(impl_->compositeProgram);
    }
    if (impl_->hdrFramebuffer != 0) {
        glDeleteFramebuffers(1, &impl_->hdrFramebuffer);
    }
    if (impl_->gBufferFramebuffer != 0) {
        glDeleteFramebuffers(1, &impl_->gBufferFramebuffer);
    }
    if (impl_->hdrTexture != 0) {
        glDeleteTextures(1, &impl_->hdrTexture);
    }
    if (impl_->positionTexture != 0) {
        glDeleteTextures(1, &impl_->positionTexture);
    }
    if (impl_->normalTexture != 0) {
        glDeleteTextures(1, &impl_->normalTexture);
    }
    if (impl_->albedoTexture != 0) {
        glDeleteTextures(1, &impl_->albedoTexture);
    }
    if (impl_->depthRenderbuffer != 0) {
        glDeleteRenderbuffers(1, &impl_->depthRenderbuffer);
    }
    if (impl_->fullscreenVertexArray != 0) {
        glDeleteVertexArrays(1, &impl_->fullscreenVertexArray);
    }
    impl_->window = nullptr;
    glfwTerminate();
}

void Renderer::initialize(bool headless) {
    if (!glfwInit()) {
        throw std::runtime_error("glfwInit failed");
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_VISIBLE, headless ? GLFW_FALSE : GLFW_TRUE);
    impl_->visible = !headless;
    impl_->window = glfwCreateWindow(
        kInitialWidth, kInitialHeight, "Dynamic Light Cluster Deferred",
        nullptr, nullptr);
    if (impl_->window == nullptr) {
        throw std::runtime_error("glfwCreateWindow failed");
    }
    glfwMakeContextCurrent(impl_->window);
    glfwSwapInterval(0);

    glewExperimental = GL_TRUE;
    const GLenum glewStatus = glewInit();
    if (glewStatus != GLEW_OK) {
        throw std::runtime_error(
            std::string("glewInit failed: ") +
            reinterpret_cast<const char*>(glewGetErrorString(glewStatus)));
    }
    while (glGetError() != GL_NO_ERROR) {
    }

    GLint major = 0;
    GLint minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    if (major < 4 || (major == 4 && minor < 5)) {
        throw std::runtime_error(
            "OpenGL 4.5 core profile is required; got " +
            std::to_string(major) + "." + std::to_string(minor));
    }

    impl_->geometryProgram = linkProgram(
        kGeometryVertexShader, kGeometryFragmentShader, "geometry");
    impl_->lightingProgram = linkProgram(
        kFullscreenVertexShader, kLightingFragmentShader, "lighting");
    impl_->compositeProgram = linkProgram(
        kFullscreenVertexShader, kCompositeFragmentShader, "composite");

    impl_->positionTexture =
        makeTexture(GL_RGBA16F, GL_RGBA, GL_FLOAT);
    impl_->normalTexture =
        makeTexture(GL_RGBA16F, GL_RGBA, GL_FLOAT);
    impl_->albedoTexture =
        makeTexture(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

    glGenFramebuffers(1, &impl_->gBufferFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, impl_->gBufferFramebuffer);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, impl_->positionTexture, 0);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
        GL_TEXTURE_2D, impl_->normalTexture, 0);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2,
        GL_TEXTURE_2D, impl_->albedoTexture, 0);
    glGenRenderbuffers(1, &impl_->depthRenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, impl_->depthRenderbuffer);
    glRenderbufferStorage(
        GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
        kInitialWidth, kInitialHeight);
    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
        GL_RENDERBUFFER, impl_->depthRenderbuffer);
    const std::array<GLenum, 3> drawBuffers = {
        GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, drawBuffers.data());
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("G-Buffer framebuffer is incomplete");
    }

    impl_->hdrTexture =
        makeTexture(GL_RGBA16F, GL_RGBA, GL_FLOAT);
    glGenFramebuffers(1, &impl_->hdrFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, impl_->hdrFramebuffer);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, impl_->hdrTexture, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("HDR framebuffer is incomplete");
    }

    glGenVertexArrays(1, &impl_->fullscreenVertexArray);
    impl_->floor = makeFloor();
    impl_->cube = makeCube();

    for (int z = -1; z <= 1; ++z) {
        for (int x = -1; x <= 1; ++x) {
            const float height = 0.55F + 0.18F * static_cast<float>((x + z + 2) % 3);
            glm::mat4 model(1.0F);
            model = glm::translate(
                model,
                glm::vec3(
                    static_cast<float>(x) * 2.15F,
                    height * 0.5F,
                    static_cast<float>(z) * 2.15F));
            model = glm::scale(
                model,
                glm::vec3(1.45F, height, 1.45F));
            impl_->cubeModels.push_back(model);
        }
    }

    const glm::mat4 floorModel = glm::scale(
        glm::mat4(1.0F), glm::vec3(14.0F, 1.0F, 14.0F));
    impl_->cubeModels.push_back(floorModel);

    const glm::mat4 projection = glm::perspective(
        glm::radians(47.0F),
        static_cast<float>(kInitialWidth) /
            static_cast<float>(kInitialHeight),
        0.1F, 100.0F);
    const glm::mat4 view = glm::lookAt(
        impl_->cameraPosition, glm::vec3(0.0F, 0.7F, 0.0F),
        glm::vec3(0.0F, 1.0F, 0.0F));
    impl_->viewProjection = projection * view;

    glGenBuffers(1, &impl_->lightBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, impl_->lightBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, impl_->lightBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
}

void Renderer::upload(const GpuBufferImage& image, bool grew) {
    const std::uint32_t capacityBytes =
        static_cast<std::uint32_t>(
            kGpuHeaderBytes +
            static_cast<std::size_t>(image.capacityLights) *
                kGpuLightBytes);
    if (image.usedBytes > image.bytes.size()) {
        throw std::runtime_error("light image used size exceeds payload");
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, impl_->lightBuffer);
    if (grew ||
        impl_->allocatedCapacityLights < image.capacityLights ||
        impl_->allocatedBytes < capacityBytes) {
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            static_cast<GLsizeiptr>(capacityBytes),
            nullptr,
            GL_DYNAMIC_DRAW);
        impl_->allocatedCapacityLights = image.capacityLights;
        impl_->allocatedBytes = capacityBytes;
    }
    glBufferSubData(
        GL_SHADER_STORAGE_BUFFER,
        0,
        static_cast<GLsizeiptr>(image.usedBytes),
        image.bytes.data());
    impl_->usedBytes = image.usedBytes;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    const GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        throw std::runtime_error(
            "light buffer upload failed with GL error " +
            std::to_string(static_cast<unsigned int>(error)));
    }
}

void Renderer::render(const GpuBufferImage& image) {
    if (impl_->lightBuffer == 0 || impl_->usedBytes == 0) {
        throw std::runtime_error("render called before light buffer upload");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, impl_->gBufferFramebuffer);
    glViewport(0, 0, kInitialWidth, kInitialHeight);
    glClearColor(0.0F, 0.0F, 0.0F, 0.0F);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(impl_->geometryProgram);
    const GLuint modelLocation =
        uniformLocation(impl_->geometryProgram, "uModel");
    const GLuint viewProjectionLocation =
        uniformLocation(impl_->geometryProgram, "uViewProjection");
    const GLuint normalMatrixLocation =
        uniformLocation(impl_->geometryProgram, "uNormalMatrix");
    const GLuint tintLocation =
        uniformLocation(impl_->geometryProgram, "uTint");
    glUniformMatrix4fv(
        static_cast<GLint>(viewProjectionLocation), 1, GL_FALSE,
        glm::value_ptr(impl_->viewProjection));

    for (std::size_t i = 0; i < impl_->cubeModels.size(); ++i) {
        const glm::mat4& model = impl_->cubeModels[i];
        glUniformMatrix4fv(
            static_cast<GLint>(modelLocation), 1, GL_FALSE,
            glm::value_ptr(model));
        const glm::mat3 normalMatrix =
            glm::transpose(glm::inverse(glm::mat3(model)));
        glUniformMatrix3fv(
            static_cast<GLint>(normalMatrixLocation), 1, GL_FALSE,
            glm::value_ptr(normalMatrix));
        if (i + 1 == impl_->cubeModels.size()) {
            glUniform3f(static_cast<GLint>(tintLocation), 1.0F, 1.0F, 1.0F);
            drawMesh(impl_->floor);
        } else {
            glUniform3f(
                static_cast<GLint>(tintLocation),
                1.0F + 0.08F * static_cast<float>(i % 3),
                1.0F,
                1.0F - 0.06F * static_cast<float>(i % 2));
            drawMesh(impl_->cube);
        }
    }
    glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, impl_->hdrFramebuffer);
    glViewport(0, 0, kInitialWidth, kInitialHeight);
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.0F, 0.0F, 0.0F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(impl_->lightingProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, impl_->positionTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, impl_->normalTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, impl_->albedoTexture);
    glUniform1i(
        static_cast<GLint>(
            uniformLocation(impl_->lightingProgram, "uGPosition")), 0);
    glUniform1i(
        static_cast<GLint>(
            uniformLocation(impl_->lightingProgram, "uGNormal")), 1);
    glUniform1i(
        static_cast<GLint>(
            uniformLocation(impl_->lightingProgram, "uGAlbedo")), 2);
    glUniform3fv(
        static_cast<GLint>(
            uniformLocation(impl_->lightingProgram, "uCameraPosition")),
        1, glm::value_ptr(impl_->cameraPosition));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, impl_->lightBuffer);
    glBindVertexArray(impl_->fullscreenVertexArray);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, kInitialWidth, kInitialHeight);
    glClearColor(0.0F, 0.0F, 0.0F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(impl_->compositeProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, impl_->hdrTexture);
    glUniform1i(
        static_cast<GLint>(
            uniformLocation(impl_->compositeProgram, "uHdrColor")), 0);
    glBindVertexArray(impl_->fullscreenVertexArray);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    glEnable(GL_DEPTH_TEST);
    const GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        throw std::runtime_error(
            "render failed with GL error " +
            std::to_string(static_cast<unsigned int>(error)));
    }
    (void)image;
}

std::vector<std::uint8_t> Renderer::readback(std::uint32_t usedBytes) {
    if (usedBytes == 0 || usedBytes > impl_->usedBytes) {
        throw std::runtime_error("invalid GPU readback size");
    }
    std::vector<std::uint8_t> bytes(usedBytes);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, impl_->lightBuffer);
    glGetBufferSubData(
        GL_SHADER_STORAGE_BUFFER, 0,
        static_cast<GLsizeiptr>(usedBytes), bytes.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    const GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        throw std::runtime_error(
            "glGetBufferSubData failed with GL error " +
            std::to_string(static_cast<unsigned int>(error)));
    }
    return bytes;
}

void Renderer::corruptCountForDiagnostic() {
    std::uint32_t count = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, impl_->lightBuffer);
    glGetBufferSubData(
        GL_SHADER_STORAGE_BUFFER, 0, sizeof(count), &count);
    ++count;
    glBufferSubData(
        GL_SHADER_STORAGE_BUFFER, 0, sizeof(count), &count);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void Renderer::pollEvents() {
    glfwPollEvents();
}

void Renderer::swapBuffers(bool headless) {
    if (!headless) {
        glfwSwapBuffers(impl_->window);
    }
}

bool Renderer::shouldClose() const {
    return glfwWindowShouldClose(impl_->window) == GLFW_TRUE;
}

int Renderer::width() const {
    return kInitialWidth;
}

int Renderer::height() const {
    return kInitialHeight;
}

}  // namespace lightcluster
