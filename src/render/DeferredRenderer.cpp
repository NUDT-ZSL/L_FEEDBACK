#include "indoor/DeferredRenderer.hpp"

#include "indoor/GLUtil.hpp"
#include "indoor/ShaderProgram.hpp"

#include "GBufferLayout.hpp"

#include <glad/gl.h>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace indoor {
namespace {

constexpr int kCubeVertexCount = 36;
constexpr float kShadowNear = 0.05f;
constexpr int kShadowMapBindBase = 5;

struct CpuVertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 tangent;
    glm::vec2 uv;
};

struct alignas(16) GPUInstance {
    glm::mat4 model;
    glm::vec4 albedoRoughness;
    glm::vec4 materialLayers;
    glm::vec4 materialParams;
};

struct alignas(16) LightBatchGPU {
    glm::vec4 positionRange[kMaxLightsPerBatch];
    glm::vec4 colorIntensity[kMaxLightsPerBatch];
    glm::ivec4 meta[kMaxLightsPerBatch];
    glm::ivec4 counts;
};

struct alignas(16) DebugLightGPU {
    glm::vec4 positionRange;
    glm::vec4 colorIntensity;
    glm::ivec4 meta;
};

static_assert(sizeof(GPUInstance) == 112);
static_assert(sizeof(LightBatchGPU) == 11 * 16 * 3 + 16);
static_assert(sizeof(DebugLightGPU) == 48);
static_assert(offsetof(GPUInstance, model) == 0);
static_assert(offsetof(GPUInstance, albedoRoughness) == 64);
static_assert(offsetof(GPUInstance, materialLayers) == 80);
static_assert(offsetof(GPUInstance, materialParams) == 96);

int chooseShadowSize(int width, int height) {
    const int smallest = std::max(1, std::min(width, height));
    const int desired = std::clamp(smallest / 8, 32, 192);
    return std::max(32, (desired / 8) * 8);
}

std::array<CpuVertex, 24> buildCubeVertices() {
    struct Face {
        glm::vec3 normal;
        glm::vec3 tangent;
    };
    const std::array<Face, 6> faces = {{
        {{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}},
        {{-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
        {{0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, 0.0f}},
    }};
    const std::array<glm::vec2, 4> uvs = {{
        {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f},
    }};
    const std::array<glm::vec2, 4> corners = {{
        {-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 1.0f},
    }};
    std::array<CpuVertex, 24> vertices{};
    for (std::size_t face = 0; face < faces.size(); ++face) {
        const glm::vec3 bitangent =
            glm::normalize(glm::cross(faces[face].normal, faces[face].tangent));
        for (std::size_t corner = 0; corner < corners.size(); ++corner) {
            CpuVertex& vertex = vertices[face * 4 + corner];
            vertex.position = faces[face].normal +
                              faces[face].tangent * corners[corner].x +
                              bitangent * corners[corner].y;
            vertex.normal = faces[face].normal;
            vertex.tangent = faces[face].tangent;
            vertex.uv = uvs[corner];
        }
    }
    return vertices;
}

std::array<std::uint32_t, 36> buildCubeIndices() {
    std::array<std::uint32_t, 36> indices{};
    for (std::uint32_t face = 0; face < 6; ++face) {
        const std::uint32_t base = face * 4;
        const std::uint32_t offset = face * 6;
        indices[offset + 0] = base + 0;
        indices[offset + 1] = base + 1;
        indices[offset + 2] = base + 2;
        indices[offset + 3] = base + 0;
        indices[offset + 4] = base + 2;
        indices[offset + 5] = base + 3;
    }
    return indices;
}

void createTexture2D(unsigned int& texture, int width, int height, GLenum internalFormat,
                     GLenum format, GLenum type, GLenum filter) {
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(internalFormat), width, height,
                 0, format, type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(filter));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(filter));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void checkFramebuffer(const char* label) {
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error(std::string("Incomplete framebuffer: ") + label +
                                 " status=" + std::to_string(status));
    }
}

std::array<glm::vec3, 6> cubeDirections() {
    return {{
        {1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f},
    }};
}

std::array<glm::vec3, 6> cubeUps() {
    return {{
        {0.0f, -1.0f, 0.0f}, {0.0f, -1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f},
        {0.0f, -1.0f, 0.0f}, {0.0f, -1.0f, 0.0f},
    }};
}

} // namespace

struct DeferredRenderer::Impl {
    RenderSettings settings;
    std::string shaderDirectory;
    MaterialTextureData materialCpu;
    ShadowVersionLedger ledger;
    ShadowCachePolicy cache{0};
    ShadowFrameStats stats;
    FrameSnapshot lastSnapshot;
    bool gpuReady = false;

    unsigned int gbufferFbo = 0;
    std::array<unsigned int, gbuffer::AttachmentCount> gbufferTextures{};
    unsigned int gbufferDepth = 0;
    unsigned int hdrFbo = 0;
    unsigned int hdrTexture = 0;
    unsigned int finalFbo = 0;
    unsigned int finalTexture = 0;

    unsigned int cubeVao = 0;
    unsigned int cubeVbo = 0;
    unsigned int cubeEbo = 0;
    unsigned int instanceVbo = 0;
    std::size_t instanceCapacity = 0;
    std::size_t instanceCount = 0;
    unsigned int emptyVao = 0;

    unsigned int shadowFbo = 0;
    unsigned int materialTexture = 0;
    unsigned int fallbackShadowCube = 0;
    unsigned int lightUbo = 0;
    unsigned int debugLightsSsbo = 0;
    std::size_t debugLightsCapacity = 0;

    std::unordered_map<std::uint32_t, unsigned int> shadowTextures;
    ShaderProgram gbufferProgram;
    ShaderProgram shadowProgram;
    ShaderProgram lightingProgram;
    ShaderProgram compositeProgram;

    Impl() = default;
    ~Impl() {
        if (gpuReady) {
            releaseGpu(true);
        }
    }

    void requireGpu(const char* operation) const {
        RenderThreadGuard::requireRenderThread(operation);
        if (!gpuReady) {
            throw std::runtime_error(std::string("GPU resources unavailable: ") +
                                     operation);
        }
    }

    void createGpuObjects() {
        RenderThreadGuard::requireRenderThread("createGpuObjects");
        createShaders();
        createCubeGeometry();
        createMaterialTexture();
        createLightBuffers();
        createRenderTargets();
        createShadowResources();
        glDisable(GL_DITHER);
        glDisable(GL_MULTISAMPLE);
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
        gpuReady = true;
    }

    void releaseGpu(bool deleteObjects) {
        if (deleteObjects && gpuReady) {
            for (const auto& [lightId, texture] : shadowTextures) {
                (void)lightId;
                glDeleteTextures(1, &texture);
            }
            for (unsigned int texture : gbufferTextures) {
                if (texture != 0) {
                    glDeleteTextures(1, &texture);
                }
            }
            if (gbufferDepth != 0) {
                glDeleteRenderbuffers(1, &gbufferDepth);
            }
            if (hdrTexture != 0) {
                glDeleteTextures(1, &hdrTexture);
            }
            if (finalTexture != 0) {
                glDeleteTextures(1, &finalTexture);
            }
            if (materialTexture != 0) {
                glDeleteTextures(1, &materialTexture);
            }
            if (fallbackShadowCube != 0) {
                glDeleteTextures(1, &fallbackShadowCube);
            }
            if (gbufferFbo != 0) {
                glDeleteFramebuffers(1, &gbufferFbo);
            }
            if (hdrFbo != 0) {
                glDeleteFramebuffers(1, &hdrFbo);
            }
            if (finalFbo != 0) {
                glDeleteFramebuffers(1, &finalFbo);
            }
            if (shadowFbo != 0) {
                glDeleteFramebuffers(1, &shadowFbo);
            }
            if (cubeVao != 0) {
                glDeleteVertexArrays(1, &cubeVao);
            }
            if (emptyVao != 0) {
                glDeleteVertexArrays(1, &emptyVao);
            }
            if (cubeVbo != 0) {
                glDeleteBuffers(1, &cubeVbo);
            }
            if (cubeEbo != 0) {
                glDeleteBuffers(1, &cubeEbo);
            }
            if (instanceVbo != 0) {
                glDeleteBuffers(1, &instanceVbo);
            }
            if (lightUbo != 0) {
                glDeleteBuffers(1, &lightUbo);
            }
            if (debugLightsSsbo != 0) {
                glDeleteBuffers(1, &debugLightsSsbo);
            }
            for (ShaderProgram* program : {&gbufferProgram, &shadowProgram,
                                           &lightingProgram, &compositeProgram}) {
                if (program->id() != 0) {
                    glDeleteProgram(program->id());
                }
            }
        }
        gbufferProgram.abandon();
        shadowProgram.abandon();
        lightingProgram.abandon();
        compositeProgram.abandon();

        gbufferFbo = 0;
        gbufferTextures.fill(0);
        gbufferDepth = 0;
        hdrFbo = 0;
        hdrTexture = 0;
        finalFbo = 0;
        finalTexture = 0;
        cubeVao = 0;
        cubeVbo = 0;
        cubeEbo = 0;
        instanceVbo = 0;
        instanceCapacity = 0;
        instanceCount = 0;
        emptyVao = 0;
        shadowFbo = 0;
        materialTexture = 0;
        fallbackShadowCube = 0;
        lightUbo = 0;
        debugLightsSsbo = 0;
        debugLightsCapacity = 0;
        shadowTextures.clear();
        cache.clear();
        gpuReady = false;
    }

    std::string shaderPath(const char* name) const {
        return shaderDirectory + "/" + name;
    }

    void createShaders() {
        gbufferProgram.loadFromFiles(shaderPath("gbuffer.vert"),
                                     shaderPath("gbuffer.frag"), shaderDirectory);
        shadowProgram.loadFromFiles(shaderPath("shadow.vert"),
                                    shaderPath("shadow.frag"), shaderDirectory);
        lightingProgram.loadFromFiles(shaderPath("fullscreen.vert"),
                                      shaderPath("lighting.frag"), shaderDirectory);
        compositeProgram.loadFromFiles(shaderPath("fullscreen.vert"),
                                       shaderPath("composite.frag"), shaderDirectory);

        lightingProgram.use();
        lightingProgram.setInt("uWorldPosition", 0);
        lightingProgram.setInt("uViewNormal", 1);
        lightingProgram.setInt("uAlbedo", 2);
        lightingProgram.setInt("uSurface", 3);
        for (int i = 0; i < kMaxLightsPerBatch; ++i) {
            const std::string uniformName =
                "uShadowMaps[" + std::to_string(i) + "]";
            const int location =
                glGetUniformLocation(lightingProgram.id(), uniformName.c_str());
            if (location >= 0) {
                glUniform1i(location, kShadowMapBindBase + i);
            }
        }

        compositeProgram.use();
        compositeProgram.setInt("uWorldPosition", 0);
        compositeProgram.setInt("uViewNormal", 1);
        compositeProgram.setInt("uAlbedo", 2);
        compositeProgram.setInt("uSurface", 3);
        compositeProgram.setInt("uHdr", 5);
        checkGlError("createShaders");
    }

    void createCubeGeometry() {
        const auto vertices = buildCubeVertices();
        const auto indices = buildCubeIndices();

        glGenVertexArrays(1, &cubeVao);
        glGenBuffers(1, &cubeVbo);
        glGenBuffers(1, &cubeEbo);
        glGenBuffers(1, &instanceVbo);
        glBindVertexArray(cubeVao);
        glBindBuffer(GL_ARRAY_BUFFER, cubeVbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(vertices)),
                     vertices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(indices)),
                     indices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(CpuVertex),
                              reinterpret_cast<void*>(offsetof(CpuVertex, position)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(CpuVertex),
                              reinterpret_cast<void*>(offsetof(CpuVertex, normal)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(CpuVertex),
                              reinterpret_cast<void*>(offsetof(CpuVertex, tangent)));
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(CpuVertex),
                              reinterpret_cast<void*>(offsetof(CpuVertex, uv)));

        glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);
        for (int column = 0; column < 4; ++column) {
            const unsigned int location = 4U + static_cast<unsigned int>(column);
            glEnableVertexAttribArray(location);
            glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE,
                                  sizeof(GPUInstance),
                                  reinterpret_cast<void*>(
                                      offsetof(GPUInstance, model) +
                                      sizeof(glm::vec4) * column));
            glVertexAttribDivisor(location, 1);
        }
        glEnableVertexAttribArray(8);
        glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, sizeof(GPUInstance),
                              reinterpret_cast<void*>(
                                  offsetof(GPUInstance, albedoRoughness)));
        glVertexAttribDivisor(8, 1);
        glEnableVertexAttribArray(9);
        glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, sizeof(GPUInstance),
                              reinterpret_cast<void*>(
                                  offsetof(GPUInstance, materialLayers)));
        glVertexAttribDivisor(9, 1);
        glEnableVertexAttribArray(10);
        glVertexAttribPointer(10, 4, GL_FLOAT, GL_FALSE, sizeof(GPUInstance),
                              reinterpret_cast<void*>(
                                  offsetof(GPUInstance, materialParams)));
        glVertexAttribDivisor(10, 1);

        glBindVertexArray(0);
        glGenVertexArrays(1, &emptyVao);
        checkGlError("createCubeGeometry");
    }

    void createMaterialTexture() {
        if (materialCpu.rgba.empty()) {
            throw std::runtime_error("CPU material texture data is empty");
        }
        glGenTextures(1, &materialTexture);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialTexture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, materialCpu.size,
                     materialCpu.size, materialCpu.layers, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, materialCpu.rgba.data());
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
        checkGlError("createMaterialTexture");
    }

    void createLightBuffers() {
        glGenBuffers(1, &lightUbo);
        glBindBuffer(GL_UNIFORM_BUFFER, lightUbo);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(LightBatchGPU), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, lightUbo);

        glGenBuffers(1, &debugLightsSsbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, debugLightsSsbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(DebugLightGPU), nullptr,
                     GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, debugLightsSsbo);
        debugLightsCapacity = 1;
        checkGlError("createLightBuffers");
    }

    void createRenderTargets() {
        glGenFramebuffers(1, &gbufferFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, gbufferFbo);
        createTexture2D(gbufferTextures[gbuffer::WorldPosition], settings.width,
                        settings.height, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_NEAREST);
        createTexture2D(gbufferTextures[gbuffer::ViewNormal], settings.width,
                        settings.height, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_NEAREST);
        createTexture2D(gbufferTextures[gbuffer::Albedo], settings.width,
                        settings.height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
                        GL_NEAREST);
        createTexture2D(gbufferTextures[gbuffer::Surface], settings.width,
                        settings.height, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_NEAREST);
        for (int i = 0; i < gbuffer::AttachmentCount; ++i) {
            glFramebufferTexture2D(GL_FRAMEBUFFER,
                                   GL_COLOR_ATTACHMENT0 + i,
                                   GL_TEXTURE_2D, gbufferTextures[i], 0);
        }
        glGenRenderbuffers(1, &gbufferDepth);
        glBindRenderbuffer(GL_RENDERBUFFER, gbufferDepth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                              settings.width, settings.height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, gbufferDepth);
        const std::array<GLenum, gbuffer::AttachmentCount> attachments = {
            GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
            GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
        glDrawBuffers(gbuffer::AttachmentCount, attachments.data());
        checkFramebuffer("G-Buffer");

        glGenFramebuffers(1, &hdrFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, hdrFbo);
        createTexture2D(hdrTexture, settings.width, settings.height, GL_RGBA16F,
                        GL_RGBA, GL_FLOAT, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, hdrTexture, 0);
        checkFramebuffer("HDR");

        glGenFramebuffers(1, &finalFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, finalFbo);
        createTexture2D(finalTexture, settings.width, settings.height, GL_RGBA8,
                        GL_RGBA, GL_UNSIGNED_BYTE, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, finalTexture, 0);
        checkFramebuffer("Final");
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        checkGlError("createRenderTargets");
    }

    void destroyRenderTargets() {
        for (unsigned int texture : gbufferTextures) {
            if (texture != 0) {
                glDeleteTextures(1, &texture);
            }
        }
        gbufferTextures.fill(0);
        if (gbufferDepth != 0) {
            glDeleteRenderbuffers(1, &gbufferDepth);
            gbufferDepth = 0;
        }
        if (hdrTexture != 0) {
            glDeleteTextures(1, &hdrTexture);
            hdrTexture = 0;
        }
        if (finalTexture != 0) {
            glDeleteTextures(1, &finalTexture);
            finalTexture = 0;
        }
        if (gbufferFbo != 0) {
            glDeleteFramebuffers(1, &gbufferFbo);
            gbufferFbo = 0;
        }
        if (hdrFbo != 0) {
            glDeleteFramebuffers(1, &hdrFbo);
            hdrFbo = 0;
        }
        if (finalFbo != 0) {
            glDeleteFramebuffers(1, &finalFbo);
            finalFbo = 0;
        }
    }

    void createShadowResources() {
        glGenFramebuffers(1, &shadowFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, shadowFbo);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);

        glGenTextures(1, &fallbackShadowCube);
        glBindTexture(GL_TEXTURE_CUBE_MAP, fallbackShadowCube);
        for (int face = 0; face < 6; ++face) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0,
                         GL_DEPTH_COMPONENT24, 1, 1, 0, GL_DEPTH_COMPONENT,
                         GL_FLOAT, nullptr);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        checkGlError("createShadowResources");
    }

    unsigned int createShadowCube() {
        unsigned int texture = 0;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_CUBE_MAP, texture);
        for (int face = 0; face < 6; ++face) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0,
                         GL_DEPTH_COMPONENT24, settings.shadowSize,
                         settings.shadowSize, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                         nullptr);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        return texture;
    }

    void uploadInstances(const Scene& scene) {
        std::vector<GPUInstance> instances;
        instances.reserve(scene.objects.size());
        for (const SceneObject& object : scene.objects) {
            GPUInstance instance{};
            instance.model = object.transform.matrix();
            instance.albedoRoughness =
                glm::vec4(object.material.albedo, object.material.roughness);
            instance.materialLayers =
                glm::vec4(static_cast<float>(object.material.baseLayer),
                          static_cast<float>(object.material.normalLayer),
                          static_cast<float>(object.material.roughMetalLayer),
                          static_cast<float>(object.id));
            instance.materialParams =
                glm::vec4(object.material.uvScale, object.material.metallic,
                          object.material.normalStrength, 0.0f);
            instances.push_back(instance);
        }
        glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);
        const std::size_t bytes = instances.size() * sizeof(GPUInstance);
        if (bytes > instanceCapacity) {
            glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(bytes),
                         instances.data(), GL_DYNAMIC_DRAW);
            instanceCapacity = bytes;
        } else if (bytes > 0) {
            glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(bytes),
                            instances.data());
        }
        instanceCount = instances.size();
    }

    void renderShadowMap(const Light& light, const Scene& scene) {
        const auto directions = cubeDirections();
        const auto ups = cubeUps();
        const glm::mat4 lightProjection = glm::perspective(
            glm::radians(90.0f), 1.0f, static_cast<float>(kShadowNear),
            std::max(light.range, 0.2f));

        glBindFramebuffer(GL_FRAMEBUFFER, shadowFbo);
        glViewport(0, 0, settings.shadowSize, settings.shadowSize);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);
        glDisable(GL_BLEND);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(2.5f, 4.0f);
        shadowProgram.use();
        shadowProgram.setMat4("uLightProjection", lightProjection);
        glBindVertexArray(cubeVao);

        const unsigned int texture = shadowTextures.at(light.id);
        for (int face = 0; face < 6; ++face) {
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                                   texture, 0);
            glClearDepth(1.0);
            glClear(GL_DEPTH_BUFFER_BIT);
            const glm::mat4 view = glm::lookAt(
                light.position, light.position + directions[face], ups[face]);
            shadowProgram.setMat4("uLightView", view);
            glDrawElementsInstanced(GL_TRIANGLES, kCubeVertexCount,
                                    GL_UNSIGNED_INT, nullptr,
                                    static_cast<GLsizei>(instanceCount));
        }
        glCullFace(GL_BACK);
        glDisable(GL_POLYGON_OFFSET_FILL);
        glBindVertexArray(0);
        ++stats.shadowMapsRenderedThisFrame;
        stats.shadowFacesRenderedThisFrame += 6;
    }

    void deleteShadowTexture(std::uint32_t lightId) {
        const auto it = shadowTextures.find(lightId);
        if (it != shadowTextures.end()) {
            glDeleteTextures(1, &it->second);
            shadowTextures.erase(it);
        }
    }

    void releaseAllShadowTextures() {
        for (const auto& [lightId, texture] : shadowTextures) {
            (void)lightId;
            glDeleteTextures(1, &texture);
        }
        shadowTextures.clear();
        cache.clear();
    }

    FrameSnapshot captureSnapshot(const Scene& scene, const Camera& camera,
                                  MaterialPresetId preset,
                                  std::uint64_t frameId) const {
        FrameSnapshot snapshot;
        snapshot.frameId = frameId;
        snapshot.sceneGeneration = scene.generation;
        snapshot.materialPreset = preset;
        snapshot.camera = camera;
        snapshot.lights.reserve(scene.lights.size());
        std::uint32_t validCount = 0;
        for (const Light& light : scene.lights) {
            LightSnapshot lightSnapshot;
            lightSnapshot.light = light;
            lightSnapshot.shadowVersion = ledger.version(light.id);
            lightSnapshot.cacheResident = cache.contains(light.id);
            if (lightSnapshot.cacheResident &&
                cache.version(light.id) == lightSnapshot.shadowVersion) {
                ++validCount;
            }
            snapshot.lights.push_back(lightSnapshot);
        }
        snapshot.cacheLruOrder = cache.lruOrder();
        (void)validCount;
        return snapshot;
    }

    void updateStats(const Scene& scene) {
        stats.totalLights = static_cast<std::uint32_t>(scene.lights.size());
        stats.residentShadowMaps = static_cast<std::uint32_t>(cache.size());
        stats.validShadowMaps = 0;
        for (const Light& light : scene.lights) {
            if (cache.contains(light.id) &&
                cache.version(light.id) == ledger.version(light.id)) {
                ++stats.validShadowMaps;
            }
        }
        stats.cacheCapacity = static_cast<std::uint32_t>(cache.capacity());
        stats.versionMigrations = ledger.migrations();
        stats.cacheBytes = static_cast<std::uint64_t>(cache.size()) * 6ULL *
                           static_cast<std::uint64_t>(settings.shadowSize) *
                           static_cast<std::uint64_t>(settings.shadowSize) * 4ULL;
    }

    void uploadDebugLights(const FrameSnapshot& snapshot) {
        std::vector<DebugLightGPU> lights;
        lights.reserve(snapshot.lights.size());
        for (const LightSnapshot& light : snapshot.lights) {
            DebugLightGPU value;
            value.positionRange = glm::vec4(light.light.position, light.light.range);
            value.colorIntensity =
                glm::vec4(light.light.color, light.light.intensity);
            value.meta = glm::ivec4(static_cast<int>(light.shadowVersion),
                                    light.cacheResident ? 1 : 0, 0,
                                    static_cast<int>(light.light.id));
            lights.push_back(value);
        }
        if (lights.empty()) {
            lights.push_back(DebugLightGPU{});
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, debugLightsSsbo);
        const std::size_t bytes = lights.size() * sizeof(DebugLightGPU);
        if (bytes > debugLightsCapacity * sizeof(DebugLightGPU)) {
            glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(bytes),
                         lights.data(), GL_DYNAMIC_DRAW);
            debugLightsCapacity = lights.size();
        } else {
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                            static_cast<GLsizeiptr>(bytes), lights.data());
        }
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, debugLightsSsbo);
    }

    void renderGBuffer(const Scene& scene, const Camera& camera,
                       MaterialPresetId preset) {
        glBindFramebuffer(GL_FRAMEBUFFER, gbufferFbo);
        glViewport(0, 0, settings.width, settings.height);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClearDepth(1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);
        glDisable(GL_BLEND);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        const float aspect = static_cast<float>(settings.width) /
                             static_cast<float>(std::max(settings.height, 1));
        gbufferProgram.use();
        gbufferProgram.setMat4("uView", camera.view());
        gbufferProgram.setMat4("uProjection", camera.projection(aspect));
        gbufferProgram.setInt("uMaterialPreset", static_cast<int>(preset));
        gbufferProgram.setInt("uMaterialTextures", 4);
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialTexture);

        uploadInstances(scene);
        glBindVertexArray(cubeVao);
        glDrawElementsInstanced(GL_TRIANGLES, kCubeVertexCount, GL_UNSIGNED_INT,
                                nullptr, static_cast<GLsizei>(instanceCount));
        glBindVertexArray(0);
    }

    void renderLighting(const FrameSnapshot& snapshot) {
        glBindFramebuffer(GL_FRAMEBUFFER, hdrFbo);
        glViewport(0, 0, settings.width, settings.height);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glEnable(GL_BLEND);
        glBlendEquation(GL_FUNC_ADD);
        glBlendFunc(GL_ONE, GL_ONE);

        for (int i = 0; i < gbuffer::AttachmentCount; ++i) {
            glActiveTexture(GL_TEXTURE0 + static_cast<int>(i));
            glBindTexture(GL_TEXTURE_2D, gbufferTextures[i]);
        }

        lightingProgram.use();
        lightingProgram.setMat4("uInverseView", glm::inverse(snapshot.camera.view()));
        lightingProgram.setVec3("uCameraPosition", snapshot.camera.position());
        lightingProgram.setFloat("uShadowNear", static_cast<float>(kShadowNear));
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, lightUbo);
        glBindVertexArray(emptyVao);

        LightBatchGPU batch{};
        int batchCount = 0;
        for (const LightSnapshot& light : snapshot.lights) {
            const int slot = batchCount;
            batch.positionRange[slot] =
                glm::vec4(light.light.position, light.light.range);
            batch.colorIntensity[slot] =
                glm::vec4(light.light.color, light.light.intensity);
            int mapVersion = 0;
            bool shadowEnabled = false;
            const auto textureIt = shadowTextures.find(light.light.id);
            unsigned int texture = fallbackShadowCube;
            if (textureIt != shadowTextures.end()) {
                texture = textureIt->second;
                mapVersion = static_cast<int>(cache.version(light.light.id));
                shadowEnabled = true;
            }
            batch.meta[slot] = glm::ivec4(
                static_cast<int>(light.shadowVersion), mapVersion,
                shadowEnabled ? 1 : 0, static_cast<int>(light.light.id));
            glActiveTexture(GL_TEXTURE0 + kShadowMapBindBase + slot);
            glBindTexture(GL_TEXTURE_CUBE_MAP, texture);
            ++batchCount;

            if (batchCount == kMaxLightsPerBatch) {
                batch.counts = glm::ivec4(batchCount, 0, 0, 0);
                glBindBuffer(GL_UNIFORM_BUFFER, lightUbo);
                glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(LightBatchGPU), &batch);
                glBindBufferBase(GL_UNIFORM_BUFFER, 0, lightUbo);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                batch = LightBatchGPU{};
                batchCount = 0;
            }
        }
        if (batchCount > 0) {
            batch.counts = glm::ivec4(batchCount, 0, 0, 0);
            glBindBuffer(GL_UNIFORM_BUFFER, lightUbo);
            glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(LightBatchGPU), &batch);
            glBindBufferBase(GL_UNIFORM_BUFFER, 0, lightUbo);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        glBindVertexArray(0);
        glDisable(GL_BLEND);
    }

    void renderComposite(const FrameSnapshot& snapshot, DebugView debugView) {
        glBindFramebuffer(GL_FRAMEBUFFER, finalFbo);
        glViewport(0, 0, settings.width, settings.height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        for (int i = 0; i < gbuffer::AttachmentCount; ++i) {
            glActiveTexture(GL_TEXTURE0 + static_cast<int>(i));
            glBindTexture(GL_TEXTURE_2D, gbufferTextures[i]);
        }
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, hdrTexture);

        uploadDebugLights(snapshot);
        compositeProgram.use();
        compositeProgram.setInt("uDebugView", static_cast<int>(debugView));
        compositeProgram.setInt("uDebugLightCount",
                                static_cast<int>(snapshot.lights.size()));
        const float occupancy = settings.shadowCacheCapacity == 0
            ? 0.0f
            : static_cast<float>(cache.size()) /
              static_cast<float>(settings.shadowCacheCapacity);
        compositeProgram.setFloat("uCacheOccupancy", occupancy);
        compositeProgram.setFloat("uExposure", 1.35f);
        glBindVertexArray(emptyVao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);
    }

    void renderRecoveredShadowMaps(const Scene& scene, const FrameSnapshot& snapshot) {
        uploadInstances(scene);
        for (std::uint32_t lightId : snapshot.cacheLruOrder) {
            const auto found = std::find_if(
                snapshot.lights.begin(), snapshot.lights.end(),
                [lightId](const LightSnapshot& light) {
                    return light.light.id == lightId;
                });
            if (found == snapshot.lights.end() || !found->cacheResident) {
                continue;
            }
            const CacheAcquireResult result = cache.acquire(lightId);
            if (result.evictedLightId.has_value()) {
                deleteShadowTexture(*result.evictedLightId);
            }
            if (!result.created) {
                continue;
            }
            shadowTextures[lightId] = createShadowCube();
            renderShadowMap(found->light, scene);
            cache.setVersion(lightId, found->shadowVersion);
        }
    }
};

DeferredRenderer::DeferredRenderer() : impl_(std::make_unique<Impl>()) {}
DeferredRenderer::~DeferredRenderer() = default;
DeferredRenderer::DeferredRenderer(DeferredRenderer&&) noexcept = default;
DeferredRenderer& DeferredRenderer::operator=(DeferredRenderer&&) noexcept = default;

void DeferredRenderer::initialize(const Scene& scene, const RenderSettings& settings,
                                  const std::string& shaderDirectory) {
    RenderThreadGuard::bindCurrentThread();
    impl_->settings = settings;
    impl_->settings.shadowSize = std::max(settings.shadowSize, 32);
    impl_->settings.shadowCacheCapacity =
        std::max<std::size_t>(settings.shadowCacheCapacity, 0);
    impl_->shaderDirectory = shaderDirectory;
    impl_->materialCpu = generateMaterialTextures(
        scene.seed ^ (scene.generation * 0x9e3779b97f4a7c15ULL));
    impl_->ledger.clear();
    impl_->ledger.initialize(scene.lights);
    impl_->cache = ShadowCachePolicy(impl_->settings.shadowCacheCapacity);
    impl_->lastSnapshot = FrameSnapshot{};
    impl_->stats = ShadowFrameStats{};
    impl_->createGpuObjects();
}

void DeferredRenderer::shutdown() {
    if (impl_ && impl_->gpuReady) {
        RenderThreadGuard::requireRenderThread("DeferredRenderer::shutdown");
        impl_->releaseGpu(true);
    }
}

void DeferredRenderer::invalidateContextWithoutDelete() {
    RenderThreadGuard::bindCurrentThread();
    impl_->releaseGpu(false);
}

void DeferredRenderer::recoverFromSnapshot(const Scene& scene,
                                           const FrameSnapshot& snapshot,
                                           const RenderSettings& settings,
                                           const std::string& shaderDirectory) {
    RenderThreadGuard::bindCurrentThread();
    impl_->settings = settings;
    impl_->settings.shadowSize = std::max(settings.shadowSize, 32);
    impl_->settings.shadowCacheCapacity =
        std::max<std::size_t>(settings.shadowCacheCapacity, 0);
    impl_->shaderDirectory = shaderDirectory;
    impl_->cache = ShadowCachePolicy(impl_->settings.shadowCacheCapacity);
    impl_->shadowTextures.clear();
    impl_->createGpuObjects();
    impl_->lastSnapshot = snapshot;
    impl_->renderRecoveredShadowMaps(scene, snapshot);
    impl_->stats = ShadowFrameStats{};
    impl_->stats.versionMigrationsThisFrame = 0;
    impl_->updateStats(scene);
}

void DeferredRenderer::resize(const Scene& scene, int width, int height) {
    (void)scene;
    impl_->requireGpu("DeferredRenderer::resize");
    const int safeWidth = std::max(width, 1);
    const int safeHeight = std::max(height, 1);
    if (safeWidth == impl_->settings.width &&
        safeHeight == impl_->settings.height) {
        return;
    }
    impl_->settings.width = safeWidth;
    impl_->settings.height = safeHeight;
    impl_->settings.shadowSize = chooseShadowSize(safeWidth, safeHeight);
    impl_->destroyRenderTargets();
    impl_->createRenderTargets();
    impl_->releaseAllShadowTextures();
    impl_->stats.shadowMapsRenderedThisFrame = 0;
    impl_->stats.shadowFacesRenderedThisFrame = 0;
    impl_->stats.cacheEvictionsThisFrame = 0;
}

void DeferredRenderer::switchScene(const Scene& scene) {
    impl_->requireGpu("DeferredRenderer::switchScene");
    impl_->releaseAllShadowTextures();
    impl_->ledger.clear();
    impl_->ledger.initialize(scene.lights);
    impl_->lastSnapshot = FrameSnapshot{};
    impl_->stats = ShadowFrameStats{};
    glDeleteTextures(1, &impl_->materialTexture);
    impl_->materialTexture = 0;
    impl_->materialCpu = generateMaterialTextures(
        scene.seed ^ (scene.generation * 0x9e3779b97f4a7c15ULL));
    impl_->createMaterialTexture();
}

FrameSnapshot DeferredRenderer::prepareFrame(
    const Scene& scene, const Camera& camera, MaterialPresetId preset,
    const std::vector<std::uint32_t>& dirtyLights, std::uint64_t frameId) {
    impl_->requireGpu("DeferredRenderer::prepareFrame");
    impl_->stats = ShadowFrameStats{};
    impl_->stats.cacheCapacity =
        static_cast<std::uint32_t>(impl_->cache.capacity());

    std::unordered_set<std::uint32_t> dirtySet;
    for (std::uint32_t lightId : dirtyLights) {
        if (findLight(scene, lightId) != nullptr && dirtySet.insert(lightId).second) {
            impl_->ledger.bump(lightId);
            ++impl_->stats.versionMigrationsThisFrame;
            impl_->stats.dirtyLightIds.push_back(lightId);
        }
    }
    std::sort(impl_->stats.dirtyLightIds.begin(),
              impl_->stats.dirtyLightIds.end());

    std::vector<const Light*> selected;
    selected.reserve(scene.lights.size());
    for (const Light& light : scene.lights) {
        selected.push_back(&light);
    }
    if (impl_->cache.capacity() < selected.size()) {
        const glm::vec3 cameraPosition = camera.position();
        std::sort(selected.begin(), selected.end(),
                  [&](const Light* a, const Light* b) {
                      const glm::vec3 deltaA = a->position - cameraPosition;
                      const glm::vec3 deltaB = b->position - cameraPosition;
                      const float da = glm::dot(deltaA, deltaA);
                      const float db = glm::dot(deltaB, deltaB);
                      return da == db ? a->id < b->id : da < db;
                  });
        selected.resize(impl_->cache.capacity());
    }

    if (!selected.empty()) {
        impl_->uploadInstances(scene);
    }
    for (const Light* light : selected) {
        const CacheAcquireResult result = impl_->cache.acquire(light->id);
        if (result.evictedLightId.has_value()) {
            impl_->deleteShadowTexture(*result.evictedLightId);
            ++impl_->stats.cacheEvictionsThisFrame;
        }
        if (result.created) {
            impl_->shadowTextures[light->id] = impl_->createShadowCube();
        }
        const std::uint64_t currentVersion = impl_->ledger.version(light->id);
        const bool needsRender =
            result.created || dirtySet.find(light->id) != dirtySet.end() ||
            impl_->cache.version(light->id) != currentVersion;
        if (needsRender) {
            const auto textureIt = impl_->shadowTextures.find(light->id);
            if (textureIt == impl_->shadowTextures.end()) {
                impl_->shadowTextures[light->id] = impl_->createShadowCube();
            }
            impl_->renderShadowMap(*light, scene);
            impl_->cache.setVersion(light->id, currentVersion);
        }
    }

    FrameSnapshot snapshot =
        impl_->captureSnapshot(scene, camera, preset, frameId);
    impl_->lastSnapshot = snapshot;
    impl_->updateStats(scene);
    return snapshot;
}

void DeferredRenderer::render(const Scene& scene, const FrameSnapshot& snapshot,
                              DebugView debugView) {
    impl_->requireGpu("DeferredRenderer::render");
    impl_->renderGBuffer(scene, snapshot.camera, snapshot.materialPreset);
    impl_->renderLighting(snapshot);
    impl_->renderComposite(snapshot, debugView);
    checkGlError("DeferredRenderer::render");
}

void DeferredRenderer::blitFinalToDefaultFramebuffer() const {
    impl_->requireGpu("DeferredRenderer::blitFinalToDefaultFramebuffer");
    glBindFramebuffer(GL_READ_FRAMEBUFFER, impl_->finalFbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, impl_->settings.width, impl_->settings.height,
                      0, 0, impl_->settings.width, impl_->settings.height,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

std::vector<std::uint8_t> DeferredRenderer::readFinalPixels() const {
    impl_->requireGpu("DeferredRenderer::readFinalPixels");
    std::vector<std::uint8_t> pixels(
        static_cast<std::size_t>(impl_->settings.width) *
        static_cast<std::size_t>(impl_->settings.height) * 4U);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, impl_->finalFbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, impl_->settings.width, impl_->settings.height, GL_RGBA,
                 GL_UNSIGNED_BYTE, pixels.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    checkGlError("DeferredRenderer::readFinalPixels");
    return pixels;
}

const ShadowFrameStats& DeferredRenderer::stats() const {
    return impl_->stats;
}

const FrameSnapshot& DeferredRenderer::lastSnapshot() const {
    return impl_->lastSnapshot;
}

const RenderSettings& DeferredRenderer::settings() const {
    return impl_->settings;
}

const ShadowVersionLedger& DeferredRenderer::versionLedger() const {
    return impl_->ledger;
}

bool DeferredRenderer::hasGpuResources() const {
    return impl_->gpuReady;
}

bool DeferredRenderer::releaseLightShadow(std::uint32_t lightId) {
    impl_->requireGpu("DeferredRenderer::releaseLightShadow");
    const bool released = impl_->cache.release(lightId);
    impl_->deleteShadowTexture(lightId);
    impl_->ledger.remove(lightId);
    impl_->lastSnapshot.lights.erase(
        std::remove_if(impl_->lastSnapshot.lights.begin(),
                       impl_->lastSnapshot.lights.end(),
                       [lightId](const LightSnapshot& light) {
                           return light.light.id == lightId;
                       }),
        impl_->lastSnapshot.lights.end());
    impl_->lastSnapshot.cacheLruOrder.erase(
        std::remove(impl_->lastSnapshot.cacheLruOrder.begin(),
                    impl_->lastSnapshot.cacheLruOrder.end(), lightId),
        impl_->lastSnapshot.cacheLruOrder.end());
    return released;
}

void DeferredRenderer::releaseAllShadowMaps() {
    impl_->requireGpu("DeferredRenderer::releaseAllShadowMaps");
    impl_->releaseAllShadowTextures();
    for (LightSnapshot& light : impl_->lastSnapshot.lights) {
        light.cacheResident = false;
    }
    impl_->lastSnapshot.cacheLruOrder.clear();
    impl_->stats.residentShadowMaps = 0;
    impl_->stats.validShadowMaps = 0;
    impl_->stats.cacheBytes = 0;
}

unsigned int DeferredRenderer::shadowTextureForLight(
    std::uint32_t lightId) const {
    const auto it = impl_->shadowTextures.find(lightId);
    return it == impl_->shadowTextures.end() ? 0 : it->second;
}

int DeferredRenderer::shadowCacheResidentCount() const {
    return static_cast<int>(impl_->cache.size());
}

} // namespace indoor
