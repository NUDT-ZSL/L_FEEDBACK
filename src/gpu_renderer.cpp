#define GLFW_INCLUDE_NONE
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "gpu_renderer.h"

#include "error.h"
#include "image_io.h"
#include "lights.h"
#include "offline.h"
#include "scene.h"
#include "sorting.h"

#include <array>
#include <algorithm>
#include <cstddef>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef DF_SHADER_DIR
#define DF_SHADER_DIR "shaders"
#endif

namespace df {
namespace {

constexpr int kMaxGpuLights = 32;

struct PackedGpuLight {
    float position[4]{};
    float colorIntensity[4]{};
    float directionType[4]{};
};

struct alignas(16) LightUniformData {
    std::int32_t lightCount = 0;
    std::int32_t maxLights = kMaxGpuLights;
    std::int32_t padding[2]{};
    PackedGpuLight lights[kMaxGpuLights]{};
};

static_assert(sizeof(PackedGpuLight) == 48, "std140 light packing");
static_assert(sizeof(LightUniformData) == 16 + 48 * kMaxGpuLights,
              "std140 light block packing");

std::string readTextFile(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw AppError(ErrorCode::E_IO, "cannot open file: " + path);
    }
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

GLuint compileShader(GLenum type, const std::string& source, const std::string& label) {
    const GLuint shader = glCreateShader(type);
    const char* text = source.c_str();
    glShaderSource(shader, 1, &text, nullptr);
    glCompileShader(shader);
    GLint success = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success != GL_TRUE) {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        glDeleteShader(shader);
        throw AppError(ErrorCode::E_SHADER_COMPILE, label + ": " + log);
    }
    return shader;
}

GLuint makeProgram(const std::string& vertexPath, const std::string& fragmentPath,
                   const std::string& label) {
    const std::string vertexSource = readTextFile(vertexPath);
    const std::string fragmentSource = readTextFile(fragmentPath);
    const GLuint vertex = compileShader(GL_VERTEX_SHADER, vertexSource, label + ".vert");
    GLuint fragment = 0;
    try {
        fragment = compileShader(GL_FRAGMENT_SHADER, fragmentSource, label + ".frag");
    } catch (...) {
        glDeleteShader(vertex);
        throw;
    }
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    GLint success = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success != GL_TRUE) {
        GLint length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
        glGetProgramInfoLog(program, length, nullptr, log.data());
        glDeleteProgram(program);
        throw AppError(ErrorCode::E_SHADER_COMPILE, label + " link: " + log);
    }
    const GLuint block = glGetUniformBlockIndex(program, "LightBlock");
    if (block != GL_INVALID_INDEX) {
        glUniformBlockBinding(program, block, 0);
    }
    return program;
}

GLuint createTexture2D(GLenum internalFormat, GLenum format, GLenum type,
                       int width, int height) {
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(internalFormat), width, height, 0,
                 format, type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return texture;
}

void checkFramebuffer(GLuint fbo, ErrorCode code, const char* label) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::ostringstream message;
        message << label << " incomplete (0x" << std::hex << status << ")";
        throw AppError(code, message.str());
    }
}

struct Vertex {
    Vec3 position;
    Vec3 normal;
};

struct Mesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei count = 0;
    bool indexed = false;
};

Mesh uploadMesh(const std::vector<Vertex>& vertices,
                const std::vector<std::uint32_t>& indices = {}) {
    Mesh mesh;
    glGenVertexArrays(1, &mesh.vao);
    glGenBuffers(1, &mesh.vbo);
    glBindVertexArray(mesh.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)),
                 vertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, position)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, normal)));
    if (!indices.empty()) {
        glGenBuffers(1, &mesh.ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(indices.size() * sizeof(std::uint32_t)),
                     indices.data(), GL_STATIC_DRAW);
        mesh.count = static_cast<GLsizei>(indices.size());
        mesh.indexed = true;
    } else {
        mesh.count = static_cast<GLsizei>(vertices.size());
    }
    glBindVertexArray(0);
    return mesh;
}

Mesh makePlaneMesh() {
    const float extent = 42.0f;
    std::vector<Vertex> vertices = {
        {{-extent, 0.0f, -extent}, {0.0f, 1.0f, 0.0f}},
        {{extent, 0.0f, -extent}, {0.0f, 1.0f, 0.0f}},
        {{extent, 0.0f, extent}, {0.0f, 1.0f, 0.0f}},
        {{-extent, 0.0f, -extent}, {0.0f, 1.0f, 0.0f}},
        {{extent, 0.0f, extent}, {0.0f, 1.0f, 0.0f}},
        {{-extent, 0.0f, extent}, {0.0f, 1.0f, 0.0f}},
    };
    return uploadMesh(vertices);
}

Mesh makeCubeMesh() {
    std::vector<Vertex> vertices = {
        {{-0.5f, -0.5f, -0.5f}, {0, 0, -1}}, {{0.5f, -0.5f, -0.5f}, {0, 0, -1}},
        {{0.5f, 0.5f, -0.5f}, {0, 0, -1}},   {{-0.5f, 0.5f, -0.5f}, {0, 0, -1}},
        {{-0.5f, -0.5f, 0.5f}, {0, 0, 1}},   {{0.5f, -0.5f, 0.5f}, {0, 0, 1}},
        {{0.5f, 0.5f, 0.5f}, {0, 0, 1}},     {{-0.5f, 0.5f, 0.5f}, {0, 0, 1}},
        {{-0.5f, -0.5f, -0.5f}, {-1, 0, 0}}, {{-0.5f, -0.5f, 0.5f}, {-1, 0, 0}},
        {{-0.5f, 0.5f, 0.5f}, {-1, 0, 0}},   {{-0.5f, 0.5f, -0.5f}, {-1, 0, 0}},
        {{0.5f, -0.5f, -0.5f}, {1, 0, 0}},   {{0.5f, -0.5f, 0.5f}, {1, 0, 0}},
        {{0.5f, 0.5f, 0.5f}, {1, 0, 0}},     {{0.5f, 0.5f, -0.5f}, {1, 0, 0}},
        {{-0.5f, -0.5f, -0.5f}, {0, -1, 0}}, {{0.5f, -0.5f, -0.5f}, {0, -1, 0}},
        {{0.5f, -0.5f, 0.5f}, {0, -1, 0}},   {{-0.5f, -0.5f, 0.5f}, {0, -1, 0}},
        {{-0.5f, 0.5f, -0.5f}, {0, 1, 0}},   {{0.5f, 0.5f, -0.5f}, {0, 1, 0}},
        {{0.5f, 0.5f, 0.5f}, {0, 1, 0}},     {{-0.5f, 0.5f, 0.5f}, {0, 1, 0}},
    };
    std::vector<std::uint32_t> indices;
    for (std::uint32_t face = 0; face < 6; ++face) {
        const std::uint32_t base = face * 4;
        indices.insert(indices.end(), {base, base + 1, base + 2,
                                       base, base + 2, base + 3});
    }
    return uploadMesh(vertices, indices);
}

Mesh makeSphereMesh() {
    constexpr int rings = 24;
    constexpr int segments = 40;
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    for (int ring = 0; ring <= rings; ++ring) {
        const float v = static_cast<float>(ring) / rings;
        const float phi = v * kPi;
        for (int segment = 0; segment <= segments; ++segment) {
            const float u = static_cast<float>(segment) / segments;
            const float theta = u * 2.0f * kPi;
            Vec3 normal{std::sin(phi) * std::cos(theta), std::cos(phi),
                        std::sin(phi) * std::sin(theta)};
            vertices.push_back({normal, normal});
        }
    }
    for (int ring = 0; ring < rings; ++ring) {
        for (int segment = 0; segment < segments; ++segment) {
            const std::uint32_t a =
                static_cast<std::uint32_t>(ring * (segments + 1) + segment);
            const std::uint32_t b = a + static_cast<std::uint32_t>(segments + 1);
            indices.insert(indices.end(), {a, b, a + 1, b, b + 1, a + 1});
        }
    }
    return uploadMesh(vertices, indices);
}

void setInt(GLuint program, const char* name, int value) {
    glUniform1i(glGetUniformLocation(program, name), value);
}

void setFloat(GLuint program, const char* name, float value) {
    glUniform1f(glGetUniformLocation(program, name), value);
}

void setVec3(GLuint program, const char* name, const Vec3& value) {
    glUniform3f(glGetUniformLocation(program, name), value.x, value.y, value.z);
}

void setMat4(GLuint program, const char* name, const Mat4& value) {
    glUniformMatrix4fv(glGetUniformLocation(program, name), 1, GL_FALSE, value.m);
}

class GpuRenderer {
public:
    GpuRenderer(const RenderConfig& config, bool visible)
        : config_(config) {
        try {
            initWindow(visible);
            initResources();
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~GpuRenderer() { cleanup(); }

    FrameOutput render(const Scene& scene, bool offline, bool collectGBuffer) {
        const auto start = std::chrono::steady_clock::now();
        const int steps = effectiveFogSteps(config_, offline);
        lightSnapshot_.sync(scene.lights);
        const std::uint64_t lightingBefore = uploadStats_.lightingUploads;
        const std::uint64_t fogBefore = uploadStats_.fogUploads;
        ensureLightingUpload(lightSnapshot_, uploadStats_);
        ensureFogUpload(lightSnapshot_, uploadStats_);
        if (uploadStats_.lightingUploads != lightingBefore ||
            uploadStats_.fogUploads != fogBefore) {
            uploadLights(scene);
        }

        FrameOutput output;
        output.width = config_.width;
        output.height = config_.height;
        output.camera = scene.camera;
        output.transparent = sortTransparent(scene, config_.transparent, scene.camera);
        output.colorRgb.resize(static_cast<std::size_t>(config_.width) * config_.height * 3u);
        output.linearDepth.assign(static_cast<std::size_t>(config_.width) * config_.height,
                                  scene.camera.farPlane);

        renderGeometry(scene);
        renderDeferred(scene);
        copySceneToFinal();
        if (config_.volumetric.enabled) {
            renderVolumetric(scene, steps);
        }
        const int transparentDrawn = renderTransparent(scene, output.transparent);
        renderPresent();

        if (collectGBuffer) {
            captureGBuffer(output);
        }
        if (collectGBuffer) {
            captureColorAndDepth(output);
        }

        output.stats.backend = "opengl-4.5";
        output.stats.fogSteps = steps;
        output.stats.activeLights =
            std::min(static_cast<int>(scene.lights.size()), config_.lights.maxLights);
        output.stats.transparentDrawn = transparentDrawn;
        output.stats.transparentDepthWrites = 0;
        output.stats.lightVersion = lightSnapshot_.version();
        output.stats.lightingUploads = uploadStats_.lightingUploads;
        output.stats.fogUploads = uploadStats_.fogUploads;
        const auto end = std::chrono::steady_clock::now();
        output.stats.cpuMilliseconds =
            std::chrono::duration<double, std::milli>(end - start).count();
        const auto* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        const auto* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
        output.stats.gpuRenderer = renderer ? renderer : "unknown";
        output.stats.gpuVersion = version ? version : "unknown";
        checkGl("render");
        return output;
    }

    void presentToDefault() {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, displayFbo_);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0, 0, config_.width, config_.height,
                          0, 0, config_.width, config_.height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    bool shouldClose() const { return glfwWindowShouldClose(window_) == GLFW_TRUE; }

private:
    void initWindow(bool visible) {
        glfwSetErrorCallback([](int code, const char* description) {
            std::cerr << "GLFW error " << code << ": "
                      << (description ? description : "unknown") << '\n';
        });
        if (glfwInit() != GLFW_TRUE) {
            throw AppError(ErrorCode::E_GL_CONTEXT, "glfwInit failed");
        }
        glfwInitialized_ = true;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window_ = glfwCreateWindow(config_.width, config_.height,
                                   "Deferred Fog Validation", nullptr, nullptr);
        if (!window_) {
            throw AppError(ErrorCode::E_GL_CONTEXT, "failed to create OpenGL 4.5 window");
        }
        glfwMakeContextCurrent(window_);
        if (gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress)) == 0) {
            throw AppError(ErrorCode::E_GL_CONTEXT, "failed to load OpenGL 4.5 functions");
        }
        if (!GLAD_GL_VERSION_4_5) {
            throw AppError(ErrorCode::E_GL_CONTEXT,
                           "OpenGL 4.5 is required; driver does not expose it");
        }
        glfwSwapInterval(1);
    }

    void initResources() {
        gPosition_ = createTexture2D(GL_RGBA32F, GL_RGBA, GL_FLOAT, config_.width, config_.height);
        gNormal_ = createTexture2D(GL_RGBA32F, GL_RGBA, GL_FLOAT, config_.width, config_.height);
        gAlbedo_ = createTexture2D(GL_RGBA32F, GL_RGBA, GL_FLOAT, config_.width, config_.height);
        gMaterial_ = createTexture2D(GL_RGBA32F, GL_RGBA, GL_FLOAT, config_.width, config_.height);
        depthTexture_ = createTexture2D(GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT,
                                        config_.width, config_.height);
        sceneColor_ = createTexture2D(GL_RGBA16F, GL_RGBA, GL_FLOAT, config_.width, config_.height);
        finalColor_ = createTexture2D(GL_RGBA16F, GL_RGBA, GL_FLOAT, config_.width, config_.height);
        displayColor_ = createTexture2D(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
                                        config_.width, config_.height);

        glGenFramebuffers(1, &gFbo_);
        glBindFramebuffer(GL_FRAMEBUFFER, gFbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPosition_, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gNormal_, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gAlbedo_, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, gMaterial_, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture_, 0);
        const GLenum gDrawBuffers[4] = {
            GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
            GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
        glDrawBuffers(4, gDrawBuffers);
        checkFramebuffer(gFbo_, ErrorCode::E_GBUFFER_FAIL, "G-Buffer FBO");

        glGenFramebuffers(1, &sceneFbo_);
        glBindFramebuffer(GL_FRAMEBUFFER, sceneFbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sceneColor_, 0);
        checkFramebuffer(sceneFbo_, ErrorCode::E_RENDER_FAIL, "lighting FBO");

        glGenFramebuffers(1, &finalFbo_);
        glBindFramebuffer(GL_FRAMEBUFFER, finalFbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, finalColor_, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture_, 0);
        checkFramebuffer(finalFbo_, ErrorCode::E_RENDER_FAIL, "composite FBO");

        glGenFramebuffers(1, &displayFbo_);
        glBindFramebuffer(GL_FRAMEBUFFER, displayFbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, displayColor_, 0);
        checkFramebuffer(displayFbo_, ErrorCode::E_RENDER_FAIL, "display FBO");

        const std::string shaderDir = DF_SHADER_DIR;
        gbufferProgram_ = makeProgram(shaderDir + "/gbuffer.vert", shaderDir + "/gbuffer.frag",
                                      "gbuffer");
        deferredProgram_ = makeProgram(shaderDir + "/fullscreen.vert", shaderDir + "/deferred.frag",
                                       "deferred");
        fogProgram_ = makeProgram(shaderDir + "/fullscreen.vert", shaderDir + "/volumetric.frag",
                                  "volumetric");
        transparentProgram_ = makeProgram(shaderDir + "/gbuffer.vert",
                                          shaderDir + "/transparent.frag", "transparent");
        presentProgram_ = makeProgram(shaderDir + "/fullscreen.vert", shaderDir + "/present.frag",
                                      "present");

        glGenVertexArrays(1, &fullscreenVao_);
        glGenBuffers(1, &fullscreenVbo_);
        const float quad[] = {
            -1, -1, 0, 0,  1, -1, 1, 0,  -1, 1, 0, 1,
            -1,  1, 0, 1,  1, -1, 1, 0,   1, 1, 1, 1,
        };
        glBindVertexArray(fullscreenVao_);
        glBindBuffer(GL_ARRAY_BUFFER, fullscreenVbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              reinterpret_cast<void*>(2 * sizeof(float)));
        glBindVertexArray(0);

        planeMesh_ = makePlaneMesh();
        cubeMesh_ = makeCubeMesh();
        sphereMesh_ = makeSphereMesh();

        glGenBuffers(1, &lightUbo_);
        glBindBuffer(GL_UNIFORM_BUFFER, lightUbo_);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(LightUniformData), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, lightUbo_);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void uploadLights(const Scene& scene) {
        LightUniformData data;
        data.lightCount = std::min(static_cast<int>(scene.lights.size()),
                                   std::min(config_.lights.maxLights, kMaxGpuLights));
        data.maxLights = data.lightCount;
        for (int i = 0; i < data.lightCount; ++i) {
            const Light& source = scene.lights[static_cast<std::size_t>(i)];
            PackedGpuLight& target = data.lights[i];
            target.position[0] = source.position.x;
            target.position[1] = source.position.y;
            target.position[2] = source.position.z;
            target.position[3] = 1.0f;
            target.colorIntensity[0] = source.color.x;
            target.colorIntensity[1] = source.color.y;
            target.colorIntensity[2] = source.color.z;
            target.colorIntensity[3] = source.intensity;
            target.directionType[0] = source.direction.x;
            target.directionType[1] = source.direction.y;
            target.directionType[2] = source.direction.z;
            target.directionType[3] = source.type == LightType::Directional ? 1.0f : 0.0f;
        }
        glBindBuffer(GL_UNIFORM_BUFFER, lightUbo_);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(data), &data);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }

    void drawMesh(const Mesh& mesh) {
        glBindVertexArray(mesh.vao);
        if (mesh.indexed) {
            glDrawElements(GL_TRIANGLES, mesh.count, GL_UNSIGNED_INT, nullptr);
        } else {
            glDrawArrays(GL_TRIANGLES, 0, mesh.count);
        }
    }

    void setPrimitiveModel(GLuint program, const Primitive& primitive,
                           const Mat4& view, const Mat4& projection) {
        Mat4 model = Mat4::translation(primitive.center);
        if (primitive.type == PrimitiveType::Box) {
            model = model * Mat4::scaling(primitive.halfExtents * 2.0f);
        } else if (primitive.type == PrimitiveType::Sphere) {
            model = model * Mat4::scaling(Vec3{primitive.radius});
        }
        setMat4(program, "uModel", model);
        setMat4(program, "uView", view);
        setMat4(program, "uProjection", projection);
    }

    const Mesh& meshFor(const Primitive& primitive) const {
        if (primitive.type == PrimitiveType::Plane) return planeMesh_;
        if (primitive.type == PrimitiveType::Box) return cubeMesh_;
        return sphereMesh_;
    }

    void renderGeometry(const Scene& scene) {
        glBindFramebuffer(GL_FRAMEBUFFER, gFbo_);
        glViewport(0, 0, config_.width, config_.height);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClearDepth(1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(gbufferProgram_);

        const float aspect = static_cast<float>(config_.width) / config_.height;
        const Mat4 projection = Mat4::perspective(
            scene.camera.fovYDegrees * kPi / 180.0f, aspect,
            scene.camera.nearPlane, scene.camera.farPlane);
        const Mat4 view = Mat4::lookAt(scene.camera.position, scene.camera.target,
                                       scene.camera.up);
        for (const Primitive& primitive : scene.opaque) {
            setPrimitiveModel(gbufferProgram_, primitive, view, projection);
            setVec3(gbufferProgram_, "uAlbedo", primitive.material.albedo);
            setFloat(gbufferProgram_, "uRoughness", primitive.material.roughness);
            setFloat(gbufferProgram_, "uMetallic", primitive.material.metallic);
            setFloat(gbufferProgram_, "uObjectId", static_cast<float>(primitive.id));
            drawMesh(meshFor(primitive));
        }
    }

    void bindGBufferTextures(GLuint program) {
        const std::array<GLuint, 4> textures = {gPosition_, gNormal_, gAlbedo_, gMaterial_};
        for (std::size_t i = 0; i < textures.size(); ++i) {
            glActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(i));
            glBindTexture(GL_TEXTURE_2D, textures[i]);
        }
        setInt(program, "uGPosition", 0);
        setInt(program, "uGNormal", 1);
        setInt(program, "uGAlbedo", 2);
        setInt(program, "uGMaterial", 3);
    }

    void renderDeferred(const Scene& scene) {
        glBindFramebuffer(GL_FRAMEBUFFER, sceneFbo_);
        glViewport(0, 0, config_.width, config_.height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glUseProgram(deferredProgram_);
        bindGBufferTextures(deferredProgram_);
        setVec3(deferredProgram_, "uCameraPosition", scene.camera.position);
        glBindVertexArray(fullscreenVao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    void copySceneToFinal() {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, sceneFbo_);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, finalFbo_);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0, 0, config_.width, config_.height,
                          0, 0, config_.width, config_.height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }

    void renderVolumetric(const Scene& scene, int steps) {
        glBindFramebuffer(GL_FRAMEBUFFER, finalFbo_);
        glViewport(0, 0, config_.width, config_.height);
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(fogProgram_);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, depthTexture_);
        setInt(fogProgram_, "uDepth", 0);

        const float aspect = static_cast<float>(config_.width) / config_.height;
        const Mat4 projection = Mat4::perspective(
            scene.camera.fovYDegrees * kPi / 180.0f, aspect,
            scene.camera.nearPlane, scene.camera.farPlane);
        const Mat4 view = Mat4::lookAt(scene.camera.position, scene.camera.target,
                                       scene.camera.up);
        setMat4(fogProgram_, "uInvProjection", inverse(projection));
        setMat4(fogProgram_, "uInvView", inverse(view));
        setVec3(fogProgram_, "uCameraPosition", scene.camera.position);
        setInt(fogProgram_, "uSteps", steps);
        setFloat(fogProgram_, "uDensityScale", config_.volumetric.densityScale);
        setFloat(fogProgram_, "uMaxDistance", config_.volumetric.maxDistance);
        setFloat(fogProgram_, "uNoiseScale", config_.volumetric.noiseScale);
        setFloat(fogProgram_, "uAbsorption", config_.volumetric.absorption);
        setFloat(fogProgram_, "uScattering", config_.volumetric.scattering);
        glBindVertexArray(fullscreenVao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDisable(GL_BLEND);
    }

    int renderTransparent(const Scene& scene, const SortResult& sorted) {
        if (sorted.objects.empty()) return 0;
        std::unordered_map<int, const Primitive*> primitives;
        for (const Primitive& primitive : scene.transparent) {
            primitives[primitive.id] = &primitive;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, finalFbo_);
        glViewport(0, 0, config_.width, config_.height);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_CULL_FACE);
        glUseProgram(transparentProgram_);

        const float aspect = static_cast<float>(config_.width) / config_.height;
        const Mat4 projection = Mat4::perspective(
            scene.camera.fovYDegrees * kPi / 180.0f, aspect,
            scene.camera.nearPlane, scene.camera.farPlane);
        const Mat4 view = Mat4::lookAt(scene.camera.position, scene.camera.target,
                                       scene.camera.up);
        setVec3(transparentProgram_, "uCameraPosition", scene.camera.position);
        int drawn = 0;
        for (const SortedTransparent& item : sorted.objects) {
            const auto found = primitives.find(item.id);
            if (found == primitives.end()) continue;
            const Primitive& primitive = *found->second;
            setPrimitiveModel(transparentProgram_, primitive, view, projection);
            setVec3(transparentProgram_, "uAlbedo", primitive.material.albedo);
            setVec3(transparentProgram_, "uTint", primitive.tint);
            setFloat(transparentProgram_, "uOpacity", primitive.opacity);
            setFloat(transparentProgram_, "uRoughness", primitive.material.roughness);
            setFloat(transparentProgram_, "uMetallic", primitive.material.metallic);
            drawMesh(meshFor(primitive));
            ++drawn;
        }
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        return drawn;
    }

    void renderPresent() {
        glBindFramebuffer(GL_FRAMEBUFFER, displayFbo_);
        glViewport(0, 0, config_.width, config_.height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glUseProgram(presentProgram_);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, finalColor_);
        setInt(presentProgram_, "uSource", 0);
        glBindVertexArray(fullscreenVao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    void captureGBuffer(FrameOutput& output) {
        output.gbuffer.width = config_.width;
        output.gbuffer.height = config_.height;
        output.gbuffer.gPosition.resize(
            static_cast<std::size_t>(config_.width) * config_.height);
        output.gbuffer.gNormal.resize(output.gbuffer.gPosition.size());
        output.gbuffer.gAlbedo.resize(output.gbuffer.gPosition.size());
        output.gbuffer.gMaterial.resize(output.gbuffer.gPosition.size());

        glBindFramebuffer(GL_READ_FRAMEBUFFER, gFbo_);
        readFloatAttachment(GL_COLOR_ATTACHMENT0, output.gbuffer.gPosition);
        readFloatAttachment(GL_COLOR_ATTACHMENT1, output.gbuffer.gNormal);
        readFloatAttachment(GL_COLOR_ATTACHMENT2, output.gbuffer.gAlbedo);
        readFloatAttachment(GL_COLOR_ATTACHMENT3, output.gbuffer.gMaterial);
        for (std::size_t i = 0; i < output.gbuffer.gPosition.size(); ++i) {
            const float depth = output.gbuffer.gPosition[i].w;
            output.linearDepth[i] = depth > 0.0f ? depth : output.camera.farPlane;
        }
    }

    void readFloatAttachment(GLenum attachment, std::vector<Vec4>& destination) {
        glReadBuffer(attachment);
        std::vector<Vec4> rows(static_cast<std::size_t>(config_.width) * config_.height);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, config_.width, config_.height,
                     GL_RGBA, GL_FLOAT, rows.data());
        for (int y = 0; y < config_.height; ++y) {
            const int sourceRow = config_.height - 1 - y;
            std::memcpy(destination.data() + static_cast<std::size_t>(y) * config_.width,
                        rows.data() + static_cast<std::size_t>(sourceRow) * config_.width,
                        static_cast<std::size_t>(config_.width) * sizeof(Vec4));
        }
    }

    void captureColorAndDepth(FrameOutput& output) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, displayFbo_);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        std::vector<std::uint8_t> rgba(
            static_cast<std::size_t>(config_.width) * config_.height * 4u);
        glReadPixels(0, 0, config_.width, config_.height,
                     GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
        for (int y = 0; y < config_.height; ++y) {
            const int sourceRow = config_.height - 1 - y;
            for (int x = 0; x < config_.width; ++x) {
                const std::size_t source =
                    (static_cast<std::size_t>(sourceRow) * config_.width + x) * 4u;
                const std::size_t target =
                    (static_cast<std::size_t>(y) * config_.width + x) * 3u;
                output.colorRgb[target + 0] = rgba[source + 0];
                output.colorRgb[target + 1] = rgba[source + 1];
                output.colorRgb[target + 2] = rgba[source + 2];
            }
        }
        if (output.linearDepth.size() !=
            static_cast<std::size_t>(config_.width) * config_.height) {
            output.linearDepth.assign(
                static_cast<std::size_t>(config_.width) * config_.height,
                output.camera.farPlane);
        }
    }

    void checkGl(const char* operation) {
        const GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            std::ostringstream message;
            message << operation << " failed with OpenGL error 0x"
                    << std::hex << error;
            throw AppError(ErrorCode::E_RENDER_FAIL, message.str());
        }
    }

    void cleanup() {
        if (lightUbo_) glDeleteBuffers(1, &lightUbo_);
        if (fullscreenVbo_) glDeleteBuffers(1, &fullscreenVbo_);
        if (fullscreenVao_) glDeleteVertexArrays(1, &fullscreenVao_);
        for (Mesh* mesh : {&planeMesh_, &cubeMesh_, &sphereMesh_}) {
            if (mesh->ebo) glDeleteBuffers(1, &mesh->ebo);
            if (mesh->vbo) glDeleteBuffers(1, &mesh->vbo);
            if (mesh->vao) glDeleteVertexArrays(1, &mesh->vao);
            *mesh = Mesh{};
        }
        for (GLuint* program : {&gbufferProgram_, &deferredProgram_, &fogProgram_,
                                &transparentProgram_, &presentProgram_}) {
            if (*program) glDeleteProgram(*program);
            *program = 0;
        }
        const std::array<GLuint*, 4> framebuffers = {
            &gFbo_, &sceneFbo_, &finalFbo_, &displayFbo_};
        for (GLuint* framebuffer : framebuffers) {
            if (*framebuffer) glDeleteFramebuffers(1, framebuffer);
            *framebuffer = 0;
        }
        const std::array<GLuint*, 8> textures = {
            &gPosition_, &gNormal_, &gAlbedo_, &gMaterial_,
            &depthTexture_, &sceneColor_, &finalColor_, &displayColor_};
        for (GLuint* texture : textures) {
            if (*texture) glDeleteTextures(1, texture);
            *texture = 0;
        }
        if (window_) {
            glfwDestroyWindow(window_);
            window_ = nullptr;
        }
        if (glfwInitialized_) {
            glfwTerminate();
            glfwInitialized_ = false;
        }
    }

    RenderConfig config_;
    GLFWwindow* window_ = nullptr;
    bool glfwInitialized_ = false;
    GLuint gFbo_ = 0;
    GLuint sceneFbo_ = 0;
    GLuint finalFbo_ = 0;
    GLuint displayFbo_ = 0;
    GLuint gPosition_ = 0;
    GLuint gNormal_ = 0;
    GLuint gAlbedo_ = 0;
    GLuint gMaterial_ = 0;
    GLuint depthTexture_ = 0;
    GLuint sceneColor_ = 0;
    GLuint finalColor_ = 0;
    GLuint displayColor_ = 0;
    GLuint gbufferProgram_ = 0;
    GLuint deferredProgram_ = 0;
    GLuint fogProgram_ = 0;
    GLuint transparentProgram_ = 0;
    GLuint presentProgram_ = 0;
    GLuint fullscreenVao_ = 0;
    GLuint fullscreenVbo_ = 0;
    GLuint lightUbo_ = 0;
    Mesh planeMesh_;
    Mesh cubeMesh_;
    Mesh sphereMesh_;
    LightSnapshot lightSnapshot_;
    LightUploadStats uploadStats_;
};

void animateScene(Scene& scene, double seconds, int lightStep) {
    const float t = static_cast<float>(seconds);
    scene.camera.position = {
        0.35f * std::sin(t * 0.24f),
        1.72f + 0.05f * std::sin(t * 0.37f),
        6.4f + 0.2f * std::cos(t * 0.19f),
    };
    scene.camera.target = {
        0.12f * std::sin(t * 0.13f),
        1.05f + 0.08f * std::cos(t * 0.22f),
        -2.4f,
    };
    const float angle = static_cast<float>(lightStep) * 0.071f;
    if (scene.lights.size() >= 3) {
        scene.lights[0].position = {2.9f + 0.55f * std::sin(angle),
                                    3.3f + 0.25f * std::cos(angle * 0.7f),
                                    1.8f + 0.55f * std::cos(angle)};
        scene.lights[1].position = {-2.8f + 0.4f * std::cos(angle * 1.3f),
                                    2.5f,
                                    -1.3f + 0.4f * std::sin(angle * 1.3f)};
        scene.lights[2].intensity = 3.8f + 1.3f * std::sin(angle * 1.7f);
        scene.lights[0].intensity = 6.8f + 1.1f * std::cos(angle * 1.1f);
    }
}

} // namespace

OfflineArtifacts renderOfflineGpu(const RenderConfig& config,
                                  const std::string& outputDirectory) {
    GpuRenderer renderer(config, false);
    const Scene scene = makeDeterministicScene(config.width, config.height);
    FrameOutput frame = renderer.render(scene, true, true);
    return writeFrameArtifacts(std::move(frame), scene, config, outputDirectory);
}

int runInteractiveGpu(const RenderConfig& config, const std::string& configPath) {
    (void)configPath;
    GpuRenderer renderer(config, true);
    Scene scene = makeDeterministicScene(config.width, config.height);
    const auto start = std::chrono::steady_clock::now();
    int lastLightStep = -1;
    int frameCount = 0;
    while (!renderer.shouldClose()) {
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const int lightStep = static_cast<int>(std::floor(elapsed * 10.0));
        animateScene(scene, elapsed, lightStep);
        FrameOutput frame = renderer.render(scene, false, false);
        renderer.presentToDefault();
        if (lightStep != lastLightStep) {
            lastLightStep = lightStep;
            std::cout << "light-version=" << frame.stats.lightVersion
                      << " fog-steps=" << frame.stats.fogSteps
                      << " transparent=" << frame.transparent.objects.size()
                      << '\n';
        }
        glfwSwapBuffers(glfwGetCurrentContext());
        glfwPollEvents();
        ++frameCount;
        if (config.maxFrames > 0 && frameCount >= config.maxFrames) break;
        if (glfwGetKey(glfwGetCurrentContext(), GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
    }
    return 0;
}

} // namespace df
