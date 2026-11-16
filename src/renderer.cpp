#include "renderer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include "gl_api.hpp"
#include "shaders.hpp"

namespace orv {
namespace {

struct GlTexture {
  GLuint id = 0;

  GlTexture() = default;
  GlTexture(const GlTexture&) = delete;
  GlTexture& operator=(const GlTexture&) = delete;
  GlTexture(GlTexture&& other) noexcept : id(other.id) { other.id = 0; }
  GlTexture& operator=(GlTexture&& other) noexcept {
    if (this != &other) {
      reset();
      id = other.id;
      other.id = 0;
    }
    return *this;
  }
  ~GlTexture() { reset(); }

  void reset() {
    if (id != 0) {
      glDeleteTextures(1, &id);
      id = 0;
    }
  }
};

struct GlFramebuffer {
  GLuint id = 0;

  GlFramebuffer() = default;
  GlFramebuffer(const GlFramebuffer&) = delete;
  GlFramebuffer& operator=(const GlFramebuffer&) = delete;
  GlFramebuffer(GlFramebuffer&& other) noexcept : id(other.id) { other.id = 0; }
  GlFramebuffer& operator=(GlFramebuffer&& other) noexcept {
    if (this != &other) {
      reset();
      id = other.id;
      other.id = 0;
    }
    return *this;
  }
  ~GlFramebuffer() { reset(); }

  void reset() {
    if (id != 0) {
      glDeleteFramebuffers(1, &id);
      id = 0;
    }
  }
};

struct GlBuffer {
  GLuint id = 0;

  GlBuffer() = default;
  GlBuffer(const GlBuffer&) = delete;
  GlBuffer& operator=(const GlBuffer&) = delete;
  GlBuffer(GlBuffer&& other) noexcept : id(other.id) { other.id = 0; }
  GlBuffer& operator=(GlBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      id = other.id;
      other.id = 0;
    }
    return *this;
  }
  ~GlBuffer() { reset(); }

  void reset() {
    if (id != 0) {
      glDeleteBuffers(1, &id);
      id = 0;
    }
  }
};

struct GlVao {
  GLuint id = 0;

  GlVao() = default;
  GlVao(const GlVao&) = delete;
  GlVao& operator=(const GlVao&) = delete;
  GlVao(GlVao&& other) noexcept : id(other.id) { other.id = 0; }
  GlVao& operator=(GlVao&& other) noexcept {
    if (this != &other) {
      reset();
      id = other.id;
      other.id = 0;
    }
    return *this;
  }
  ~GlVao() { reset(); }

  void reset() {
    if (id != 0) {
      glDeleteVertexArrays(1, &id);
      id = 0;
    }
  }
};

class Program {
 public:
  Program() = default;
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;
  ~Program() {
    if (id_ != 0) {
      glDeleteProgram(id_);
    }
  }

  void build(const char* vertexSource, const char* fragmentSource,
             const std::string& label) {
    const GLuint vertex = compile(GL_VERTEX_SHADER, vertexSource, label + " vertex");
    GLuint fragment = 0;
    try {
      fragment = compile(GL_FRAGMENT_SHADER, fragmentSource, label + " fragment");
    } catch (...) {
      glDeleteShader(vertex);
      throw;
    }

    const GLuint program = glCreateProgram();
    if (program == 0) {
      glDeleteShader(vertex);
      glDeleteShader(fragment);
      throw renderError("RENDER_ERR_SHADER_BUILD", 3,
                        "glCreateProgram failed for " + label);
    }
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE) {
      GLint logLength = 0;
      glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
      std::string log(static_cast<std::size_t>(std::max(logLength, 1)), '\0');
      glGetProgramInfoLog(program, static_cast<GLsizei>(log.size()), nullptr, log.data());
      log.resize(std::strlen(log.c_str()));
      glDeleteProgram(program);
      throw renderError("RENDER_ERR_SHADER_BUILD", 3,
                        "failed to link " + label + ": " + log);
    }
    if (id_ != 0) {
      glDeleteProgram(id_);
    }
    id_ = program;
  }

  void use() const { glUseProgram(id_); }
  GLuint id() const { return id_; }

 private:
  static GLuint compile(GLenum type, const char* source, const std::string& label) {
    const GLuint shader = glCreateShader(type);
    if (shader == 0) {
      throw renderError("RENDER_ERR_SHADER_BUILD", 3,
                        "glCreateShader failed for " + label);
    }
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled != GL_TRUE) {
      GLint logLength = 0;
      glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
      std::string log(static_cast<std::size_t>(std::max(logLength, 1)), '\0');
      glGetShaderInfoLog(shader, static_cast<GLsizei>(log.size()), nullptr, log.data());
      log.resize(std::strlen(log.c_str()));
      glDeleteShader(shader);
      throw renderError("RENDER_ERR_SHADER_BUILD", 3,
                        "failed to compile " + label + ": " + log);
    }
    return shader;
  }

  GLuint id_ = 0;
};

struct GpuMesh {
  GlBuffer vertexBuffer;
  GlBuffer indexBuffer;
  GlVao vao;
  GLsizei indexCount = 0;
};

void setInt(const Program& program, const char* name, GLint value) {
  const GLint location = glGetUniformLocation(program.id(), name);
  if (location >= 0) {
    glUniform1i(location, value);
  }
}

void setFloat(const Program& program, const char* name, GLfloat value) {
  const GLint location = glGetUniformLocation(program.id(), name);
  if (location >= 0) {
    glUniform1f(location, value);
  }
}

void setVec3(const Program& program, const char* name, const Vec3& value) {
  const GLint location = glGetUniformLocation(program.id(), name);
  if (location >= 0) {
    const GLfloat values[3] = {value.x, value.y, value.z};
    glUniform3fv(location, 1, values);
  }
}

void setVec3Array(const Program& program, const char* name, const std::vector<float>& values,
                  GLsizei count) {
  if (count <= 0) {
    return;
  }
  const GLint location = glGetUniformLocation(program.id(), name);
  if (location >= 0) {
    glUniform3fv(location, count, values.data());
  }
}

void setFloatArray(const Program& program, const char* name, const std::vector<float>& values,
                   GLsizei count) {
  if (count <= 0) {
    return;
  }
  const GLint location = glGetUniformLocation(program.id(), name);
  if (location >= 0) {
    glUniform1fv(location, count, values.data());
  }
}

void setMat4(const Program& program, const char* name, const Mat4& value) {
  const GLint location = glGetUniformLocation(program.id(), name);
  if (location >= 0) {
    glUniformMatrix4fv(location, 1, GL_FALSE, value.data());
  }
}

std::string framebufferStatusText(GLenum status) {
  std::ostringstream stream;
  stream << "0x" << std::hex << status;
  return stream.str();
}

void throwFramebufferIncomplete(const std::string& label, GLenum status) {
  throw renderError("RENDER_ERR_FRAMEBUFFER_INCOMPLETE", 3,
                    label + " framebuffer is incomplete (" +
                        framebufferStatusText(status) + ")");
}

void checkGl(const std::string& stage) {
  const GLenum error = glGetError();
  if (error != GL_NO_ERROR) {
    std::ostringstream stream;
    stream << "OpenGL error 0x" << std::hex << error << " during " << stage;
    throw renderError("RENDER_ERR_OPENGL", 3, stream.str());
  }
}

void createTexture2D(GlTexture& texture, GLenum internalFormat, GLenum format, GLenum type,
                     int width, int height) {
  glGenTextures(1, &texture.id);
  if (texture.id == 0) {
    throw renderError("RENDER_ERR_TEXTURE", 3, "failed to allocate an OpenGL texture");
  }
  glBindTexture(GL_TEXTURE_2D, texture.id);
  glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(internalFormat), width, height, 0,
               format, type, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
}

void createCubeDepthTexture(GlTexture& texture, int size) {
  glGenTextures(1, &texture.id);
  if (texture.id == 0) {
    throw renderError("RENDER_ERR_TEXTURE", 3,
                      "failed to allocate a point-shadow cube texture");
  }
  glBindTexture(GL_TEXTURE_CUBE_MAP, texture.id);
  for (int face = 0; face < 6; ++face) {
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + static_cast<GLenum>(face), 0,
                 GL_DEPTH_COMPONENT24, size, size, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                 nullptr);
  }
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, 0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

GpuMesh createGpuMesh(const Mesh& mesh) {
  GpuMesh result;
  std::vector<float> vertices;
  vertices.reserve(mesh.positions.size() * 11);
  for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
    const Vec3& position = mesh.positions[i];
    const Vec3& normal = mesh.normals[i];
    const Vec3& tangent = mesh.tangents[i];
    const Vec2& uv = mesh.uvs[i];
    vertices.insert(vertices.end(),
                    {position.x, position.y, position.z, normal.x, normal.y, normal.z,
                     tangent.x, tangent.y, tangent.z, uv.x, uv.y});
  }

  glGenBuffers(1, &result.vertexBuffer.id);
  glGenBuffers(1, &result.indexBuffer.id);
  glGenVertexArrays(1, &result.vao.id);
  if (result.vertexBuffer.id == 0 || result.indexBuffer.id == 0 || result.vao.id == 0) {
    throw renderError("RENDER_ERR_BUFFER", 3, "failed to allocate mesh buffers");
  }
  glBindVertexArray(result.vao.id);
  glBindBuffer(GL_ARRAY_BUFFER, result.vertexBuffer.id);
  glBufferData(GL_ARRAY_BUFFER,
               static_cast<GLsizeiptr>(vertices.size() * sizeof(float)), vertices.data(),
               GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, result.indexBuffer.id);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,
               static_cast<GLsizeiptr>(mesh.indices.size() * sizeof(std::uint32_t)),
               mesh.indices.data(), GL_STATIC_DRAW);

  constexpr GLsizei stride = 11 * sizeof(float);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
                        reinterpret_cast<const void*>(0));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                        reinterpret_cast<const void*>(3 * sizeof(float)));
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride,
                        reinterpret_cast<const void*>(6 * sizeof(float)));
  glEnableVertexAttribArray(3);
  glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride,
                        reinterpret_cast<const void*>(9 * sizeof(float)));
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  result.indexCount = static_cast<GLsizei>(mesh.indices.size());
  return result;
}

float luminance(double red, double green, double blue) {
  return static_cast<float>(0.2126 * red + 0.7152 * green + 0.0722 * blue);
}

double acesToneMap(double value) {
  value = std::max(value, 0.0);
  return (value * (2.51 * value + 0.03)) / (value * (2.43 * value + 0.59) + 0.14);
}

std::uint8_t toByte(double value) {
  return static_cast<std::uint8_t>(std::clamp(value, 0.0, 1.0) * 255.0 + 0.5);
}

}  // namespace

struct Renderer::Impl {
  const Scene& scene;
  RenderOptions options;
  InterruptCheck interruptCheck;

  std::vector<GpuMesh> meshes;
  std::array<GlTexture, 5> gbufferTextures;
  GlTexture gbufferDepth;
  GlTexture finalColor;
  GlTexture directionalShadow;
  std::vector<GlTexture> pointShadows;
  GlFramebuffer gbufferFramebuffer;
  GlFramebuffer finalFramebuffer;
  GlFramebuffer directionalShadowFramebuffer;
  GlFramebuffer pointShadowFramebuffer;
  GlVao fullscreenVao;

  Program geometryProgram;
  Program shadowProgram;
  Program deferredProgram;

  Mat4 view;
  Mat4 projection;
  Mat4 viewProjection;
  Mat4 directionalLightViewProjection;
  std::vector<std::array<Mat4, 6>> pointFaceViewProjections;
  std::vector<float> pointFarPlanes;
  Vec3 boundsCenter;
  float boundsRadius = 1.0f;

  Impl(const Scene& sceneValue, const RenderOptions& optionsValue,
       InterruptCheck interrupt)
      : scene(sceneValue), options(optionsValue), interruptCheck(std::move(interrupt)) {
    initialize();
  }

  void checkInterrupt() const {
    if (interruptCheck && interruptCheck()) {
      throw renderError("RENDER_INTERRUPTED", 130,
                        "rendering interrupted by a handled signal");
    }
  }

  void initialize() {
    initializeMatrices();
    compilePrograms();
    uploadMeshes();
    createFramebuffers();
    createFullscreenVao();
    checkGl("resource initialization");
  }

  void initializeMatrices() {
    const float aspect =
        static_cast<float>(options.width) / static_cast<float>(options.height);
    view = lookAt(scene.camera.position, scene.camera.target, scene.camera.up);
    projection = perspective(radians(scene.camera.fovYDegrees), aspect,
                             scene.camera.nearPlane, scene.camera.farPlane);
    viewProjection = projection * view;

    Vec3 minimum{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max()};
    Vec3 maximum{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                 -std::numeric_limits<float>::max()};
    for (const Mesh& mesh : scene.meshes) {
      for (const Vec3& position : mesh.positions) {
        const Vec3 world = transformPoint(mesh.model, position);
        minimum.x = std::min(minimum.x, world.x);
        minimum.y = std::min(minimum.y, world.y);
        minimum.z = std::min(minimum.z, world.z);
        maximum.x = std::max(maximum.x, world.x);
        maximum.y = std::max(maximum.y, world.y);
        maximum.z = std::max(maximum.z, world.z);
      }
    }
    boundsCenter = (minimum + maximum) * 0.5f;
    boundsRadius = std::max(length(maximum - minimum) * 0.5f, 0.25f);

    if (scene.hasDirectionalLight) {
      const Vec3 direction = normalize(scene.directionalLight.direction);
      const Vec3 up = std::abs(direction.y) > 0.95f ? Vec3{0.0f, 0.0f, 1.0f}
                                                    : Vec3{0.0f, 1.0f, 0.0f};
      const Vec3 eye = boundsCenter - direction * (boundsRadius * 2.5f);
      const Mat4 lightView = lookAt(eye, boundsCenter, up);
      const Mat4 lightProjection =
          orthographic(-boundsRadius * 1.35f, boundsRadius * 1.35f,
                       -boundsRadius * 1.35f, boundsRadius * 1.35f, 0.01f,
                       boundsRadius * 6.0f + 1.0f);
      directionalLightViewProjection = lightProjection * lightView;
    }

    static constexpr std::array<Vec3, 6> faceDirections = {
        Vec3{1, 0, 0},  Vec3{-1, 0, 0}, Vec3{0, 1, 0},
        Vec3{0, -1, 0}, Vec3{0, 0, 1},  Vec3{0, 0, -1}};
    static constexpr std::array<Vec3, 6> faceUps = {
        Vec3{0, 1, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1},
        Vec3{0, 0, -1}, Vec3{0, 1, 0}, Vec3{0, 1, 0}};

    pointFaceViewProjections.reserve(scene.pointLights.size());
    pointFarPlanes.reserve(scene.pointLights.size());
    for (const PointLight& light : scene.pointLights) {
      const float nearPlane = std::max(0.01f, light.range * 0.001f);
      const float farPlane = std::max(light.range, nearPlane + 0.01f);
      const Mat4 pointProjection =
          perspective(radians(90.0f), 1.0f, nearPlane, farPlane);
      std::array<Mat4, 6> faceMatrices{};
      for (std::size_t face = 0; face < 6; ++face) {
        const Mat4 faceView =
            lookAt(light.position, light.position + faceDirections[face], faceUps[face]);
        faceMatrices[face] = pointProjection * faceView;
      }
      pointFaceViewProjections.push_back(faceMatrices);
      pointFarPlanes.push_back(farPlane);
    }
  }

  void compilePrograms() {
    geometryProgram.build(shaders::kGeometryVertex, shaders::kGeometryFragment,
                          "geometry program");
    shadowProgram.build(shaders::kShadowVertex, shaders::kShadowFragment,
                        "shadow program");
    deferredProgram.build(shaders::kFullscreenVertex, shaders::kDeferredFragment,
                          "deferred lighting program");
  }

  void uploadMeshes() {
    meshes.reserve(scene.meshes.size());
    for (const Mesh& mesh : scene.meshes) {
      meshes.push_back(createGpuMesh(mesh));
    }
  }

  void createFramebuffers() {
    createGBuffer();
    createFinalFramebuffer();
    if (scene.hasDirectionalLight) {
      createDirectionalShadowFramebuffer();
    }
    if (!scene.pointLights.empty()) {
      createPointShadowFramebuffers();
    }
  }

  void createGBuffer() {
    glGenFramebuffers(1, &gbufferFramebuffer.id);
    if (gbufferFramebuffer.id == 0) {
      throwFramebufferIncomplete("G-Buffer", 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, gbufferFramebuffer.id);
    for (std::size_t i = 0; i < gbufferTextures.size(); ++i) {
      createTexture2D(gbufferTextures[i], GL_RGBA16F, GL_RGBA, GL_FLOAT, options.width,
                      options.height);
      glFramebufferTexture2D(GL_FRAMEBUFFER,
                             GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(i),
                             GL_TEXTURE_2D, gbufferTextures[i].id, 0);
    }
    createTexture2D(gbufferDepth, GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL,
                    GL_UNSIGNED_INT_24_8,
                    options.width, options.height);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D,
                           gbufferDepth.id, 0);

    const GLenum drawBuffers[5] = {
        GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2,
        GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4};
    glDrawBuffers(5, drawBuffers);

    if (std::getenv("OFFLINE_RENDER_FORCE_FRAMEBUFFER_INCOMPLETE") != nullptr) {
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, 0, 0);
    }
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    const bool forcedIncomplete =
        std::getenv("OFFLINE_RENDER_FORCE_FRAMEBUFFER_INCOMPLETE") != nullptr;
    if (forcedIncomplete || status != GL_FRAMEBUFFER_COMPLETE) {
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      throwFramebufferIncomplete(
          "G-Buffer", forcedIncomplete ? GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT : status);
    }

    const GLfloat albedoClear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const GLfloat normalClear[4] = {0.0f, 0.0f, 1.0f, 1.0f};
    const GLfloat positionClear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const GLfloat tangentClear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const GLfloat rotationClear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    glClearBufferfv(GL_COLOR, 0, albedoClear);
    glClearBufferfv(GL_COLOR, 1, normalClear);
    glClearBufferfv(GL_COLOR, 2, positionClear);
    glClearBufferfv(GL_COLOR, 3, tangentClear);
    glClearBufferfv(GL_COLOR, 4, rotationClear);
    glClearBufferfi(GL_DEPTH_STENCIL, 0, 1.0f, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  void createFinalFramebuffer() {
    glGenFramebuffers(1, &finalFramebuffer.id);
    if (finalFramebuffer.id == 0) {
      throwFramebufferIncomplete("lighting output", 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, finalFramebuffer.id);
    createTexture2D(finalColor, GL_RGBA16F, GL_RGBA, GL_FLOAT, options.width,
                    options.height);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           finalColor.id, 0);
    const GLenum drawBuffer = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &drawBuffer);
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      throwFramebufferIncomplete("lighting output", status);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  void createDirectionalShadowFramebuffer() {
    glGenFramebuffers(1, &directionalShadowFramebuffer.id);
    if (directionalShadowFramebuffer.id == 0) {
      throwFramebufferIncomplete("directional shadow", 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, directionalShadowFramebuffer.id);
    createTexture2D(directionalShadow, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT,
                    options.shadowSize, options.shadowSize);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                           directionalShadow.id, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      throwFramebufferIncomplete("directional shadow", status);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  void createPointShadowFramebuffers() {
    glGenFramebuffers(1, &pointShadowFramebuffer.id);
    if (pointShadowFramebuffer.id == 0) {
      throwFramebufferIncomplete("point shadow", 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, pointShadowFramebuffer.id);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    pointShadows.resize(scene.pointLights.size());
    for (GlTexture& cube : pointShadows) {
      createCubeDepthTexture(cube, options.shadowSize);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                             GL_TEXTURE_CUBE_MAP_POSITIVE_X, cube.id, 0);
      const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
      if (status != GL_FRAMEBUFFER_COMPLETE) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        throwFramebufferIncomplete("point shadow", status);
      }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  void createFullscreenVao() {
    glGenVertexArrays(1, &fullscreenVao.id);
    if (fullscreenVao.id == 0) {
      throw renderError("RENDER_ERR_BUFFER", 3,
                        "failed to allocate the fullscreen vertex array");
    }
  }

  void drawMeshes(const Program& program, const Mat4& lightViewProjection,
                  bool radialDistance, const PointLight* pointLight, float farPlane) {
    for (std::size_t i = 0; i < meshes.size(); ++i) {
      checkInterrupt();
      setMat4(program, "uModel", scene.meshes[i].model);
      setMat4(program, "uLightViewProjection", lightViewProjection);
      setInt(program, "uRadialDistance", radialDistance ? 1 : 0);
      if (pointLight != nullptr) {
        setVec3(program, "uLightPosition", pointLight->position);
        setFloat(program, "uLightFar", farPlane);
      }
      glBindVertexArray(meshes[i].vao.id);
      glDrawElements(GL_TRIANGLES, meshes[i].indexCount, GL_UNSIGNED_INT, nullptr);
    }
    glBindVertexArray(0);
  }

  void renderShadowMaps() {
    checkInterrupt();
    shadowProgram.use();
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    if (scene.hasDirectionalLight) {
      glBindFramebuffer(GL_FRAMEBUFFER, directionalShadowFramebuffer.id);
      glViewport(0, 0, options.shadowSize, options.shadowSize);
      glDrawBuffer(GL_NONE);
      glReadBuffer(GL_NONE);
      glClear(GL_DEPTH_BUFFER_BIT);
      drawMeshes(shadowProgram, directionalLightViewProjection, false, nullptr, 0.0f);
    }

    for (std::size_t lightIndex = 0; lightIndex < scene.pointLights.size(); ++lightIndex) {
      const PointLight& light = scene.pointLights[lightIndex];
      const float farPlane = pointFarPlanes[lightIndex];
      glBindFramebuffer(GL_FRAMEBUFFER, pointShadowFramebuffer.id);
      glDrawBuffer(GL_NONE);
      glReadBuffer(GL_NONE);
      glViewport(0, 0, options.shadowSize, options.shadowSize);
      for (GLenum face = 0; face < 6; ++face) {
        checkInterrupt();
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                               pointShadows[lightIndex].id, 0);
        glClear(GL_DEPTH_BUFFER_BIT);
        drawMeshes(shadowProgram, pointFaceViewProjections[lightIndex][face], true,
                   &light, farPlane);
      }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_CULL_FACE);
    checkGl("shadow map pass");
  }

  void renderGeometryPass() {
    checkInterrupt();
    glBindFramebuffer(GL_FRAMEBUFFER, gbufferFramebuffer.id);
    glViewport(0, 0, options.width, options.height);
    const GLenum drawBuffers[5] = {
        GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2,
        GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4};
    glDrawBuffers(5, drawBuffers);
    const GLfloat albedoClear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const GLfloat normalClear[4] = {0.0f, 0.0f, 1.0f, 1.0f};
    const GLfloat positionClear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const GLfloat tangentClear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const GLfloat rotationClear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    glClearBufferfv(GL_COLOR, 0, albedoClear);
    glClearBufferfv(GL_COLOR, 1, normalClear);
    glClearBufferfv(GL_COLOR, 2, positionClear);
    glClearBufferfv(GL_COLOR, 3, tangentClear);
    glClearBufferfv(GL_COLOR, 4, rotationClear);
    glClearBufferfi(GL_DEPTH_STENCIL, 0, 1.0f, 0);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    geometryProgram.use();
    setMat4(geometryProgram, "uViewProjection", viewProjection);
    setMat4(geometryProgram, "uView", view);
    for (std::size_t i = 0; i < meshes.size(); ++i) {
      checkInterrupt();
      const Mesh& mesh = scene.meshes[i];
      setMat4(geometryProgram, "uModel", mesh.model);
      setVec3(geometryProgram, "uBaseColor", mesh.material.baseColor);
      setFloat(geometryProgram, "uRoughness", mesh.material.roughness);
      setFloat(geometryProgram, "uMetallic", mesh.material.metallic);
      setFloat(geometryProgram, "uAnisotropyStrength",
               mesh.material.anisotropyStrength);
      setFloat(geometryProgram, "uAnisotropyRotation",
               mesh.material.anisotropyRotationRadians);
      glBindVertexArray(meshes[i].vao.id);
      glDrawElements(GL_TRIANGLES, meshes[i].indexCount, GL_UNSIGNED_INT, nullptr);
    }
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_CULL_FACE);
    checkGl("geometry pass");
  }

  void bindDeferredTextures() {
    const std::array<GLuint, 5> textureIds = {
        gbufferTextures[0].id, gbufferTextures[1].id, gbufferTextures[2].id,
        gbufferTextures[3].id, gbufferTextures[4].id};
    for (std::size_t i = 0; i < textureIds.size(); ++i) {
      glActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(i));
      glBindTexture(GL_TEXTURE_2D, textureIds[i]);
    }
    glActiveTexture(GL_TEXTURE0 + 5);
    glBindTexture(GL_TEXTURE_2D, directionalShadow.id);
    for (std::size_t i = 0; i < 4; ++i) {
      glActiveTexture(GL_TEXTURE0 + 6 + static_cast<GLenum>(i));
      const GLuint cubeId = i < pointShadows.size() ? pointShadows[i].id : 0;
      glBindTexture(GL_TEXTURE_CUBE_MAP, cubeId);
    }
  }

  void renderLightingPass() {
    checkInterrupt();
    glBindFramebuffer(GL_FRAMEBUFFER, finalFramebuffer.id);
    glViewport(0, 0, options.width, options.height);
    const GLenum drawBuffer = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &drawBuffer);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDepthMask(GL_FALSE);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    deferredProgram.use();
    bindDeferredTextures();
    for (int unit = 0; unit <= 5; ++unit) {
      setInt(deferredProgram,
             ("u" + std::string(unit == 0   ? "AlbedoMetallic"
                                : unit == 1 ? "NormalRoughness"
                                : unit == 2 ? "PositionDepth"
                                : unit == 3 ? "TangentAnisotropy"
                                : unit == 4 ? "AnisotropyRotation"
                                            : "DirectionalShadow"))
                 .c_str(),
             unit);
    }
    for (int i = 0; i < 4; ++i) {
      setInt(deferredProgram, ("uPointShadow" + std::to_string(i)).c_str(), 6 + i);
    }

    setVec3(deferredProgram, "uCameraPosition", scene.camera.position);
    setInt(deferredProgram, "uHasDirectional", scene.hasDirectionalLight ? 1 : 0);
    setInt(deferredProgram, "uShadowMapSize", options.shadowSize);
    setFloat(deferredProgram, "uShadowBias", options.shadowBias);
    if (scene.hasDirectionalLight) {
      setVec3(deferredProgram, "uDirectionalDirection",
              scene.directionalLight.direction);
      setVec3(deferredProgram, "uDirectionalColor", scene.directionalLight.color);
      setFloat(deferredProgram, "uDirectionalIntensity",
               scene.directionalLight.intensity);
      setMat4(deferredProgram, "uDirectionalViewProjection",
              directionalLightViewProjection);
    }

    const GLsizei pointCount = static_cast<GLsizei>(scene.pointLights.size());
    std::vector<float> positions;
    std::vector<float> colors;
    std::vector<float> intensities;
    std::vector<float> ranges;
    positions.reserve(static_cast<std::size_t>(pointCount) * 3);
    colors.reserve(static_cast<std::size_t>(pointCount) * 3);
    intensities.reserve(static_cast<std::size_t>(pointCount));
    ranges.reserve(static_cast<std::size_t>(pointCount));
    for (const PointLight& light : scene.pointLights) {
      positions.insert(positions.end(), {light.position.x, light.position.y,
                                         light.position.z});
      colors.insert(colors.end(), {light.color.x, light.color.y, light.color.z});
      intensities.push_back(light.intensity);
      ranges.push_back(light.range);
    }
    setInt(deferredProgram, "uPointLightCount", pointCount);
    setVec3Array(deferredProgram, "uPointPositions", positions, pointCount);
    setVec3Array(deferredProgram, "uPointColors", colors, pointCount);
    setFloatArray(deferredProgram, "uPointIntensities", intensities, pointCount);
    setFloatArray(deferredProgram, "uPointRanges", ranges, pointCount);
    setFloatArray(deferredProgram, "uPointFars", pointFarPlanes, pointCount);

    glBindVertexArray(fullscreenVao.id);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    checkGl("deferred lighting pass");
  }

  std::vector<float> readRgbaFloats(const GlFramebuffer& framebuffer,
                                    GLenum attachment) {
    std::vector<float> pixels(static_cast<std::size_t>(options.width) *
                              static_cast<std::size_t>(options.height) * 4U);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.id);
    glReadBuffer(attachment);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, options.width, options.height, GL_RGBA, GL_FLOAT, pixels.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    checkGl("framebuffer readback");
    return pixels;
  }

  RenderOutput finishFrame(double renderTimeMs) {
    const std::vector<float> finalPixels =
        readRgbaFloats(finalFramebuffer, GL_COLOR_ATTACHMENT0);
    const std::vector<float> positionPixels =
        readRgbaFloats(gbufferFramebuffer, GL_COLOR_ATTACHMENT2);
    std::vector<float> normalPixels;
    if (options.pass == RenderPass::Geometry) {
      normalPixels = readRgbaFloats(gbufferFramebuffer, GL_COLOR_ATTACHMENT1);
    }

    Image8 image;
    image.width = options.width;
    image.height = options.height;
    image.rgb.resize(static_cast<std::size_t>(options.width) *
                     static_cast<std::size_t>(options.height) * 3U);

    std::uint64_t visiblePixels = 0;
    double luminanceSum = 0.0;
    double maximumLuminance = 0.0;
    for (int y = 0; y < options.height; ++y) {
      const int sourceY = options.height - 1 - y;
      for (int x = 0; x < options.width; ++x) {
        const std::size_t pixelIndex =
            static_cast<std::size_t>(sourceY) * static_cast<std::size_t>(options.width) +
            static_cast<std::size_t>(x);
        const std::size_t source = pixelIndex * 4U;
        const std::size_t destination =
            (static_cast<std::size_t>(y) * static_cast<std::size_t>(options.width) +
             static_cast<std::size_t>(x)) *
            3U;
        const bool visible = positionPixels[source + 3U] > 0.0001f;
        if (visible) {
          ++visiblePixels;
        }

        const double red = acesToneMap(finalPixels[source]);
        const double green = acesToneMap(finalPixels[source + 1U]);
        const double blue = acesToneMap(finalPixels[source + 2U]);
        const double pixelLuma = luminance(red, green, blue);
        maximumLuminance = std::max(maximumLuminance, pixelLuma);
        if (visible) {
          luminanceSum += pixelLuma;
        }

        if (options.pass == RenderPass::Geometry) {
          image.rgb[destination] = toByte(normalPixels[source] * 0.5 + 0.5);
          image.rgb[destination + 1U] = toByte(normalPixels[source + 1U] * 0.5 + 0.5);
          image.rgb[destination + 2U] = toByte(normalPixels[source + 2U] * 0.5 + 0.5);
        } else if (options.pass == RenderPass::Lighting) {
          const std::uint8_t shadow =
              toByte(static_cast<double>(finalPixels[source + 3U]));
          image.rgb[destination] = shadow;
          image.rgb[destination + 1U] = shadow;
          image.rgb[destination + 2U] = shadow;
        } else {
          image.rgb[destination] = toByte(std::pow(red, 1.0 / 2.2));
          image.rgb[destination + 1U] = toByte(std::pow(green, 1.0 / 2.2));
          image.rgb[destination + 2U] = toByte(std::pow(blue, 1.0 / 2.2));
        }
      }
    }

    double averageLuminance =
        visiblePixels == 0 ? 0.0
                           : luminanceSum / static_cast<double>(visiblePixels);
    if (visiblePixels == 0) {
      maximumLuminance = 0.0;
    }
    bool averageClamped = false;
    bool maximumClamped = false;
    const auto clampLuminance = [](double value, bool& clamped) {
      if (value < -1.0e-7 || value > 1.0 + 1.0e-7) {
        clamped = true;
      }
      return std::clamp(value, 0.0, 1.0);
    };
    averageLuminance = clampLuminance(averageLuminance, averageClamped);
    maximumLuminance = clampLuminance(maximumLuminance, maximumClamped);

    std::string shadowMethod;
    if (scene.hasDirectionalLight) {
      shadowMethod = "directional_2d_pcf3x3";
    }
    if (!scene.pointLights.empty()) {
      if (!shadowMethod.empty()) {
        shadowMethod += "+";
      }
      shadowMethod += "point_cubemap_pcf3x3";
    }
    if (shadowMethod.empty()) {
      shadowMethod = "none";
    }

    RenderOutput output;
    output.image = std::move(image);
    output.stats.width = options.width;
    output.stats.height = options.height;
    output.stats.renderTimeMs = renderTimeMs;
    output.stats.visiblePixels = visiblePixels;
    output.stats.averageLuminance = averageLuminance;
    output.stats.maximumLuminance = maximumLuminance;
    output.stats.averageLuminanceClamped = averageClamped;
    output.stats.maximumLuminanceClamped = maximumClamped;
    output.stats.shadowMethod = shadowMethod;
    output.stats.lightCount = scene.lightCount();
    output.stats.pass =
        options.pass == RenderPass::Geometry
            ? "geometry"
            : (options.pass == RenderPass::Lighting ? "lighting" : "full");
    return output;
  }

  RenderOutput render() {
    checkInterrupt();
    const auto start = std::chrono::steady_clock::now();
    renderShadowMaps();
    renderGeometryPass();
    renderLightingPass();
    glFinish();
    const auto end = std::chrono::steady_clock::now();
    checkInterrupt();
    const double milliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();
    return finishFrame(milliseconds);
  }
};

Renderer::Renderer(const Scene& scene, const RenderOptions& options,
                   InterruptCheck interruptCheck)
    : impl_(std::make_unique<Impl>(scene, options, std::move(interruptCheck))) {}

Renderer::~Renderer() = default;

RenderOutput Renderer::render() { return impl_->render(); }

}  // namespace orv
