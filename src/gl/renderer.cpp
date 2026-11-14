#include "lf/renderer.h"

#include "lf/brdf.h"
#include "lf/error_code.h"
#include "lf/log.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>

namespace lf {

namespace {

constexpr int kMaxLights = 32;
constexpr float kPi = 3.14159265358979323846F;

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw AppError(ErrorCode::E_RESOURCE_MISSING,
                   "required shader is missing: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  if (buffer.str().empty()) {
    throw AppError(ErrorCode::E_RESOURCE_MISSING,
                   "required shader is empty: " + path.string());
  }
  return buffer.str();
}

gl::GLuint compile_shader(gl::GLenum type, const std::filesystem::path& path) {
  const std::string source = read_text_file(path);
  const char* sourcePointer = source.c_str();
  const gl::GLint sourceLength = static_cast<gl::GLint>(source.size());
  const gl::GLuint shader = gl::gl.CreateShader(type);
  gl::gl.ShaderSource(shader, 1, &sourcePointer, &sourceLength);
  gl::gl.CompileShader(shader);

  gl::GLint compiled = gl::GL_FALSE;
  gl::gl.GetShaderiv(shader, gl::GL_COMPILE_STATUS, &compiled);
  if (compiled == static_cast<gl::GLint>(gl::GL_FALSE)) {
    gl::GLint logLength = 0;
    gl::gl.GetShaderiv(shader, gl::GL_INFO_LOG_LENGTH, &logLength);
    std::string log(static_cast<std::size_t>(std::max(logLength, 1)), '\0');
    gl::gl.GetShaderInfoLog(shader, static_cast<gl::GLsizei>(log.size()), nullptr, log.data());
    gl::gl.DeleteShader(shader);
    throw AppError(ErrorCode::E_SHADER_COMPILE,
                   "shader compile failed (" + path.string() + "): " + log);
  }
  return shader;
}

gl::GLuint link_program(const std::filesystem::path& vertexPath,
                        const std::filesystem::path& fragmentPath) {
  const gl::GLuint vertex = compile_shader(gl::GL_VERTEX_SHADER, vertexPath);
  gl::GLuint fragment = 0;
  try {
    fragment = compile_shader(gl::GL_FRAGMENT_SHADER, fragmentPath);
  } catch (...) {
    gl::gl.DeleteShader(vertex);
    throw;
  }

  const gl::GLuint program = gl::gl.CreateProgram();
  gl::gl.AttachShader(program, vertex);
  gl::gl.AttachShader(program, fragment);
  gl::gl.LinkProgram(program);
  gl::gl.DeleteShader(vertex);
  gl::gl.DeleteShader(fragment);

  gl::GLint linked = gl::GL_FALSE;
  gl::gl.GetProgramiv(program, gl::GL_LINK_STATUS, &linked);
  if (linked == static_cast<gl::GLint>(gl::GL_FALSE)) {
    gl::GLint logLength = 0;
    gl::gl.GetProgramiv(program, gl::GL_INFO_LOG_LENGTH, &logLength);
    std::string log(static_cast<std::size_t>(std::max(logLength, 1)), '\0');
    gl::gl.GetProgramInfoLog(program, static_cast<gl::GLsizei>(log.size()), nullptr, log.data());
    gl::gl.DeleteProgram(program);
    throw AppError(ErrorCode::E_SHADER_LINK,
                   "shader link failed (" + vertexPath.filename().string() + " + " +
                       fragmentPath.filename().string() + "): " + log);
  }
  return program;
}

struct TgaImage {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> rgba;
};

std::uint16_t read_u16(const std::vector<std::uint8_t>& data, std::size_t offset) {
  return static_cast<std::uint16_t>(data.at(offset)) |
         static_cast<std::uint16_t>(data.at(offset + 1U) << 8U);
}

TgaImage load_tga(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw AppError(ErrorCode::E_TEXTURE_LOAD,
                   "cannot open texture: " + path.string());
  }
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
  if (bytes.size() < 18U) {
    throw AppError(ErrorCode::E_TEXTURE_LOAD,
                   "texture header is truncated: " + path.string());
  }

  const std::uint8_t idLength = bytes[0];
  const std::uint8_t colorMapType = bytes[1];
  const std::uint8_t imageType = bytes[2];
  if (colorMapType != 0U || imageType != 2U) {
    throw AppError(ErrorCode::E_TEXTURE_LOAD,
                   "only uncompressed true-color TGA is supported: " + path.string());
  }
  const int width = static_cast<int>(read_u16(bytes, 12U));
  const int height = static_cast<int>(read_u16(bytes, 14U));
  const int bitsPerPixel = static_cast<int>(bytes[16]);
  if (width <= 0 || height <= 0 || (bitsPerPixel != 24 && bitsPerPixel != 32)) {
    throw AppError(ErrorCode::E_TEXTURE_LOAD,
                   "unsupported TGA dimensions or bit depth: " + path.string());
  }

  const std::size_t bytesPerPixel = static_cast<std::size_t>(bitsPerPixel / 8);
  const std::size_t pixelOffset =
      18U + static_cast<std::size_t>(idLength) + static_cast<std::size_t>(bytes[7]);
  const std::size_t pixelBytes =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * bytesPerPixel;
  if (bytes.size() < pixelOffset + pixelBytes) {
    throw AppError(ErrorCode::E_TEXTURE_LOAD,
                   "texture pixel payload is truncated: " + path.string());
  }

  const bool topOrigin = (bytes[17] & 0x20U) != 0U;
  TgaImage image;
  image.width = width;
  image.height = height;
  image.rgba.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4U);
  for (int y = 0; y < height; ++y) {
    const int sourceY = topOrigin ? y : (height - 1 - y);
    for (int x = 0; x < width; ++x) {
      const std::size_t sourceIndex =
          pixelOffset + (static_cast<std::size_t>(sourceY) * static_cast<std::size_t>(width) +
                         static_cast<std::size_t>(x)) *
                            bytesPerPixel;
      const std::size_t targetIndex =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x)) *
          4U;
      image.rgba[targetIndex] = bytes[sourceIndex + 2U];
      image.rgba[targetIndex + 1U] = bytes[sourceIndex + 1U];
      image.rgba[targetIndex + 2U] = bytes[sourceIndex];
      image.rgba[targetIndex + 3U] =
          bytesPerPixel == 4U ? bytes[sourceIndex + 3U] : std::uint8_t{255};
    }
  }
  return image;
}

gl::GLuint create_texture_rgba8(int width, int height, const void* pixels, bool mipmap,
                                gl::GLenum wrapMode) {
  gl::GLuint texture = 0;
  gl::gl.GenTextures(1, &texture);
  gl::gl.BindTexture(gl::GL_TEXTURE_2D, texture);
  gl::gl.PixelStorei(gl::GL_UNPACK_ALIGNMENT, 1);
  gl::gl.TexImage2D(gl::GL_TEXTURE_2D, 0, static_cast<gl::GLint>(gl::GL_RGBA8), width, height,
                    0, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, pixels);
  gl::gl.TexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_WRAP_S,
                       static_cast<gl::GLint>(wrapMode));
  gl::gl.TexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_WRAP_T,
                       static_cast<gl::GLint>(wrapMode));
  gl::gl.TexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MAG_FILTER,
                       static_cast<gl::GLint>(gl::GL_LINEAR));
  if (mipmap) {
    gl::gl.GenerateMipmap(gl::GL_TEXTURE_2D);
    gl::gl.TexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MIN_FILTER,
                         static_cast<gl::GLint>(gl::GL_LINEAR_MIPMAP_LINEAR));
  } else {
    gl::gl.TexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MIN_FILTER,
                         static_cast<gl::GLint>(gl::GL_LINEAR));
  }
  gl::gl.BindTexture(gl::GL_TEXTURE_2D, 0);
  return texture;
}

gl::GLuint load_texture_or_fallback(const std::filesystem::path& path,
                                    const std::array<std::uint8_t, 4>& fallback) {
  try {
    const TgaImage image = load_tga(path);
    return create_texture_rgba8(image.width, image.height, image.rgba.data(), true,
                                gl::GL_REPEAT);
  } catch (const AppError& error) {
    log_error(ErrorCode::E_TEXTURE_LOAD,
              std::string(error.what()) + "; using deterministic fallback texture");
    return create_texture_rgba8(1, 1, fallback.data(), false, gl::GL_CLAMP_TO_EDGE);
  }
}

}  // namespace

Renderer::Renderer(std::filesystem::path shaderDirectory, int width, int height)
    : shaderDirectory_(std::move(shaderDirectory)), width_(width), height_(height) {
  if (width_ <= 0 || height_ <= 0) {
    throw AppError(ErrorCode::E_ARGUMENT, "renderer dimensions must be positive");
  }
}

Renderer::~Renderer() {
  destroy_textures();
  destroy_gbuffer();
  if (ebo_ != 0) {
    gl::gl.DeleteBuffers(1, &ebo_);
  }
  if (vbo_ != 0) {
    gl::gl.DeleteBuffers(1, &vbo_);
  }
  if (vao_ != 0) {
    gl::gl.DeleteVertexArrays(1, &vao_);
  }
  if (gbufferProgram_ != 0) {
    gl::gl.DeleteProgram(gbufferProgram_);
  }
  if (deferredProgram_ != 0) {
    gl::gl.DeleteProgram(deferredProgram_);
  }
}

void Renderer::initialize(const CharacterSettings& settings) {
  gl::GLint majorVersion = 0;
  gl::GLint minorVersion = 0;
  gl::gl.GetIntegerv(gl::GL_MAJOR_VERSION, &majorVersion);
  gl::gl.GetIntegerv(gl::GL_MINOR_VERSION, &minorVersion);
  if (majorVersion < 4 || (majorVersion == 4 && minorVersion < 3)) {
    const auto* version = reinterpret_cast<const char*>(gl::gl.GetString(gl::GL_VERSION));
    throw AppError(ErrorCode::E_RENDER_CONTEXT_CREATE,
                   std::string("OpenGL 4.3 is required, active version: ") +
                       (version == nullptr ? "unknown" : version));
  }

  create_programs();
  mesh_ = generate_character_mesh(settings);
  create_mesh();
  create_gbuffer();
  log_info(std::string("OpenGL renderer: ") +
           reinterpret_cast<const char*>(gl::gl.GetString(gl::GL_RENDERER)) + ", GLSL " +
           reinterpret_cast<const char*>(gl::gl.GetString(gl::GL_SHADING_LANGUAGE_VERSION)));
}

void Renderer::create_programs() {
  gbufferProgram_ =
      link_program(shaderDirectory_ / "gbuffer.vert", shaderDirectory_ / "gbuffer.frag");
  deferredProgram_ =
      link_program(shaderDirectory_ / "deferred.vert", shaderDirectory_ / "deferred.frag");

  gbufferUniforms_.joints =
      gl::gl.GetUniformLocation(gbufferProgram_, "uJoints[0]");
  gbufferUniforms_.model = gl::gl.GetUniformLocation(gbufferProgram_, "uModel");
  gbufferUniforms_.normalMatrix =
      gl::gl.GetUniformLocation(gbufferProgram_, "uNormalMatrix");
  gbufferUniforms_.view = gl::gl.GetUniformLocation(gbufferProgram_, "uView");
  gbufferUniforms_.projection =
      gl::gl.GetUniformLocation(gbufferProgram_, "uProjection");
  gbufferUniforms_.layerWeights =
      gl::gl.GetUniformLocation(gbufferProgram_, "uLayerWeights");

  lightUniforms_.count = gl::gl.GetUniformLocation(deferredProgram_, "uLightCount");
  lightUniforms_.type = gl::gl.GetUniformLocation(deferredProgram_, "uLightType[0]");
  lightUniforms_.position =
      gl::gl.GetUniformLocation(deferredProgram_, "uLightPosition[0]");
  lightUniforms_.direction =
      gl::gl.GetUniformLocation(deferredProgram_, "uLightDirection[0]");
  lightUniforms_.color = gl::gl.GetUniformLocation(deferredProgram_, "uLightColor[0]");
  lightUniforms_.intensity =
      gl::gl.GetUniformLocation(deferredProgram_, "uLightIntensity[0]");
  lightUniforms_.range = gl::gl.GetUniformLocation(deferredProgram_, "uLightRange[0]");
  lightUniforms_.innerCos =
      gl::gl.GetUniformLocation(deferredProgram_, "uLightInnerCos[0]");
  lightUniforms_.outerCos =
      gl::gl.GetUniformLocation(deferredProgram_, "uLightOuterCos[0]");
  lightUniforms_.attenuation =
      gl::gl.GetUniformLocation(deferredProgram_, "uLightAttenuation[0]");
  lightUniforms_.cameraPosition =
      gl::gl.GetUniformLocation(deferredProgram_, "uCameraPosition");

  gPosition_ = gl::gl.GetUniformLocation(deferredProgram_, "uGPosition");
  gNormal_ = gl::gl.GetUniformLocation(deferredProgram_, "uGNormal");
  gAlbedo_ = gl::gl.GetUniformLocation(deferredProgram_, "uGAlbedo");
  gLayerWeights_ = gl::gl.GetUniformLocation(deferredProgram_, "uGLayerWeights");
  gRoughnessMetallic_ =
      gl::gl.GetUniformLocation(deferredProgram_, "uGRoughnessMetallic");

  gl::gl.UseProgram(gbufferProgram_);
  set_sampler_array(gbufferProgram_, "uBaseColorTextures", 0, 3);
  set_sampler_array(gbufferProgram_, "uNormalTextures", 3, 3);
  set_sampler_array(gbufferProgram_, "uRoughnessTextures", 6, 3);
  set_sampler_array(gbufferProgram_, "uMetallicTextures", 9, 3);

  gl::gl.UseProgram(deferredProgram_);
  gl::gl.Uniform1i(gPosition_, 0);
  gl::gl.Uniform1i(gNormal_, 1);
  gl::gl.Uniform1i(gAlbedo_, 2);
  gl::gl.Uniform1i(gLayerWeights_, 3);
  gl::gl.Uniform1i(gRoughnessMetallic_, 4);
  gl::gl.UseProgram(0);
}

void Renderer::set_sampler_array(gl::GLuint program, const char* name, int firstUnit,
                                 int count) {
  const std::string uniformName = std::string(name) + "[0]";
  const gl::GLint location = gl::gl.GetUniformLocation(program, uniformName.c_str());
  if (location < 0) {
    return;
  }
  std::array<gl::GLint, 16> units{};
  for (int index = 0; index < count; ++index) {
    units[static_cast<std::size_t>(index)] = firstUnit + index;
  }
  gl::gl.Uniform1iv(location, count, units.data());
}

void Renderer::create_mesh() {
  gl::gl.GenVertexArrays(1, &vao_);
  gl::gl.GenBuffers(1, &vbo_);
  gl::gl.GenBuffers(1, &ebo_);
  gl::gl.BindVertexArray(vao_);
  gl::gl.BindBuffer(gl::GL_ARRAY_BUFFER, vbo_);
  gl::gl.BufferData(gl::GL_ARRAY_BUFFER,
                    static_cast<gl::GLsizeiptr>(mesh_.vertices.size() * sizeof(SkinnedVertex)),
                    mesh_.vertices.data(), gl::GL_STATIC_DRAW);
  gl::gl.BindBuffer(gl::GL_ELEMENT_ARRAY_BUFFER, ebo_);
  gl::gl.BufferData(
      gl::GL_ELEMENT_ARRAY_BUFFER,
      static_cast<gl::GLsizeiptr>(mesh_.indices.size() * sizeof(std::uint32_t)),
      mesh_.indices.data(), gl::GL_STATIC_DRAW);

  const auto stride = static_cast<gl::GLsizei>(sizeof(SkinnedVertex));
  gl::gl.EnableVertexAttribArray(0);
  gl::gl.VertexAttribPointer(0, 3, gl::GL_FLOAT, gl::GL_FALSE, stride,
                             reinterpret_cast<const void*>(offsetof(SkinnedVertex, position)));
  gl::gl.EnableVertexAttribArray(1);
  gl::gl.VertexAttribPointer(1, 3, gl::GL_FLOAT, gl::GL_FALSE, stride,
                             reinterpret_cast<const void*>(offsetof(SkinnedVertex, normal)));
  gl::gl.EnableVertexAttribArray(2);
  gl::gl.VertexAttribPointer(2, 3, gl::GL_FLOAT, gl::GL_FALSE, stride,
                             reinterpret_cast<const void*>(offsetof(SkinnedVertex, tangent)));
  gl::gl.EnableVertexAttribArray(3);
  gl::gl.VertexAttribPointer(3, 2, gl::GL_FLOAT, gl::GL_FALSE, stride,
                             reinterpret_cast<const void*>(offsetof(SkinnedVertex, uv)));
  gl::gl.EnableVertexAttribArray(4);
  gl::gl.VertexAttribIPointer(4, 4, gl::GL_UNSIGNED_INT, stride,
                              reinterpret_cast<const void*>(offsetof(SkinnedVertex, joints)));
  gl::gl.EnableVertexAttribArray(5);
  gl::gl.VertexAttribPointer(5, 4, gl::GL_FLOAT, gl::GL_FALSE, stride,
                             reinterpret_cast<const void*>(offsetof(SkinnedVertex, weights)));
  gl::gl.BindVertexArray(0);
  gl::gl.BindBuffer(gl::GL_ARRAY_BUFFER, 0);
  gl::gl.BindBuffer(gl::GL_ELEMENT_ARRAY_BUFFER, 0);

  log_info("generated skinned character: " + std::to_string(mesh_.vertices.size()) +
           " vertices, " + std::to_string(mesh_.indices.size() / 3U) + " triangles, " +
           std::to_string(mesh_.inverseBindMatrices.size()) + " bones");
}

void Renderer::create_gbuffer() {
  destroy_gbuffer();
  gl::gl.GenFramebuffers(1, &gbufferFbo_);
  gl::gl.BindFramebuffer(gl::GL_FRAMEBUFFER, gbufferFbo_);
  gl::gl.GenTextures(static_cast<gl::GLsizei>(gbufferTextures_.size()),
                     gbufferTextures_.data());

  struct Attachment {
    gl::GLenum internalFormat;
    gl::GLenum format;
    gl::GLenum type;
  };
  const std::array<Attachment, 5> attachments = {{
      {gl::GL_RGBA32F, gl::GL_RGBA, gl::GL_FLOAT},
      {gl::GL_RGBA16F, gl::GL_RGBA, gl::GL_FLOAT},
      {gl::GL_RGBA8, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE},
      {gl::GL_RGBA16F, gl::GL_RGBA, gl::GL_FLOAT},
      {gl::GL_RG16F, gl::GL_RG, gl::GL_FLOAT},
  }};
  for (std::size_t index = 0; index < attachments.size(); ++index) {
    gl::gl.BindTexture(gl::GL_TEXTURE_2D, gbufferTextures_[index]);
    gl::gl.TexImage2D(gl::GL_TEXTURE_2D, 0,
                      static_cast<gl::GLint>(attachments[index].internalFormat), width_,
                      height_, 0, attachments[index].format, attachments[index].type,
                      nullptr);
    gl::gl.TexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MIN_FILTER,
                         static_cast<gl::GLint>(gl::GL_NEAREST));
    gl::gl.TexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MAG_FILTER,
                         static_cast<gl::GLint>(gl::GL_NEAREST));
    gl::gl.TexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_WRAP_S,
                         static_cast<gl::GLint>(gl::GL_CLAMP_TO_EDGE));
    gl::gl.TexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_WRAP_T,
                         static_cast<gl::GLint>(gl::GL_CLAMP_TO_EDGE));
    gl::gl.FramebufferTexture2D(
        gl::GL_FRAMEBUFFER,
        static_cast<gl::GLenum>(static_cast<int>(gl::GL_COLOR_ATTACHMENT0) +
                                static_cast<int>(index)),
        gl::GL_TEXTURE_2D, gbufferTextures_[index], 0);
  }
  gl::gl.GenRenderbuffers(1, &depthRbo_);
  gl::gl.BindRenderbuffer(gl::GL_RENDERBUFFER, depthRbo_);
  gl::gl.RenderbufferStorage(gl::GL_RENDERBUFFER, gl::GL_DEPTH_COMPONENT24, width_, height_);
  gl::gl.FramebufferRenderbuffer(gl::GL_FRAMEBUFFER, gl::GL_DEPTH_ATTACHMENT,
                                 gl::GL_RENDERBUFFER, depthRbo_);

  const std::array<gl::GLenum, 5> drawBuffers = {
      gl::GL_COLOR_ATTACHMENT0, gl::GL_COLOR_ATTACHMENT1, gl::GL_COLOR_ATTACHMENT2,
      gl::GL_COLOR_ATTACHMENT3, gl::GL_COLOR_ATTACHMENT4};
  gl::gl.DrawBuffers(static_cast<gl::GLsizei>(drawBuffers.size()), drawBuffers.data());
  const gl::GLenum status = gl::gl.CheckFramebufferStatus(gl::GL_FRAMEBUFFER);
  gl::gl.BindFramebuffer(gl::GL_FRAMEBUFFER, 0);
  gl::gl.BindTexture(gl::GL_TEXTURE_2D, 0);
  gl::gl.BindRenderbuffer(gl::GL_RENDERBUFFER, 0);
  if (status != gl::GL_FRAMEBUFFER_COMPLETE) {
    throw AppError(ErrorCode::E_FRAMEBUFFER_INCOMPLETE,
                   "G-Buffer framebuffer is incomplete, status=" +
                       std::to_string(static_cast<int>(status)));
  }
}

void Renderer::destroy_gbuffer() {
  if (depthRbo_ != 0) {
    gl::gl.DeleteRenderbuffers(1, &depthRbo_);
    depthRbo_ = 0;
  }
  for (gl::GLuint& texture : gbufferTextures_) {
    if (texture != 0) {
      gl::gl.DeleteTextures(1, &texture);
      texture = 0;
    }
  }
  if (gbufferFbo_ != 0) {
    gl::gl.DeleteFramebuffers(1, &gbufferFbo_);
    gbufferFbo_ = 0;
  }
}

void Renderer::apply_material(const Material& material,
                              const std::filesystem::path& assetRoot) {
  TextureSet replacement;
  const std::array<std::array<std::uint8_t, 4>, 3> baseFallbacks = {{
      {160, 110, 92, 255}, {128, 140, 154, 255}, {130, 70, 58, 255}}};
  const std::array<std::uint8_t, 4> normalFallback = {128, 128, 255, 255};
  const std::array<std::uint8_t, 4> roughnessFallback = {170, 170, 170, 255};
  const std::array<std::uint8_t, 4> metallicFallback = {20, 20, 20, 255};
  for (std::size_t layer = 0; layer < 3U; ++layer) {
    const MaterialLayer& materialLayer = material.layers.at(layer);
    replacement.baseColor[layer] = load_texture_or_fallback(
        assetRoot / materialLayer.baseColorTexture, baseFallbacks[layer]);
    replacement.normal[layer] =
        load_texture_or_fallback(assetRoot / materialLayer.normalTexture, normalFallback);
    replacement.roughness[layer] =
        load_texture_or_fallback(assetRoot / materialLayer.roughnessTexture,
                                 roughnessFallback);
    replacement.metallic[layer] =
        load_texture_or_fallback(assetRoot / materialLayer.metallicTexture,
                                 metallicFallback);
  }
  destroy_textures();
  textures_ = replacement;
  log_info("material textures applied: " + material.name);
}

void Renderer::destroy_textures() {
  for (std::array<gl::GLuint, 3>* set :
       {&textures_.baseColor, &textures_.normal, &textures_.roughness, &textures_.metallic}) {
    for (gl::GLuint& texture : *set) {
      if (texture != 0) {
        gl::gl.DeleteTextures(1, &texture);
        texture = 0;
      }
    }
  }
}

void Renderer::set_lighting_state(const LightingState& state) { state_ = state; }

void Renderer::resize(int width, int height) {
  if (width <= 0 || height <= 0 || (width == width_ && height == height_)) {
    return;
  }
  width_ = width;
  height_ = height;
  create_gbuffer();
}

glm::vec3 Renderer::camera_position() const { return glm::vec3(2.8F, 1.65F, 3.9F); }

void Renderer::upload_lights(const std::vector<Light>& lights) {
  std::array<gl::GLint, kMaxLights> types{};
  std::array<gl::GLfloat, kMaxLights * 3> positions{};
  std::array<gl::GLfloat, kMaxLights * 3> directions{};
  std::array<gl::GLfloat, kMaxLights * 3> colors{};
  std::array<gl::GLfloat, kMaxLights> intensities{};
  std::array<gl::GLfloat, kMaxLights> ranges{};
  std::array<gl::GLfloat, kMaxLights> innerCos{};
  std::array<gl::GLfloat, kMaxLights> outerCos{};
  std::array<gl::GLfloat, kMaxLights * 3> attenuation{};

  int count = 0;
  for (const Light& light : lights) {
    if (!light.enabled || count >= kMaxLights) {
      continue;
    }
    const auto index = static_cast<std::size_t>(count);
    types[index] = static_cast<gl::GLint>(light.type);
    const std::size_t vectorIndex = index * 3U;
    positions[vectorIndex] = light.position.x;
    positions[vectorIndex + 1U] = light.position.y;
    positions[vectorIndex + 2U] = light.position.z;
    directions[vectorIndex] = light.direction.x;
    directions[vectorIndex + 1U] = light.direction.y;
    directions[vectorIndex + 2U] = light.direction.z;
    colors[vectorIndex] = light.color.x;
    colors[vectorIndex + 1U] = light.color.y;
    colors[vectorIndex + 2U] = light.color.z;
    intensities[index] = light.intensity;
    ranges[index] = light.range;
    innerCos[index] = std::cos(light.innerConeDegrees * kPi / 180.0F);
    outerCos[index] = std::cos(light.outerConeDegrees * kPi / 180.0F);
    attenuation[vectorIndex] = light.constantAttenuation;
    attenuation[vectorIndex + 1U] = light.linearAttenuation;
    attenuation[vectorIndex + 2U] = light.quadraticAttenuation;
    ++count;
  }

  gl::gl.Uniform1i(lightUniforms_.count, count);
  gl::gl.Uniform1iv(lightUniforms_.type, count, types.data());
  gl::gl.Uniform3fv(lightUniforms_.position, count, positions.data());
  gl::gl.Uniform3fv(lightUniforms_.direction, count, directions.data());
  gl::gl.Uniform3fv(lightUniforms_.color, count, colors.data());
  gl::gl.Uniform1fv(lightUniforms_.intensity, count, intensities.data());
  gl::gl.Uniform1fv(lightUniforms_.range, count, ranges.data());
  gl::gl.Uniform1fv(lightUniforms_.innerCos, count, innerCos.data());
  gl::gl.Uniform1fv(lightUniforms_.outerCos, count, outerCos.data());
  gl::gl.Uniform3fv(lightUniforms_.attenuation, count, attenuation.data());
}

void Renderer::render() {
  const glm::vec3 eye = camera_position();
  const glm::mat4 view =
      glm::lookAt(eye, glm::vec3(0.0F, 0.93F, 0.0F), glm::vec3(0.0F, 1.0F, 0.0F));
  const glm::mat4 projection = glm::perspective(
      glm::radians(42.0F), static_cast<float>(width_) / static_cast<float>(height_), 0.05F,
      40.0F);
  const glm::mat4 model(1.0F);
  const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(model)));
  glm::vec3 weights = state_.materialWeights;
  normalize_material_weights(weights);
  const std::array<glm::mat4, kBoneCount> joints =
      compute_skin_matrices(mesh_, state_.animationTime);

  gl::gl.BindFramebuffer(gl::GL_FRAMEBUFFER, gbufferFbo_);
  gl::gl.Viewport(0, 0, width_, height_);
  gl::gl.ClearColor(0.0F, 0.0F, 0.0F, 0.0F);
  gl::gl.Clear(gl::GL_COLOR_BUFFER_BIT | gl::GL_DEPTH_BUFFER_BIT);
  gl::gl.Enable(gl::GL_DEPTH_TEST);
  gl::gl.DepthFunc(gl::GL_LESS);
  gl::gl.DepthMask(gl::GL_TRUE);
  gl::gl.Disable(gl::GL_CULL_FACE);
  gl::gl.Disable(gl::GL_BLEND);
  gl::gl.Disable(gl::GL_FRAMEBUFFER_SRGB);

  gl::gl.UseProgram(gbufferProgram_);
  gl::gl.UniformMatrix4fv(gbufferUniforms_.joints, kBoneCount, gl::GL_FALSE,
                          glm::value_ptr(joints[0]));
  gl::gl.UniformMatrix4fv(gbufferUniforms_.model, 1, gl::GL_FALSE,
                          glm::value_ptr(model));
  gl::gl.UniformMatrix3fv(gbufferUniforms_.normalMatrix, 1, gl::GL_FALSE,
                          glm::value_ptr(normalMatrix));
  gl::gl.UniformMatrix4fv(gbufferUniforms_.view, 1, gl::GL_FALSE,
                          glm::value_ptr(view));
  gl::gl.UniformMatrix4fv(gbufferUniforms_.projection, 1, gl::GL_FALSE,
                          glm::value_ptr(projection));
  gl::gl.Uniform3f(gbufferUniforms_.layerWeights, weights.x, weights.y, weights.z);

  for (int layer = 0; layer < 3; ++layer) {
    const auto index = static_cast<std::size_t>(layer);
    gl::gl.ActiveTexture(static_cast<gl::GLenum>(gl::GL_TEXTURE0 + index));
    gl::gl.BindTexture(gl::GL_TEXTURE_2D, textures_.baseColor[index]);
    gl::gl.ActiveTexture(static_cast<gl::GLenum>(gl::GL_TEXTURE0 + 3U + index));
    gl::gl.BindTexture(gl::GL_TEXTURE_2D, textures_.normal[index]);
    gl::gl.ActiveTexture(static_cast<gl::GLenum>(gl::GL_TEXTURE0 + 6U + index));
    gl::gl.BindTexture(gl::GL_TEXTURE_2D, textures_.roughness[index]);
    gl::gl.ActiveTexture(static_cast<gl::GLenum>(gl::GL_TEXTURE0 + 9U + index));
    gl::gl.BindTexture(gl::GL_TEXTURE_2D, textures_.metallic[index]);
  }

  gl::gl.BindVertexArray(vao_);
  gl::gl.DrawElements(gl::GL_TRIANGLES, static_cast<gl::GLsizei>(mesh_.indices.size()),
                      gl::GL_UNSIGNED_INT, nullptr);

  gl::gl.BindFramebuffer(gl::GL_FRAMEBUFFER, 0);
  gl::gl.Viewport(0, 0, width_, height_);
  gl::gl.Disable(gl::GL_DEPTH_TEST);
  gl::gl.DepthMask(gl::GL_TRUE);
  gl::gl.Disable(gl::GL_CULL_FACE);
  gl::gl.Disable(gl::GL_BLEND);
  gl::gl.ClearColor(0.018F, 0.022F, 0.030F, 1.0F);
  gl::gl.Clear(gl::GL_COLOR_BUFFER_BIT);

  gl::gl.UseProgram(deferredProgram_);
  for (int index = 0; index < 5; ++index) {
    gl::gl.ActiveTexture(static_cast<gl::GLenum>(gl::GL_TEXTURE0 + static_cast<unsigned>(index)));
    gl::gl.BindTexture(gl::GL_TEXTURE_2D,
                       gbufferTextures_[static_cast<std::size_t>(index)]);
  }
  const glm::vec3 eyeValue = camera_position();
  gl::gl.Uniform3f(lightUniforms_.cameraPosition, eyeValue.x, eyeValue.y, eyeValue.z);
  upload_lights(state_.lights);
  gl::gl.DrawArrays(gl::GL_TRIANGLES, 0, 3);

  gl::gl.BindFramebuffer(gl::GL_FRAMEBUFFER, 0);
  gl::gl.UseProgram(0);
  gl::gl.BindVertexArray(0);
  for (unsigned int unit = 0; unit < 12U; ++unit) {
    gl::gl.ActiveTexture(gl::GL_TEXTURE0 + unit);
    gl::gl.BindTexture(gl::GL_TEXTURE_2D, 0);
  }
  gl::gl.ActiveTexture(gl::GL_TEXTURE0);
  gl::gl.Disable(gl::GL_DEPTH_TEST);
  gl::gl.DepthMask(gl::GL_TRUE);
  gl::gl.Disable(gl::GL_CULL_FACE);
  gl::gl.Disable(gl::GL_BLEND);
  gl::gl.Disable(gl::GL_SCISSOR_TEST);
  gl::gl.Disable(gl::GL_FRAMEBUFFER_SRGB);
  gl::gl.Viewport(0, 0, width_, height_);
}

std::vector<std::uint8_t> Renderer::read_pixels() const {
  std::vector<std::uint8_t> pixels(
      static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * 4U);
  gl::gl.Finish();
  gl::gl.PixelStorei(gl::GL_PACK_ALIGNMENT, 1);
  gl::gl.ReadPixels(0, 0, width_, height_, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE,
                    pixels.data());
  return pixels;
}

}  // namespace lf

