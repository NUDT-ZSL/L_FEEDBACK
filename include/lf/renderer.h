#pragma once

#include "lf/config.h"
#include "lf/gl_api.h"
#include "lf/mesh.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace lf {

class Renderer {
 public:
  Renderer(std::filesystem::path shaderDirectory, int width, int height);
  ~Renderer();

  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;

  void initialize(const CharacterSettings& settings);
  void apply_material(const Material& material, const std::filesystem::path& assetRoot);
  void set_lighting_state(const LightingState& state);
  void resize(int width, int height);
  void render();

  [[nodiscard]] std::vector<std::uint8_t> read_pixels() const;
  [[nodiscard]] glm::vec3 camera_position() const;
  [[nodiscard]] int width() const noexcept { return width_; }
  [[nodiscard]] int height() const noexcept { return height_; }

 private:
  struct TextureSet {
    std::array<gl::GLuint, 3> baseColor{};
    std::array<gl::GLuint, 3> normal{};
    std::array<gl::GLuint, 3> roughness{};
    std::array<gl::GLuint, 3> metallic{};
  };

  struct GbufferUniforms {
    gl::GLint joints = -1;
    gl::GLint model = -1;
    gl::GLint normalMatrix = -1;
    gl::GLint view = -1;
    gl::GLint projection = -1;
    gl::GLint layerWeights = -1;
  };

  struct LightUniforms {
    gl::GLint count = -1;
    gl::GLint type = -1;
    gl::GLint position = -1;
    gl::GLint direction = -1;
    gl::GLint color = -1;
    gl::GLint intensity = -1;
    gl::GLint range = -1;
    gl::GLint innerCos = -1;
    gl::GLint outerCos = -1;
    gl::GLint attenuation = -1;
    gl::GLint cameraPosition = -1;
  };

  void create_programs();
  void create_mesh();
  void create_gbuffer();
  void destroy_gbuffer();
  void upload_textures(const Material& material, const std::filesystem::path& assetRoot);
  void destroy_textures();
  void upload_lights(const std::vector<Light>& lights);
  void set_sampler_array(gl::GLuint program, const char* name, int firstUnit, int count);

  std::filesystem::path shaderDirectory_;
  int width_ = 0;
  int height_ = 0;
  SkinnedMesh mesh_;
  TextureSet textures_{};
  LightingState state_{};

  gl::GLuint gbufferProgram_ = 0;
  gl::GLuint deferredProgram_ = 0;
  gl::GLuint vao_ = 0;
  gl::GLuint vbo_ = 0;
  gl::GLuint ebo_ = 0;
  gl::GLuint gbufferFbo_ = 0;
  std::array<gl::GLuint, 5> gbufferTextures_{};
  gl::GLuint depthRbo_ = 0;
  GbufferUniforms gbufferUniforms_{};
  LightUniforms lightUniforms_{};
  gl::GLint gPosition_ = -1;
  gl::GLint gNormal_ = -1;
  gl::GLint gAlbedo_ = -1;
  gl::GLint gLayerWeights_ = -1;
  gl::GLint gRoughnessMetallic_ = -1;
};

}  // namespace lf

