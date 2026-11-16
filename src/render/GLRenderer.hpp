#pragma once

#include <glad/glad.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/CapeSorter.hpp"
#include "core/Equipment.hpp"
#include "core/Mesh.hpp"
#include "core/Report.hpp"

namespace lcr {

class ShaderProgram;

struct OrbitCamera {
  float yaw = 0.58F;
  float pitch = 0.13F;
  float distance = 4.25F;
  float targetX = 0.0F;
  float targetY = 1.05F;
  float targetZ = 0.0F;

  Vec3 position() const;
  Vec3 target() const;
};

class GLRenderer {
 public:
  GLRenderer();
  ~GLRenderer();

  GLRenderer(const GLRenderer&) = delete;
  GLRenderer& operator=(const GLRenderer&) = delete;

  void initialize(
      const CharacterAssets& assets,
      const EquipmentCatalog& equipment,
      bool forceShaderFailure,
      int width,
      int height);
  void resize(int width, int height);
  void render(
      const Pose& pose,
      const EquipmentSet& equipment,
      std::size_t equipmentIndex,
      const OrbitCamera& camera);

  const PipelineReport& pipelineReport() const { return pipeline_; }

 private:
  struct GpuMesh {
    std::string name;
    MaterialSlot materialSlot = MaterialSlot::Body;
    bool transparent = false;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLuint indexCount = 0;
    Mesh mesh;
  };

  struct GpuTextureSet {
    GLuint body = 0;
    GLuint weapon = 0;
  };

  void uploadMesh(const MeshPart& part);
  void uploadTexture(const TextureData& texture, GLuint& id);
  void drawOpaqueParts(
      const EquipmentSet& equipment,
      std::size_t equipmentIndex);
  void drawCape(
      const EquipmentSet& equipment,
      std::size_t equipmentIndex,
      const std::vector<Mat4>& skinMatrices,
      const Vec3 cameraPosition);
  void updateBonePalette(const std::vector<Mat4>& skinMatrices);
  void applyMaterial(const Material& material, const bool capeMode);

  std::vector<GpuMesh> meshes_;
  std::vector<GpuTextureSet> textures_;
  std::size_t capeMeshIndex_ = 0;
  const CharacterAssets* assets_ = nullptr;
  GLuint shaderProgram_ = 0;
  int width_ = 1;
  int height_ = 1;
  PipelineReport pipeline_;

  std::unique_ptr<ShaderProgram> shaderObject_;
};

}  // namespace lcr
