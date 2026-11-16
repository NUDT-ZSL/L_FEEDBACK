#include "render/GLRenderer.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "core/Error.hpp"
#include "core/Serializer.hpp"
#include "render/ShaderProgram.hpp"
#include "render/ShaderSources.hpp"

namespace lcr {
namespace {

constexpr GLuint kMaxShaderBones = 24;
static_assert(kBoneCount <= static_cast<int>(kMaxShaderBones), "shader bone palette is too small");
static_assert(sizeof(Vertex) == 64U, "Vertex layout must remain packed and explicit");

void checkGlError(const char* stage) {
  const GLenum error = glGetError();
  if (error != GL_NO_ERROR) {
    fail(
        ErrorCode::GpuInit,
        stage,
        "OpenGL error 0x" + std::to_string(static_cast<unsigned int>(error)));
  }
}

}  // namespace

Vec3 OrbitCamera::position() const {
  const float horizontal = std::cos(pitch) * distance;
  return {
      targetX + std::sin(yaw) * horizontal,
      targetY + std::sin(pitch) * distance,
      targetZ - std::cos(yaw) * horizontal,
  };
}

Vec3 OrbitCamera::target() const { return {targetX, targetY, targetZ}; }

GLRenderer::GLRenderer() = default;

GLRenderer::~GLRenderer() {
  shaderObject_.reset();
  for (const GpuMesh& mesh : meshes_) {
    if (mesh.ebo != 0U) {
      glDeleteBuffers(1, &mesh.ebo);
    }
    if (mesh.vbo != 0U) {
      glDeleteBuffers(1, &mesh.vbo);
    }
    if (mesh.vao != 0U) {
      glDeleteVertexArrays(1, &mesh.vao);
    }
  }
  for (const GpuTextureSet& texture : textures_) {
    if (texture.body != 0U) {
      glDeleteTextures(1, &texture.body);
    }
    if (texture.weapon != 0U) {
      glDeleteTextures(1, &texture.weapon);
    }
  }
}

void GLRenderer::initialize(
    const CharacterAssets& assets,
    const EquipmentCatalog& equipment,
    const bool forceShaderFailure,
    const int width,
    const int height) {
  width_ = std::max(1, width);
  height_ = std::max(1, height);
  assets_ = &assets;
  glViewport(0, 0, width_, height_);
  glDisable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glEnable(GL_MULTISAMPLE);
  glClearColor(0.027F, 0.032F, 0.047F, 1.0F);

  shaderObject_ = std::make_unique<ShaderProgram>(
      kCharacterVertexShader,
      kCharacterFragmentShader,
      forceShaderFailure);
  shaderProgram_ = shaderObject_->id();
  ++pipeline_.shaderProgramBuilds;
  pipeline_.initialized = true;
  const auto* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
  const auto* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
  pipeline_.openGlVersion = version != nullptr ? version : "<unknown>";
  pipeline_.rendererName = renderer != nullptr ? renderer : "<unknown>";

  for (std::size_t i = 0; i < equipment.sets().size(); ++i) {
    GpuTextureSet texture;
    uploadTexture(equipment.sets()[i].bodyTexture, texture.body);
    uploadTexture(equipment.sets()[i].weaponTexture, texture.weapon);
    textures_.push_back(texture);
  }
  for (const MeshPart& part : assets.parts) {
    uploadMesh(part);
    if (part.name == "Cape") {
      capeMeshIndex_ = meshes_.size() - 1U;
    }
  }
  if (textures_.size() != equipment.sets().size()) {
    fail(ErrorCode::GpuInit, "gpu.textures", "incomplete GPU texture set");
  }
  glUseProgram(shaderProgram_);
  shaderObject_->setInt("uBaseTexture", 0);
  shaderObject_->setVec3("uMainLightDirection", -0.38F, 0.82F, -0.48F);
  shaderObject_->setVec3("uMainLightColor", 1.28F, 1.12F, 0.94F);
  shaderObject_->setVec3("uFillLightDirection", 0.68F, 0.26F, 0.46F);
  shaderObject_->setVec3("uFillLightColor", 0.28F, 0.40F, 0.67F);
  shaderObject_->setVec3("uAmbientColor", 0.16F, 0.17F, 0.20F);
  shaderObject_->setFloat("uFeatherStart", 0.018F);
  shaderObject_->setFloat("uFeatherWidth", 0.105F);
  glUseProgram(0);
  checkGlError("gpu.initialize");
}

void GLRenderer::uploadMesh(const MeshPart& part) {
  GpuMesh gpu;
  gpu.name = part.name;
  gpu.materialSlot = part.materialSlot;
  gpu.transparent = part.transparent;
  gpu.indexCount = static_cast<GLuint>(part.mesh.indices.size());
  gpu.mesh = part.mesh;

  glGenVertexArrays(1, &gpu.vao);
  glGenBuffers(1, &gpu.vbo);
  glGenBuffers(1, &gpu.ebo);
  if (gpu.vao == 0U || gpu.vbo == 0U || gpu.ebo == 0U) {
    fail(ErrorCode::GpuInit, "gpu.mesh", "failed to allocate buffers for " + part.name);
  }
  glBindVertexArray(gpu.vao);
  glBindBuffer(GL_ARRAY_BUFFER, gpu.vbo);
  glBufferData(
      GL_ARRAY_BUFFER,
      static_cast<GLsizeiptr>(part.mesh.vertices.size() * sizeof(Vertex)),
      part.mesh.vertices.data(),
      GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpu.ebo);
  glBufferData(
      GL_ELEMENT_ARRAY_BUFFER,
      static_cast<GLsizeiptr>(part.mesh.indices.size() * sizeof(std::uint32_t)),
      part.mesh.indices.data(),
      part.transparent ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        reinterpret_cast<const void*>(offsetof(Vertex, position)));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        reinterpret_cast<const void*>(offsetof(Vertex, normal)));
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        reinterpret_cast<const void*>(offsetof(Vertex, uv)));
  glEnableVertexAttribArray(3);
  glVertexAttribIPointer(3, 4, GL_INT, sizeof(Vertex),
                         reinterpret_cast<const void*>(offsetof(Vertex, joints)));
  glEnableVertexAttribArray(4);
  glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        reinterpret_cast<const void*>(offsetof(Vertex, weights)));
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  meshes_.push_back(std::move(gpu));
  ++pipeline_.vaoBuilds;
}

void GLRenderer::uploadTexture(const TextureData& texture, GLuint& id) {
  if (texture.width == 0U || texture.height == 0U ||
      texture.rgba.size() !=
          static_cast<std::size_t>(texture.width) * texture.height * 4U) {
    fail(ErrorCode::TextureGeneration, "gpu.texture", "invalid CPU texture payload");
  }
  glGenTextures(1, &id);
  if (id == 0U) {
    fail(ErrorCode::GpuInit, "gpu.texture", "glGenTextures failed");
  }
  glBindTexture(GL_TEXTURE_2D, id);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(
      GL_TEXTURE_2D,
      0,
      GL_RGBA8,
      static_cast<GLsizei>(texture.width),
      static_cast<GLsizei>(texture.height),
      0,
      GL_RGBA,
      GL_UNSIGNED_BYTE,
      texture.rgba.data());
  glGenerateMipmap(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, 0);
  ++pipeline_.textureUploads;
}

void GLRenderer::resize(const int width, const int height) {
  width_ = std::max(1, width);
  height_ = std::max(1, height);
  glViewport(0, 0, width_, height_);
}

void GLRenderer::updateBonePalette(const std::vector<Mat4>& skinMatrices) {
  if (skinMatrices.size() > kMaxShaderBones) {
    fail(ErrorCode::BoneIndex, "gpu.bonePalette", "bone palette exceeds shader capacity");
  }
  std::vector<float> packed;
  packed.reserve(skinMatrices.size() * 16U);
  for (const Mat4& matrix : skinMatrices) {
    packed.insert(packed.end(), matrix.m.begin(), matrix.m.end());
  }
  glUniformMatrix4fv(
      shaderObject_->uniform("uBones[0]"),
      static_cast<GLsizei>(skinMatrices.size()),
      GL_FALSE,
      packed.data());
}

void GLRenderer::applyMaterial(const Material& material, const bool capeMode) {
  shaderObject_->setVec3(
      "uBaseColor", material.baseColor.x, material.baseColor.y, material.baseColor.z);
  shaderObject_->setFloat("uMetallic", material.metallic);
  shaderObject_->setFloat("uRoughness", material.roughness);
  shaderObject_->setVec3(
      "uEmissiveColor",
      material.emissiveColor.x,
      material.emissiveColor.y,
      material.emissiveColor.z);
  shaderObject_->setFloat("uEmissiveStrength", material.emissiveStrength);
  shaderObject_->setFloat("uOpacity", material.opacity);
  shaderObject_->setFloat("uCapeMode", capeMode ? 1.0F : 0.0F);
}

void GLRenderer::drawOpaqueParts(
    const EquipmentSet& equipment,
    const std::size_t equipmentIndex) {
  const GpuTextureSet& textureSet = textures_.at(equipmentIndex);
  for (const GpuMesh& mesh : meshes_) {
    if (mesh.transparent) {
      continue;
    }
    const bool weapon = mesh.materialSlot == MaterialSlot::Weapon;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, weapon ? textureSet.weapon : textureSet.body);
    applyMaterial(weapon ? equipment.weaponMaterial : equipment.bodyMaterial, false);
    glBindVertexArray(mesh.vao);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(mesh.indexCount), GL_UNSIGNED_INT, nullptr);
  }
}

void GLRenderer::drawCape(
    const EquipmentSet& equipment,
    const std::size_t equipmentIndex,
    const std::vector<Mat4>& skinMatrices,
    const Vec3 cameraPosition) {
  if (capeMeshIndex_ >= meshes_.size()) {
    fail(ErrorCode::InternalInvariant, "gpu.cape", "cape mesh was not uploaded");
  }
  GpuMesh& cape = meshes_[capeMeshIndex_];
  const std::vector<Vec3> positions =
      transformedVertexPositions(cape.mesh, skinMatrices);
  const std::vector<std::uint32_t> order =
      sortCapeTrianglesBackToFront(cape.mesh, positions, cameraPosition);
  std::vector<std::uint32_t> sortedIndices;
  sortedIndices.reserve(order.size() * 3U);
  for (const std::uint32_t triangle : order) {
    sortedIndices.push_back(cape.mesh.indices[triangle * 3U]);
    sortedIndices.push_back(cape.mesh.indices[triangle * 3U + 1U]);
    sortedIndices.push_back(cape.mesh.indices[triangle * 3U + 2U]);
  }

  const GpuTextureSet& textureSet = textures_.at(equipmentIndex);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textureSet.body);
  applyMaterial(equipment.capeMaterial, true);
  glBindVertexArray(cape.vao);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cape.ebo);
  glBufferSubData(
      GL_ELEMENT_ARRAY_BUFFER,
      0,
      static_cast<GLsizeiptr>(sortedIndices.size() * sizeof(std::uint32_t)),
      sortedIndices.data());
  glDrawElements(
      GL_TRIANGLES,
      static_cast<GLsizei>(sortedIndices.size()),
      GL_UNSIGNED_INT,
      nullptr);
}

void GLRenderer::render(
    const Pose& pose,
    const EquipmentSet& equipment,
    const std::size_t equipmentIndex,
    const OrbitCamera& camera) {
  if (assets_ == nullptr) {
    fail(ErrorCode::InternalInvariant, "gpu.render", "renderer was not initialized");
  }
  if (equipmentIndex >= textures_.size()) {
    fail(ErrorCode::InvalidArgument, "gpu.render", "equipment index has no uploaded textures");
  }
  glViewport(0, 0, width_, height_);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glUseProgram(shaderProgram_);

  const Vec3 eye = camera.position();
  const Vec3 target = camera.target();
  const glm::vec3 glmEye(eye.x, eye.y, eye.z);
  const glm::vec3 glmTarget(target.x, target.y, target.z);
  const glm::mat4 view = glm::lookAt(glmEye, glmTarget, glm::vec3(0.0F, 1.0F, 0.0F));
  const glm::mat4 projection = glm::perspective(
      glm::radians(42.0F),
      static_cast<float>(width_) / static_cast<float>(height_),
      0.05F,
      100.0F);
  const glm::mat4 viewProjection = projection * view;
  shaderObject_->setMat4("uViewProjection", glm::value_ptr(viewProjection));
  shaderObject_->setVec3("uCameraPosition", eye.x, eye.y, eye.z);

  const std::vector<Mat4> skinMatrices =
      assets_->skeleton.computeSkinMatrices(pose);
  updateBonePalette(skinMatrices);

  glDisable(GL_BLEND);
  glDepthMask(GL_TRUE);
  glEnable(GL_DEPTH_TEST);
  drawOpaqueParts(equipment, equipmentIndex);

  // The opaque pass already populated depth. The cape is sorted back-to-front
  // on the CPU and tested against that depth buffer without writing depth.
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDepthMask(GL_FALSE);
  drawCape(equipment, equipmentIndex, skinMatrices, eye);
  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);
  glBindVertexArray(0);
  glUseProgram(0);
}

}  // namespace lcr
