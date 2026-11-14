#pragma once

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace lf {

constexpr int kBoneCount = 18;
constexpr int kPoseKeyframeCount = 2;

enum BoneIndex : int {
  Root = 0,
  Pelvis,
  Spine,
  Chest,
  Neck,
  Head,
  UpperArmL,
  LowerArmL,
  HandL,
  UpperArmR,
  LowerArmR,
  HandR,
  ThighL,
  CalfL,
  FootL,
  ThighR,
  CalfR,
  FootR,
};

struct BoneDefinition {
  const char* name = "";
  int parent = -1;
  glm::vec3 localTranslation{0.0F};
};

struct PoseKeyframe {
  const char* name = "";
  std::array<glm::vec3, kBoneCount> localEulerRadians{};
};

struct CharacterSettings {
  float height = 1.80F;
  float shoulderWidth = 0.62F;
  float limbProportion = 1.0F;
};

struct SkinnedVertex {
  glm::vec3 position{0.0F};
  glm::vec3 normal{0.0F, 1.0F, 0.0F};
  glm::vec3 tangent{1.0F, 0.0F, 0.0F};
  glm::vec2 uv{0.0F};
  glm::ivec4 joints{0};
  glm::vec4 weights{1.0F, 0.0F, 0.0F, 0.0F};
};

struct SkinnedMesh {
  CharacterSettings settings;
  std::array<BoneDefinition, kBoneCount> bones{};
  std::array<glm::mat4, kBoneCount> inverseBindMatrices{};
  std::vector<SkinnedVertex> vertices;
  std::vector<std::uint32_t> indices;
};

const std::array<PoseKeyframe, kPoseKeyframeCount>& built_in_pose_keyframes();
const std::array<BoneDefinition, kBoneCount>& built_in_bone_definitions(
    const CharacterSettings& settings = {});

SkinnedMesh generate_character_mesh(const CharacterSettings& settings);
std::array<glm::mat4, kBoneCount> compute_skin_matrices(const SkinnedMesh& mesh,
                                                        float animationTime);

}  // namespace lf

