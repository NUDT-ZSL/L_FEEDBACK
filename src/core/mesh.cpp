#include "lf/mesh.h"

#include "lf/error_code.h"

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace lf {

namespace {

constexpr int kRadialSegments = 18;
constexpr int kCapRings = 4;

struct Ring {
  glm::vec3 center{0.0F};
  float radius = 0.0F;
  float uvV = 0.0F;
  bool cap = false;
  bool startCap = false;
  glm::vec3 sphereCenter{0.0F};
  glm::vec3 axis{0.0F, 1.0F, 0.0F};
  glm::vec3 tangentAxis{1.0F, 0.0F, 0.0F};
  glm::vec3 bitangentAxis{0.0F, 0.0F, 1.0F};
};

struct CapsuleSpec {
  glm::vec3 start{0.0F};
  glm::vec3 end{0.0F};
  float radius = 0.05F;
  int ownerBone = Root;
  int parentBone = Root;
};

float smooth_step(float edge0, float edge1, float value) {
  if (edge0 == edge1) {
    return value < edge0 ? 0.0F : 1.0F;
  }
  const float t = std::clamp((value - edge0) / (edge1 - edge0), 0.0F, 1.0F);
  return t * t * (3.0F - 2.0F * t);
}

glm::mat4 make_local_transform(const glm::vec3& translation, const glm::vec3& euler) {
  glm::mat4 matrix = glm::translate(glm::mat4(1.0F), translation);
  matrix = glm::rotate(matrix, euler.z, glm::vec3(0.0F, 0.0F, 1.0F));
  matrix = glm::rotate(matrix, euler.y, glm::vec3(0.0F, 1.0F, 0.0F));
  matrix = glm::rotate(matrix, euler.x, glm::vec3(1.0F, 0.0F, 0.0F));
  return matrix;
}

std::array<glm::mat4, kBoneCount> compute_global_matrices(
    const std::array<BoneDefinition, kBoneCount>& bones,
    const std::array<glm::vec3, kBoneCount>& rotations) {
  std::array<glm::mat4, kBoneCount> global{glm::mat4(1.0F)};
  for (int index = 0; index < kBoneCount; ++index) {
    const glm::mat4 local =
        make_local_transform(bones[static_cast<std::size_t>(index)].localTranslation,
                             rotations[static_cast<std::size_t>(index)]);
    const int parent = bones[static_cast<std::size_t>(index)].parent;
    if (parent >= 0) {
      global[static_cast<std::size_t>(index)] =
          global[static_cast<std::size_t>(parent)] * local;
    } else {
      global[static_cast<std::size_t>(index)] = local;
    }
  }
  return global;
}

glm::vec3 bone_position(const std::array<glm::mat4, kBoneCount>& matrices, int bone) {
  return glm::vec3(matrices[static_cast<std::size_t>(bone)][3]);
}

std::vector<Ring> build_capsule_rings(const CapsuleSpec& capsule) {
  glm::vec3 axis = capsule.end - capsule.start;
  float length = glm::length(axis);
  if (length < 1.0e-5F) {
    axis = glm::vec3(0.0F, 1.0F, 0.0F);
    length = 0.0F;
  } else {
    axis /= length;
  }

  const glm::vec3 helper =
      std::abs(axis.y) < 0.92F ? glm::vec3(0.0F, 1.0F, 0.0F) : glm::vec3(1.0F, 0.0F, 0.0F);
  const glm::vec3 tangent = glm::normalize(glm::cross(helper, axis));
  const glm::vec3 bitangent = glm::normalize(glm::cross(axis, tangent));

  std::vector<Ring> rings;
  rings.reserve(static_cast<std::size_t>(kCapRings * 2 + 2));
  const glm::vec3 startSphereCenter = capsule.start - axis * capsule.radius;
  const glm::vec3 endSphereCenter = capsule.end + axis * capsule.radius;

  for (int index = 0; index <= kCapRings; ++index) {
    const float t = static_cast<float>(index) / static_cast<float>(kCapRings);
    const float theta = -glm::half_pi<float>() + t * glm::half_pi<float>();
    const float radial = std::cos(theta);
    Ring ring;
    ring.center = startSphereCenter + axis * (capsule.radius * (1.0F + std::sin(theta)));
    ring.radius = capsule.radius * radial;
    ring.uvV = 0.5F * t;
    ring.cap = true;
    ring.startCap = true;
    ring.sphereCenter = startSphereCenter;
    ring.axis = axis;
    ring.tangentAxis = tangent;
    ring.bitangentAxis = bitangent;
    rings.push_back(ring);
  }

  Ring cylinderStart;
  cylinderStart.center = capsule.start;
  cylinderStart.radius = capsule.radius;
  cylinderStart.uvV = 0.5F;
  cylinderStart.axis = axis;
  cylinderStart.tangentAxis = tangent;
  cylinderStart.bitangentAxis = bitangent;
  rings.push_back(cylinderStart);

  Ring cylinderEnd = cylinderStart;
  cylinderEnd.center = capsule.end;
  cylinderEnd.uvV = 1.5F;
  rings.push_back(cylinderEnd);

  for (int index = 1; index <= kCapRings; ++index) {
    const float t = static_cast<float>(index) / static_cast<float>(kCapRings);
    const float theta = t * glm::half_pi<float>();
    const float radial = std::cos(theta);
    Ring ring;
    ring.center = capsule.end + axis * (capsule.radius * std::sin(theta));
    ring.radius = capsule.radius * radial;
    ring.uvV = 1.5F + 0.5F * t;
    ring.cap = true;
    ring.startCap = false;
    ring.sphereCenter = endSphereCenter;
    ring.axis = axis;
    ring.tangentAxis = tangent;
    ring.bitangentAxis = bitangent;
    rings.push_back(ring);
  }
  return rings;
}

void append_capsule(const CapsuleSpec& capsule, SkinnedMesh& mesh) {
  const std::vector<Ring> rings = build_capsule_rings(capsule);
  const std::uint32_t baseVertex = static_cast<std::uint32_t>(mesh.vertices.size());
  const int owner = std::clamp(capsule.ownerBone, 0, kBoneCount - 1);
  const int parent = std::clamp(capsule.parentBone, 0, kBoneCount - 1);

  for (std::size_t ringIndex = 0; ringIndex < rings.size(); ++ringIndex) {
    const Ring& ring = rings[ringIndex];
    const float t = rings.size() > 1U
                        ? static_cast<float>(ringIndex) / static_cast<float>(rings.size() - 1U)
                        : 0.0F;
    const float ownerWeight = 0.68F + 0.32F * smooth_step(0.0F, 1.0F, t);
    const float parentWeight = 1.0F - ownerWeight;
    for (int segment = 0; segment <= kRadialSegments; ++segment) {
      const float angle = static_cast<float>(segment) /
                          static_cast<float>(kRadialSegments) * glm::two_pi<float>();
      const float cosine = std::cos(angle);
      const float sine = std::sin(angle);
      const glm::vec3 radial =
          ring.tangentAxis * cosine + ring.bitangentAxis * sine;
      const glm::vec3 position = ring.center + radial * ring.radius;

      glm::vec3 normal = radial;
      if (ring.cap && ring.radius > 1.0e-6F) {
        normal = glm::normalize(position - ring.sphereCenter);
      } else if (ring.cap) {
        normal = ring.startCap ? -ring.axis : ring.axis;
      }
      glm::vec3 tangent = -ring.tangentAxis * sine + ring.bitangentAxis * cosine;
      if (glm::length(tangent) < 1.0e-6F) {
        tangent = ring.tangentAxis;
      } else {
        tangent = glm::normalize(tangent);
      }

      SkinnedVertex vertex;
      vertex.position = position;
      vertex.normal = normal;
      vertex.tangent = tangent;
      vertex.uv = glm::vec2(static_cast<float>(segment) /
                                static_cast<float>(kRadialSegments),
                            ring.uvV);
      vertex.joints = glm::ivec4(owner, parent, 0, 0);
      vertex.weights = glm::vec4(ownerWeight, parentWeight, 0.0F, 0.0F);
      mesh.vertices.push_back(vertex);
    }
  }

  for (std::size_t ringIndex = 0; ringIndex + 1U < rings.size(); ++ringIndex) {
    for (int segment = 0; segment < kRadialSegments; ++segment) {
      const std::uint32_t row =
          static_cast<std::uint32_t>(ringIndex) *
          static_cast<std::uint32_t>(kRadialSegments + 1);
      const std::uint32_t nextRow =
          static_cast<std::uint32_t>(ringIndex + 1U) *
          static_cast<std::uint32_t>(kRadialSegments + 1);
      const std::uint32_t a = baseVertex + row + static_cast<std::uint32_t>(segment);
      const std::uint32_t b = a + 1U;
      const std::uint32_t c =
          baseVertex + nextRow + static_cast<std::uint32_t>(segment);
      const std::uint32_t d = c + 1U;
      if (rings[ringIndex].radius > 1.0e-5F || rings[ringIndex + 1U].radius > 1.0e-5F) {
        mesh.indices.insert(mesh.indices.end(), {a, c, b, b, c, d});
      }
    }
  }
}

std::array<BoneDefinition, kBoneCount> make_bones(const CharacterSettings& settings) {
  const float h = settings.height;
  const float s = settings.shoulderWidth;
  const float limb = settings.limbProportion;
  std::array<BoneDefinition, kBoneCount> bones{};

  bones[Root] = {"Root", -1, {0.0F, 0.0F, 0.0F}};
  bones[Pelvis] = {"Pelvis", Root, {0.0F, 0.53F * h, 0.0F}};
  bones[Spine] = {"Spine", Pelvis, {0.0F, 0.10F * h, 0.0F}};
  bones[Chest] = {"Chest", Spine, {0.0F, 0.12F * h, 0.0F}};
  bones[Neck] = {"Neck", Chest, {0.0F, 0.10F * h, 0.0F}};
  bones[Head] = {"Head", Neck, {0.0F, 0.07F * h, 0.0F}};

  bones[UpperArmL] = {"UpperArm.L", Chest, {0.46F * s, 0.035F * h, 0.0F}};
  bones[LowerArmL] = {"LowerArm.L", UpperArmL, {0.035F * h, -0.145F * limb * h, 0.0F}};
  bones[HandL] = {"Hand.L", LowerArmL, {0.015F * h, -0.135F * limb * h, 0.0F}};

  bones[UpperArmR] = {"UpperArm.R", Chest, {-0.46F * s, 0.035F * h, 0.0F}};
  bones[LowerArmR] = {"LowerArm.R", UpperArmR, {-0.035F * h, -0.145F * limb * h, 0.0F}};
  bones[HandR] = {"Hand.R", LowerArmR, {-0.015F * h, -0.135F * limb * h, 0.0F}};

  bones[ThighL] = {"Thigh.L", Pelvis, {0.10F * h, -0.035F * h, 0.0F}};
  bones[CalfL] = {"Calf.L", ThighL, {0.0F, -0.225F * limb * h, 0.0F}};
  bones[FootL] = {"Foot.L", CalfL, {0.0F, -0.205F * limb * h, 0.055F * h}};

  bones[ThighR] = {"Thigh.R", Pelvis, {-0.10F * h, -0.035F * h, 0.0F}};
  bones[CalfR] = {"Calf.R", ThighR, {0.0F, -0.225F * limb * h, 0.0F}};
  bones[FootR] = {"Foot.R", CalfR, {0.0F, -0.205F * limb * h, 0.055F * h}};
  return bones;
}

const std::array<PoseKeyframe, kPoseKeyframeCount> kPoseKeyframes = {{
    {"rest",
     {glm::vec3(0.0F), glm::vec3(0.0F), glm::vec3(0.0F), glm::vec3(0.0F),
      glm::vec3(0.0F), glm::vec3(0.0F), glm::vec3(0.0F), glm::vec3(0.0F),
      glm::vec3(0.0F), glm::vec3(0.0F), glm::vec3(0.0F), glm::vec3(0.0F),
      glm::vec3(0.0F), glm::vec3(0.0F), glm::vec3(0.0F), glm::vec3(0.0F),
      glm::vec3(0.0F), glm::vec3(0.0F)}},
    {"walk",
     {glm::vec3(0.0F), glm::vec3(0.0F, 0.03F, 0.0F), glm::vec3(0.03F, 0.0F, 0.0F),
      glm::vec3(0.0F, -0.04F, 0.0F), glm::vec3(0.0F), glm::vec3(0.0F),
      glm::vec3(-0.50F, 0.0F, -0.06F), glm::vec3(-0.18F, 0.0F, 0.0F),
      glm::vec3(0.0F), glm::vec3(0.55F, 0.0F, 0.06F), glm::vec3(-0.22F, 0.0F, 0.0F),
      glm::vec3(0.0F), glm::vec3(0.62F, 0.0F, 0.0F), glm::vec3(-0.42F, 0.0F, 0.0F),
      glm::vec3(0.12F, 0.0F, 0.0F), glm::vec3(-0.56F, 0.0F, 0.0F),
      glm::vec3(-0.18F, 0.0F, 0.0F), glm::vec3(0.08F, 0.0F, 0.0F)}},
}};

}  // namespace

const std::array<PoseKeyframe, kPoseKeyframeCount>& built_in_pose_keyframes() {
  return kPoseKeyframes;
}

const std::array<BoneDefinition, kBoneCount>& built_in_bone_definitions(
    const CharacterSettings& settings) {
  static thread_local CharacterSettings cachedSettings{};
  static thread_local std::array<BoneDefinition, kBoneCount> cachedBones{};
  if (cachedSettings.height != settings.height ||
      cachedSettings.shoulderWidth != settings.shoulderWidth ||
      cachedSettings.limbProportion != settings.limbProportion) {
    cachedBones = make_bones(settings);
    cachedSettings = settings;
  }
  return cachedBones;
}

SkinnedMesh generate_character_mesh(const CharacterSettings& settings) {
  if (!std::isfinite(settings.height) || settings.height < 1.2F || settings.height > 2.4F ||
      !std::isfinite(settings.shoulderWidth) || settings.shoulderWidth < 0.30F ||
      settings.shoulderWidth > 1.10F || !std::isfinite(settings.limbProportion) ||
      settings.limbProportion < 0.65F || settings.limbProportion > 1.35F) {
    throw AppError(ErrorCode::E_MESH_GENERATION,
                   "character proportions are outside supported ranges");
  }

  SkinnedMesh mesh;
  mesh.settings = settings;
  mesh.bones = make_bones(settings);
  const auto zeroRotations = std::array<glm::vec3, kBoneCount>{};
  const auto restGlobal = compute_global_matrices(mesh.bones, zeroRotations);
  for (int index = 0; index < kBoneCount; ++index) {
    mesh.inverseBindMatrices[static_cast<std::size_t>(index)] =
        glm::inverse(restGlobal[static_cast<std::size_t>(index)]);
  }

  const float h = settings.height;
  const float limb = settings.limbProportion;
  const glm::vec3 pelvis = bone_position(restGlobal, Pelvis);
  const glm::vec3 chest = bone_position(restGlobal, Chest);
  const glm::vec3 neck = bone_position(restGlobal, Neck);
  const glm::vec3 head = bone_position(restGlobal, Head);
  const glm::vec3 upperArmL = bone_position(restGlobal, UpperArmL);
  const glm::vec3 lowerArmL = bone_position(restGlobal, LowerArmL);
  const glm::vec3 handL = bone_position(restGlobal, HandL);
  const glm::vec3 upperArmR = bone_position(restGlobal, UpperArmR);
  const glm::vec3 lowerArmR = bone_position(restGlobal, LowerArmR);
  const glm::vec3 handR = bone_position(restGlobal, HandR);
  const glm::vec3 thighL = bone_position(restGlobal, ThighL);
  const glm::vec3 calfL = bone_position(restGlobal, CalfL);
  const glm::vec3 footL = bone_position(restGlobal, FootL);
  const glm::vec3 thighR = bone_position(restGlobal, ThighR);
  const glm::vec3 calfR = bone_position(restGlobal, CalfR);
  const glm::vec3 footR = bone_position(restGlobal, FootR);

  const std::array<CapsuleSpec, 15> capsules = {{
      {pelvis + glm::vec3(0.0F, -0.055F * h, 0.0F), chest, 0.115F * h, Spine, Pelvis},
      {neck, head + glm::vec3(0.0F, 0.075F * h, 0.0F), 0.092F * h, Head, Neck},
      {chest + glm::vec3(0.0F, 0.03F * h, 0.0F), neck, 0.048F * h, Neck, Chest},
      {upperArmL, lowerArmL, 0.040F * h, LowerArmL, UpperArmL},
      {lowerArmL, handL, 0.034F * h, HandL, LowerArmL},
      {handL - (handL - lowerArmL) * 0.22F, handL + (handL - lowerArmL) * 0.08F, 0.042F * h,
       HandL, LowerArmL},
      {upperArmR, lowerArmR, 0.040F * h, LowerArmR, UpperArmR},
      {lowerArmR, handR, 0.034F * h, HandR, LowerArmR},
      {handR - (handR - lowerArmR) * 0.22F, handR + (handR - lowerArmR) * 0.08F, 0.042F * h,
       HandR, LowerArmR},
      {thighL, calfL, 0.062F * h, CalfL, ThighL},
      {calfL, footL, 0.047F * h, FootL, CalfL},
      {footL, footL + glm::vec3(0.0F, -0.02F * h, 0.09F * h * limb), 0.042F * h, FootL,
       CalfL},
      {thighR, calfR, 0.062F * h, CalfR, ThighR},
      {calfR, footR, 0.047F * h, FootR, CalfR},
      {footR, footR + glm::vec3(0.0F, -0.02F * h, 0.09F * h * limb), 0.042F * h, FootR,
       CalfR},
  }};

  for (const CapsuleSpec& capsule : capsules) {
    append_capsule(capsule, mesh);
  }
  if (mesh.vertices.empty() || mesh.indices.empty()) {
    throw AppError(ErrorCode::E_MESH_GENERATION, "generated character mesh is empty");
  }
  for (const SkinnedVertex& vertex : mesh.vertices) {
    const float sum = vertex.weights.x + vertex.weights.y + vertex.weights.z +
                      vertex.weights.w;
    if (!std::isfinite(sum) || std::abs(sum - 1.0F) > 1.0e-4F) {
      throw AppError(ErrorCode::E_MESH_GENERATION,
                     "generated skin weights are not normalized");
    }
  }
  return mesh;
}

std::array<glm::mat4, kBoneCount> compute_skin_matrices(const SkinnedMesh& mesh,
                                                        float animationTime) {
  if (!std::isfinite(animationTime) || animationTime < 0.0F) {
    throw AppError(ErrorCode::E_MESH_GENERATION, "animation time must be finite and non-negative");
  }
  const float phase =
      0.5F - 0.5F * std::cos(animationTime * glm::two_pi<float>() * 0.85F);
  std::array<glm::vec3, kBoneCount> rotations{};
  for (std::size_t index = 0; index < rotations.size(); ++index) {
    rotations[index] =
        glm::mix(kPoseKeyframes[0].localEulerRadians[index],
                 kPoseKeyframes[1].localEulerRadians[index], phase);
  }
  const std::array<glm::mat4, kBoneCount> poseGlobal =
      compute_global_matrices(mesh.bones, rotations);
  std::array<glm::mat4, kBoneCount> skinMatrices{};
  for (int index = 0; index < kBoneCount; ++index) {
    skinMatrices[static_cast<std::size_t>(index)] =
        poseGlobal[static_cast<std::size_t>(index)] *
        mesh.inverseBindMatrices[static_cast<std::size_t>(index)];
  }
  return skinMatrices;
}

}  // namespace lf

