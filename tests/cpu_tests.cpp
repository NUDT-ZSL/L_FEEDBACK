#include "error.hpp"
#include "frame_data.hpp"
#include "scene.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

int gChecks = 0;

void check(bool condition, const std::string &message) {
    ++gChecks;
    if (!condition) {
        std::fprintf(stderr, "[FAIL] %s\n", message.c_str());
        std::exit(1);
    }
}

void checkNear(float actual, float expected, float epsilon,
               const std::string &message) {
    check(std::abs(actual - expected) <= epsilon, message);
}

bool finiteMatrix(const glm::mat4 &matrix) {
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            if (!std::isfinite(matrix[column][row])) {
                return false;
            }
        }
    }
    return true;
}

void testMaterials() {
    check(demo::kMaterialPresets.size() == 3, "three material presets exist");
    check(demo::kMaterialPresets[0].name == "Ceramic Blue",
          "preset 1 is Ceramic Blue");
    check(demo::kMaterialPresets[1].name == "Copper", "preset 2 is Copper");
    check(demo::kMaterialPresets[2].name == "Neon Green",
          "preset 3 is Neon Green");
    for (const demo::MaterialPreset &material : demo::kMaterialPresets) {
        check(material.roughness >= 0.0f && material.roughness <= 1.0f,
              "material roughness is in [0, 1]");
        check(material.emissiveStrength >= 0.0f,
              "material emissive strength is non-negative");
    }
    check(demo::kMaterialPresets[2].emissiveStrength >
              demo::kMaterialPresets[0].emissiveStrength,
          "emissive material differs measurably from ceramic");
}

void testSkeletonAndAnimation() {
    check(demo::kJointDefinitions.size() == 7, "seven joints are defined");
    check(demo::kJointDefinitions[0].name == "Root" &&
              demo::kJointDefinitions[0].parent == -1,
          "root is the hierarchy root");
    check(demo::kJointDefinitions[1].name == "Torso" &&
              demo::kJointDefinitions[1].parent == 0,
          "torso is parented to root");
    check(demo::kJointDefinitions[2].name == "Head" &&
              demo::kJointDefinitions[2].parent == 1,
          "head is parented to torso");
    for (std::size_t joint = 0; joint < demo::kJointDefinitions.size(); ++joint) {
        const int parent = demo::kJointDefinitions[joint].parent;
        check(parent < static_cast<int>(joint),
              "joint parents precede children");
    }

    const auto &clips = demo::animationClips();
    check(clips.size() == 3, "three animation clips exist");
    check(clips[0].name == "Idle", "clip 0 is Idle");
    check(clips[1].name == "Walk", "clip 1 is Walk");
    check(clips[2].name == "Wave", "clip 2 is Wave");

    for (const demo::AnimationClip &clip : clips) {
        check(clip.duration > 0.0f, "clip duration is positive");
        check(clip.keyframes.size() >= 2, "clip has at least two keyframes");
        checkNear(clip.keyframes.front().time, 0.0f, 1.0e-6f,
                  "clip starts at zero");
        checkNear(clip.keyframes.back().time, clip.duration, 1.0e-6f,
                  "clip ends at duration");
        for (std::size_t i = 1; i < clip.keyframes.size(); ++i) {
            check(clip.keyframes[i - 1].time < clip.keyframes[i].time,
                  "keyframe times are strictly increasing");
        }
    }

    const demo::Pose idleAtOne = demo::sampleClip(clips[0], 1.0);
    checkNear(idleAtOne.joints[0].translation.y, 0.012f, 1.0e-5f,
              "idle keyframe interpolates root translation");

    const demo::Pose walkAtPeak = demo::sampleClip(clips[1], 0.3);
    checkNear(walkAtPeak.joints[0].translation.y, 0.045f, 1.0e-5f,
              "walk keyframe carries root vertical position");

    const demo::Pose wave = demo::sampleClip(clips[2], 0.35);
    check(std::abs(glm::dot(wave.joints[5].rotation,
                            glm::quat(1.0f, 0.0f, 0.0f, 0.0f))) < 0.999f,
          "wave keyframe rotates the right upper arm");

    const demo::Pose rest = demo::restPose();
    const auto restWorld = demo::computeJointWorldMatrices(rest);
    checkNear(restWorld[6][3].x, 0.31f, 1.0e-5f,
              "lower arm inherits upper-arm X translation");
    checkNear(restWorld[6][3].y, 0.39f, 1.0e-5f,
              "lower arm inherits hierarchical Y translation");
    for (const glm::mat4 &matrix : restWorld) {
        check(finiteMatrix(matrix), "rest-pose world matrices are finite");
    }

    demo::AnimationState state;
    check(state.clip() == demo::ClipId::Idle, "animation starts in Idle");
    state.nextClip();
    check(state.clip() == demo::ClipId::Walk, "Idle advances to Walk");
    state.nextClip();
    check(state.clip() == demo::ClipId::Wave, "Walk advances to Wave");
    state.nextClip();
    check(state.clip() == demo::ClipId::Idle, "Wave wraps to Idle");
    state.update(1.0);
    checkNear(static_cast<float>(state.timeSeconds()), 1.0f, 1.0e-6f,
              "animation time advances in seconds");
    state.update(2.5);
    checkNear(static_cast<float>(state.timeSeconds()), 0.5f, 1.0e-5f,
              "idle timeline wraps by clip duration");
}

void testMesh() {
    const demo::MeshData mesh = demo::buildCharacterMesh();
    check(!mesh.vertices.empty(), "procedural mesh has vertices");
    check(!mesh.indices.empty(), "procedural mesh has indices");
    check(mesh.indices.size() % 3 == 0, "index count is triangle-aligned");
    check(mesh.sections.size() == demo::kJointCount,
          "every joint has a mesh section");

    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        const demo::Vertex &vertex = mesh.vertices[i];
        check(std::isfinite(vertex.position.x) &&
                  std::isfinite(vertex.position.y) &&
                  std::isfinite(vertex.position.z),
              "vertex positions are finite");
        check(std::isfinite(vertex.normal.x) &&
                  std::isfinite(vertex.normal.y) &&
                  std::isfinite(vertex.normal.z),
              "vertex normals are finite");
        check(std::isfinite(vertex.texCoord.x) &&
                  std::isfinite(vertex.texCoord.y),
              "texture coordinates are finite");
        check(vertex.jointIndex >= 0.0f &&
                  vertex.jointIndex < static_cast<float>(demo::kJointCount),
              "vertex joint index is in range");
    }
    for (std::uint32_t index : mesh.indices) {
        check(index < mesh.vertices.size(), "index references an existing vertex");
    }

    for (const demo::MeshSection &section : mesh.sections) {
        check(section.indexCount > 0, "mesh section contains triangles");
        check(section.indexOffset + section.indexCount <= mesh.indices.size(),
              "mesh section lies within the index buffer");
    }

    bool badResourceThrown = false;
    try {
        static_cast<void>(demo::buildCharacterMesh(true));
    } catch (const demo::AppError &error) {
        badResourceThrown =
            error.code() == demo::ErrorCode::SceneResourceInvalid;
    }
    check(badResourceThrown,
          "invalid procedural mesh fails with E3001 resource validation");
}

void testFrameData() {
    check(sizeof(demo::FrameUniforms) == 704, "std140 UBO is 704 bytes");
    check(offsetof(demo::FrameUniforms, viewProjection) == 0,
          "viewProjection offset is 0");
    check(offsetof(demo::FrameUniforms, model) == 64, "model offset is 64");
    check(offsetof(demo::FrameUniforms, diffuseColor) == 128,
          "diffuseColor offset is 128");
    check(offsetof(demo::FrameUniforms, materialParams) == 144,
          "materialParams offset is 144");
    check(offsetof(demo::FrameUniforms, lightDirection) == 160,
          "lightDirection offset is 160");
    check(offsetof(demo::FrameUniforms, viewPosition) == 176,
          "viewPosition offset is 176");
    check(offsetof(demo::FrameUniforms, jointWorld) == 192,
          "jointWorld offset is 192");

    std::array<glm::mat4, demo::kMaxJoints> joints{};
    joints.fill(glm::mat4(1.0f));
    const glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(1, 2, 3));
    const demo::FrameUniforms frame = demo::makeFrameUniforms(
        glm::mat4(1.0f), model, demo::kMaterialPresets[2].diffuseColor,
        demo::kMaterialPresets[2].roughness,
        demo::kMaterialPresets[2].emissiveStrength, joints,
        glm::normalize(glm::vec3(1, 1, 0)), glm::vec3(4, 5, 6), 1.25);

    checkNear(frame.diffuseColor.r, demo::kMaterialPresets[2].diffuseColor.r,
              1.0e-6f, "frame diffuse color follows the selected preset");
    checkNear(frame.materialParams.x, demo::kMaterialPresets[2].roughness,
              1.0e-6f, "frame roughness follows the selected preset");
    checkNear(frame.materialParams.y,
              demo::kMaterialPresets[2].emissiveStrength, 1.0e-6f,
              "frame emissive follows the selected preset");
    checkNear(frame.materialParams.z, 1.25f, 1.0e-6f,
              "frame packet carries animation time");
    checkNear(frame.viewPosition.w, 1.0f, 1.0e-6f,
              "view position w is std140-safe");
    check(std::memcmp(&frame.jointWorld[0], &joints[0], sizeof(glm::mat4)) == 0,
          "joint matrices are copied into the frame packet");
}

} // namespace

int main() {
    testMaterials();
    testSkeletonAndAnimation();
    testMesh();
    testFrameData();
    std::printf("[CPU TEST] %d checks passed\n", gChecks);
    return 0;
}
