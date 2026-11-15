#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace demo {

constexpr std::size_t kJointCount = 7;

enum class ClipId : std::size_t {
    Idle = 0,
    Walk = 1,
    Wave = 2
};

constexpr std::size_t kClipCount = 3;

struct JointDefinition {
    std::string_view name;
    int parent;
};

struct MaterialPreset {
    std::string_view name;
    glm::vec3 diffuseColor;
    float roughness;
    float emissiveStrength;
};

extern const std::array<JointDefinition, kJointCount> kJointDefinitions;
extern const std::array<MaterialPreset, 3> kMaterialPresets;

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    float jointIndex;
};

static_assert(offsetof(Vertex, position) == 0);
static_assert(offsetof(Vertex, normal) == 12);
static_assert(offsetof(Vertex, texCoord) == 24);
static_assert(offsetof(Vertex, jointIndex) == 32);
static_assert(sizeof(Vertex) == 36);

struct MeshSection {
    std::size_t jointIndex;
    std::size_t indexOffset;
    std::size_t indexCount;
};

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<MeshSection> sections;
};

struct JointLocal {
    glm::vec3 translation{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
};

struct Pose {
    std::array<JointLocal, kJointCount> joints;
};

struct Keyframe {
    float time;
    Pose pose;
};

struct AnimationClip {
    std::string_view name;
    float duration;
    std::vector<Keyframe> keyframes;
};

const std::array<AnimationClip, kClipCount> &animationClips();
Pose restPose();
Pose sampleClip(const AnimationClip &clip, double timeSeconds);
std::array<glm::mat4, kJointCount> computeJointWorldMatrices(const Pose &pose);

class AnimationState {
public:
    AnimationState() = default;

    ClipId clip() const noexcept {
        return clip_;
    }

    double timeSeconds() const noexcept {
        return timeSeconds_;
    }

    const AnimationClip &currentClip() const;
    void update(double deltaSeconds);
    void nextClip();
    void selectClip(ClipId clip);

private:
    ClipId clip_ = ClipId::Idle;
    double timeSeconds_ = 0.0;
};

MeshData buildCharacterMesh(bool injectInvalidIndex = false);
void validateMesh(const MeshData &mesh);

} // namespace demo
