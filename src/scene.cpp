#include "scene.hpp"

#include "error.hpp"

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace demo {

namespace {

constexpr std::size_t kRootJoint = 0;
constexpr std::size_t kTorsoJoint = 1;
constexpr std::size_t kHeadJoint = 2;
constexpr std::size_t kLeftUpperArmJoint = 3;
constexpr std::size_t kLeftLowerArmJoint = 4;
constexpr std::size_t kRightUpperArmJoint = 5;
constexpr std::size_t kRightLowerArmJoint = 6;

glm::quat eulerDegrees(float x, float y, float z) {
    return glm::normalize(glm::quat(glm::radians(glm::vec3(x, y, z))));
}

template <typename Edit>
Keyframe makeKeyframe(float time, Edit edit) {
    Pose pose = restPose();
    edit(pose);
    return Keyframe{time, pose};
}

std::array<AnimationClip, kClipCount> createAnimationClips() {
    std::array<AnimationClip, kClipCount> clips{};

    clips[static_cast<std::size_t>(ClipId::Idle)] = AnimationClip{
        "Idle",
        3.0f,
        {
            makeKeyframe(0.0f, [](Pose &) {}),
            makeKeyframe(1.0f, [](Pose &pose) {
                pose.joints[kRootJoint].translation.y = 0.012f;
                pose.joints[kTorsoJoint].rotation = eulerDegrees(1.2f, 0.0f, 0.0f);
                pose.joints[kHeadJoint].rotation = eulerDegrees(-0.8f, 2.5f, 0.0f);
                pose.joints[kLeftUpperArmJoint].rotation = eulerDegrees(2.5f, 0.0f, -1.5f);
                pose.joints[kRightUpperArmJoint].rotation = eulerDegrees(-2.5f, 0.0f, 1.5f);
                pose.joints[kLeftLowerArmJoint].rotation = eulerDegrees(2.0f, 0.0f, 0.0f);
                pose.joints[kRightLowerArmJoint].rotation = eulerDegrees(-2.0f, 0.0f, 0.0f);
            }),
            makeKeyframe(2.0f, [](Pose &pose) {
                pose.joints[kRootJoint].translation.y = 0.018f;
                pose.joints[kTorsoJoint].rotation = eulerDegrees(-1.0f, 0.0f, 0.0f);
                pose.joints[kHeadJoint].rotation = eulerDegrees(0.8f, -2.5f, 0.0f);
                pose.joints[kLeftUpperArmJoint].rotation = eulerDegrees(-2.0f, 0.0f, -0.5f);
                pose.joints[kRightUpperArmJoint].rotation = eulerDegrees(2.0f, 0.0f, 0.5f);
                pose.joints[kLeftLowerArmJoint].rotation = eulerDegrees(-1.5f, 0.0f, 0.0f);
                pose.joints[kRightLowerArmJoint].rotation = eulerDegrees(1.5f, 0.0f, 0.0f);
            }),
            makeKeyframe(3.0f, [](Pose &) {}),
        }};

    clips[static_cast<std::size_t>(ClipId::Walk)] = AnimationClip{
        "Walk",
        1.2f,
        {
            makeKeyframe(0.0f, [](Pose &pose) {
                pose.joints[kRootJoint].translation = glm::vec3(-0.015f, 0.0f, 0.0f);
                pose.joints[kTorsoJoint].rotation = eulerDegrees(0.0f, -3.0f, 0.0f);
                pose.joints[kLeftUpperArmJoint].rotation = eulerDegrees(28.0f, 0.0f, 0.0f);
                pose.joints[kRightUpperArmJoint].rotation = eulerDegrees(-28.0f, 0.0f, 0.0f);
                pose.joints[kLeftLowerArmJoint].rotation = eulerDegrees(12.0f, 0.0f, 0.0f);
                pose.joints[kRightLowerArmJoint].rotation = eulerDegrees(-12.0f, 0.0f, 0.0f);
                pose.joints[kHeadJoint].rotation = eulerDegrees(0.0f, 3.0f, 0.0f);
            }),
            makeKeyframe(0.3f, [](Pose &pose) {
                pose.joints[kRootJoint].translation = glm::vec3(0.0f, 0.045f, 0.0f);
                pose.joints[kTorsoJoint].rotation = eulerDegrees(2.0f, 0.0f, 0.0f);
                pose.joints[kHeadJoint].rotation = eulerDegrees(-1.0f, 0.0f, 0.0f);
            }),
            makeKeyframe(0.6f, [](Pose &pose) {
                pose.joints[kRootJoint].translation = glm::vec3(0.015f, 0.0f, 0.0f);
                pose.joints[kTorsoJoint].rotation = eulerDegrees(0.0f, 3.0f, 0.0f);
                pose.joints[kLeftUpperArmJoint].rotation = eulerDegrees(-28.0f, 0.0f, 0.0f);
                pose.joints[kRightUpperArmJoint].rotation = eulerDegrees(28.0f, 0.0f, 0.0f);
                pose.joints[kLeftLowerArmJoint].rotation = eulerDegrees(-12.0f, 0.0f, 0.0f);
                pose.joints[kRightLowerArmJoint].rotation = eulerDegrees(12.0f, 0.0f, 0.0f);
                pose.joints[kHeadJoint].rotation = eulerDegrees(0.0f, -3.0f, 0.0f);
            }),
            makeKeyframe(0.9f, [](Pose &pose) {
                pose.joints[kRootJoint].translation = glm::vec3(0.0f, 0.045f, 0.0f);
                pose.joints[kTorsoJoint].rotation = eulerDegrees(-2.0f, 0.0f, 0.0f);
                pose.joints[kHeadJoint].rotation = eulerDegrees(1.0f, 0.0f, 0.0f);
            }),
            makeKeyframe(1.2f, [](Pose &pose) {
                pose.joints[kRootJoint].translation = glm::vec3(-0.015f, 0.0f, 0.0f);
                pose.joints[kTorsoJoint].rotation = eulerDegrees(0.0f, -3.0f, 0.0f);
                pose.joints[kLeftUpperArmJoint].rotation = eulerDegrees(28.0f, 0.0f, 0.0f);
                pose.joints[kRightUpperArmJoint].rotation = eulerDegrees(-28.0f, 0.0f, 0.0f);
                pose.joints[kLeftLowerArmJoint].rotation = eulerDegrees(12.0f, 0.0f, 0.0f);
                pose.joints[kRightLowerArmJoint].rotation = eulerDegrees(-12.0f, 0.0f, 0.0f);
                pose.joints[kHeadJoint].rotation = eulerDegrees(0.0f, 3.0f, 0.0f);
            }),
        }};

    clips[static_cast<std::size_t>(ClipId::Wave)] = AnimationClip{
        "Wave",
        2.4f,
        {
            makeKeyframe(0.0f, [](Pose &) {}),
            makeKeyframe(0.35f, [](Pose &pose) {
                pose.joints[kRootJoint].translation.y = 0.018f;
                pose.joints[kRightUpperArmJoint].rotation = eulerDegrees(-12.0f, 0.0f, 112.0f);
                pose.joints[kRightLowerArmJoint].rotation = eulerDegrees(0.0f, 0.0f, -24.0f);
                pose.joints[kHeadJoint].rotation = eulerDegrees(0.0f, -5.0f, 5.0f);
                pose.joints[kLeftUpperArmJoint].rotation = eulerDegrees(-4.0f, 0.0f, -2.0f);
            }),
            makeKeyframe(0.70f, [](Pose &pose) {
                pose.joints[kRootJoint].translation.y = 0.020f;
                pose.joints[kRightUpperArmJoint].rotation = eulerDegrees(-12.0f, 0.0f, 112.0f);
                pose.joints[kRightLowerArmJoint].rotation = eulerDegrees(0.0f, 0.0f, 28.0f);
                pose.joints[kHeadJoint].rotation = eulerDegrees(0.0f, -7.0f, 7.0f);
                pose.joints[kLeftUpperArmJoint].rotation = eulerDegrees(-4.0f, 0.0f, -2.0f);
            }),
            makeKeyframe(1.05f, [](Pose &pose) {
                pose.joints[kRootJoint].translation.y = 0.016f;
                pose.joints[kRightUpperArmJoint].rotation = eulerDegrees(-10.0f, 0.0f, 108.0f);
                pose.joints[kRightLowerArmJoint].rotation = eulerDegrees(0.0f, 0.0f, -25.0f);
                pose.joints[kHeadJoint].rotation = eulerDegrees(0.0f, -4.0f, 5.0f);
                pose.joints[kLeftUpperArmJoint].rotation = eulerDegrees(-4.0f, 0.0f, -2.0f);
            }),
            makeKeyframe(1.40f, [](Pose &pose) {
                pose.joints[kRootJoint].translation.y = 0.020f;
                pose.joints[kRightUpperArmJoint].rotation = eulerDegrees(-12.0f, 0.0f, 112.0f);
                pose.joints[kRightLowerArmJoint].rotation = eulerDegrees(0.0f, 0.0f, 28.0f);
                pose.joints[kHeadJoint].rotation = eulerDegrees(0.0f, -6.0f, 7.0f);
                pose.joints[kLeftUpperArmJoint].rotation = eulerDegrees(-4.0f, 0.0f, -2.0f);
            }),
            makeKeyframe(1.75f, [](Pose &pose) {
                pose.joints[kRootJoint].translation.y = 0.012f;
                pose.joints[kRightUpperArmJoint].rotation = eulerDegrees(-8.0f, 0.0f, 70.0f);
                pose.joints[kRightLowerArmJoint].rotation = eulerDegrees(0.0f, 0.0f, -15.0f);
                pose.joints[kHeadJoint].rotation = eulerDegrees(0.0f, -2.0f, 2.0f);
            }),
            makeKeyframe(2.10f, [](Pose &pose) {
                pose.joints[kRootJoint].translation.y = 0.006f;
                pose.joints[kRightUpperArmJoint].rotation = eulerDegrees(-3.0f, 0.0f, 24.0f);
            }),
            makeKeyframe(2.40f, [](Pose &) {}),
        }};

    return clips;
}

glm::vec3 faceCenter(const glm::vec3 &center, const glm::vec3 &half,
                     const glm::vec3 &normal) {
    return center + glm::vec3(normal.x * half.x, normal.y * half.y, normal.z * half.z);
}

void addQuad(MeshData &mesh, std::uint32_t jointIndex,
             const std::array<glm::vec3, 4> &positions, const glm::vec3 &normal) {
    if (mesh.vertices.size() > std::numeric_limits<std::uint32_t>::max() - 4) {
        throw AppError(ErrorCode::SceneResourceInvalid, "Vertex count exceeds uint32 index range");
    }

    const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
    const std::array<glm::vec2, 4> uv = {
        glm::vec2(0.0f, 1.0f),
        glm::vec2(1.0f, 1.0f),
        glm::vec2(1.0f, 0.0f),
        glm::vec2(0.0f, 0.0f),
    };

    for (std::size_t i = 0; i < 4; ++i) {
        mesh.vertices.push_back(Vertex{positions[i], normal, uv[i],
                                       static_cast<float>(jointIndex)});
    }
    mesh.indices.insert(mesh.indices.end(),
                        {base, base + 1, base + 2, base, base + 2, base + 3});
}

void addBox(MeshData &mesh, std::uint32_t jointIndex, const glm::vec3 &center,
            const glm::vec3 &half) {
    const glm::vec3 nx = faceCenter(center, half, glm::vec3(1.0f, 0.0f, 0.0f));
    const glm::vec3 px = faceCenter(center, half, glm::vec3(-1.0f, 0.0f, 0.0f));
    const glm::vec3 ny = faceCenter(center, half, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::vec3 py = faceCenter(center, half, glm::vec3(0.0f, -1.0f, 0.0f));
    const glm::vec3 nz = faceCenter(center, half, glm::vec3(0.0f, 0.0f, 1.0f));
    const glm::vec3 pz = faceCenter(center, half, glm::vec3(0.0f, 0.0f, -1.0f));

    addQuad(mesh, jointIndex,
            {px + glm::vec3(0.0f, -half.y, -half.z),
             px + glm::vec3(0.0f, -half.y, half.z),
             px + glm::vec3(0.0f, half.y, half.z),
             px + glm::vec3(0.0f, half.y, -half.z)},
            glm::vec3(1.0f, 0.0f, 0.0f));
    addQuad(mesh, jointIndex,
            {nx + glm::vec3(0.0f, -half.y, half.z),
             nx + glm::vec3(0.0f, -half.y, -half.z),
             nx + glm::vec3(0.0f, half.y, -half.z),
             nx + glm::vec3(0.0f, half.y, half.z)},
            glm::vec3(-1.0f, 0.0f, 0.0f));
    addQuad(mesh, jointIndex,
            {py + glm::vec3(-half.x, 0.0f, -half.z),
             py + glm::vec3(half.x, 0.0f, -half.z),
             py + glm::vec3(half.x, 0.0f, half.z),
             py + glm::vec3(-half.x, 0.0f, half.z)},
            glm::vec3(0.0f, 1.0f, 0.0f));
    addQuad(mesh, jointIndex,
            {ny + glm::vec3(-half.x, 0.0f, half.z),
             ny + glm::vec3(half.x, 0.0f, half.z),
             ny + glm::vec3(half.x, 0.0f, -half.z),
             ny + glm::vec3(-half.x, 0.0f, -half.z)},
            glm::vec3(0.0f, -1.0f, 0.0f));
    addQuad(mesh, jointIndex,
            {pz + glm::vec3(-half.x, -half.y, 0.0f),
             pz + glm::vec3(half.x, -half.y, 0.0f),
             pz + glm::vec3(half.x, half.y, 0.0f),
             pz + glm::vec3(-half.x, half.y, 0.0f)},
            glm::vec3(0.0f, 0.0f, 1.0f));
    addQuad(mesh, jointIndex,
            {nz + glm::vec3(half.x, -half.y, 0.0f),
             nz + glm::vec3(-half.x, -half.y, 0.0f),
             nz + glm::vec3(-half.x, half.y, 0.0f),
             nz + glm::vec3(half.x, half.y, 0.0f)},
            glm::vec3(0.0f, 0.0f, -1.0f));
}

void addSphere(MeshData &mesh, std::uint32_t jointIndex, const glm::vec3 &center,
               float radius, int segments, int rings) {
    if (segments < 3 || rings < 2) {
        throw AppError(ErrorCode::SceneResourceInvalid, "Sphere tessellation is too low");
    }

    const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
    for (int ring = 0; ring <= rings; ++ring) {
        const float v = static_cast<float>(ring) / static_cast<float>(rings);
        const float latitude = -glm::half_pi<float>() + glm::pi<float>() * v;
        const float latitudeRadius = std::cos(latitude);
        const float y = std::sin(latitude);

        for (int segment = 0; segment <= segments; ++segment) {
            const float u = static_cast<float>(segment) / static_cast<float>(segments);
            const float longitude = glm::two_pi<float>() * u;
            const glm::vec3 normal(latitudeRadius * std::cos(longitude), y,
                                   latitudeRadius * std::sin(longitude));
            mesh.vertices.push_back(
                Vertex{center + normal * radius, normal, glm::vec2(u, v),
                       static_cast<float>(jointIndex)});
        }
    }

    const std::uint32_t stride = static_cast<std::uint32_t>(segments + 1);
    for (int ring = 0; ring < rings; ++ring) {
        for (int segment = 0; segment < segments; ++segment) {
            const std::uint32_t a = base + static_cast<std::uint32_t>(ring) * stride +
                                    static_cast<std::uint32_t>(segment);
            const std::uint32_t b = a + stride;
            mesh.indices.insert(mesh.indices.end(), {a, b, a + 1, b, b + 1, a + 1});
        }
    }
}

void addSection(MeshData &mesh, std::size_t jointIndex, std::size_t firstIndex) {
    mesh.sections.push_back(
        MeshSection{jointIndex, firstIndex, mesh.indices.size() - firstIndex});
}

bool finiteVec3(const glm::vec3 &value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

} // namespace

const std::array<JointDefinition, kJointCount> kJointDefinitions = {{
    {"Root", -1},
    {"Torso", static_cast<int>(kRootJoint)},
    {"Head", static_cast<int>(kTorsoJoint)},
    {"Arm.L.Upper", static_cast<int>(kTorsoJoint)},
    {"Arm.L.Lower", static_cast<int>(kLeftUpperArmJoint)},
    {"Arm.R.Upper", static_cast<int>(kTorsoJoint)},
    {"Arm.R.Lower", static_cast<int>(kRightUpperArmJoint)},
}};

const std::array<MaterialPreset, 3> kMaterialPresets = {{
    {"Ceramic Blue", glm::vec3(0.18f, 0.50f, 0.88f), 0.28f, 0.03f},
    {"Copper", glm::vec3(0.78f, 0.32f, 0.12f), 0.55f, 0.02f},
    {"Neon Green", glm::vec3(0.22f, 0.95f, 0.32f), 0.12f, 0.65f},
}};

const std::array<AnimationClip, kClipCount> &animationClips() {
    static const std::array<AnimationClip, kClipCount> clips = createAnimationClips();
    return clips;
}

Pose restPose() {
    Pose pose{};
    pose.joints[kRootJoint].translation = glm::vec3(0.0f, 0.0f, 0.0f);
    pose.joints[kTorsoJoint].translation = glm::vec3(0.0f, 0.18f, 0.0f);
    pose.joints[kHeadJoint].translation = glm::vec3(0.0f, 0.62f, 0.0f);
    pose.joints[kLeftUpperArmJoint].translation = glm::vec3(-0.31f, 0.54f, 0.0f);
    pose.joints[kLeftLowerArmJoint].translation = glm::vec3(0.0f, -0.33f, 0.0f);
    pose.joints[kRightUpperArmJoint].translation = glm::vec3(0.31f, 0.54f, 0.0f);
    pose.joints[kRightLowerArmJoint].translation = glm::vec3(0.0f, -0.33f, 0.0f);
    return pose;
}

Pose sampleClip(const AnimationClip &clip, double timeSeconds) {
    if (clip.keyframes.empty()) {
        throw AppError(ErrorCode::SceneResourceInvalid, "Animation clip has no keyframes");
    }

    const double duration = std::max(static_cast<double>(clip.duration), 1.0e-6);
    double wrapped = std::fmod(std::max(timeSeconds, 0.0), duration);
    if (wrapped < 0.0) {
        wrapped += duration;
    }

    const auto &keys = clip.keyframes;
    if (keys.size() == 1 || wrapped <= keys.front().time) {
        return keys.front().pose;
    }

    for (std::size_t i = 0; i + 1 < keys.size(); ++i) {
        if (wrapped <= keys[i + 1].time) {
            const double span = static_cast<double>(keys[i + 1].time - keys[i].time);
            const float alpha = span > 0.0
                                    ? static_cast<float>((wrapped - keys[i].time) / span)
                                    : 0.0f;
            Pose pose{};
            for (std::size_t joint = 0; joint < kJointCount; ++joint) {
                const JointLocal &a = keys[i].pose.joints[joint];
                const JointLocal &b = keys[i + 1].pose.joints[joint];
                pose.joints[joint].translation =
                    glm::mix(a.translation, b.translation, alpha);
                pose.joints[joint].rotation =
                    glm::normalize(glm::slerp(a.rotation, b.rotation, alpha));
            }
            return pose;
        }
    }
    return keys.back().pose;
}

std::array<glm::mat4, kJointCount> computeJointWorldMatrices(const Pose &pose) {
    std::array<glm::mat4, kJointCount> worlds{};
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        const JointLocal &local = pose.joints[joint];
        const glm::mat4 localMatrix =
            glm::translate(glm::mat4(1.0f), local.translation) *
            glm::mat4_cast(glm::normalize(local.rotation));
        const int parent = kJointDefinitions[joint].parent;
        worlds[joint] = parent >= 0
                            ? worlds[static_cast<std::size_t>(parent)] * localMatrix
                            : localMatrix;
    }
    return worlds;
}

const AnimationClip &AnimationState::currentClip() const {
    return animationClips()[static_cast<std::size_t>(clip_)];
}

void AnimationState::update(double deltaSeconds) {
    if (deltaSeconds <= 0.0) {
        return;
    }

    const double duration = std::max(static_cast<double>(currentClip().duration), 1.0e-6);
    timeSeconds_ = std::fmod(timeSeconds_ + deltaSeconds, duration);
}

void AnimationState::nextClip() {
    const std::size_t next =
        (static_cast<std::size_t>(clip_) + 1) % kClipCount;
    clip_ = static_cast<ClipId>(next);
    timeSeconds_ = 0.0;
}

void AnimationState::selectClip(ClipId clip) {
    const auto index = static_cast<std::size_t>(clip);
    if (index >= kClipCount) {
        throw AppError(ErrorCode::SceneResourceInvalid, "Invalid animation clip index");
    }
    clip_ = clip;
    timeSeconds_ = 0.0;
}

MeshData buildCharacterMesh(bool injectInvalidIndex) {
    MeshData mesh;
    mesh.vertices.reserve(4096);
    mesh.indices.reserve(6144);
    mesh.sections.reserve(kJointCount);

    std::size_t first = mesh.indices.size();
    addBox(mesh, static_cast<std::uint32_t>(kRootJoint), glm::vec3(0.0f, 0.0f, 0.0f),
           glm::vec3(0.24f, 0.13f, 0.14f));
    addSection(mesh, kRootJoint, first);

    first = mesh.indices.size();
    addBox(mesh, static_cast<std::uint32_t>(kTorsoJoint), glm::vec3(0.0f, 0.32f, 0.0f),
           glm::vec3(0.22f, 0.28f, 0.13f));
    addSection(mesh, kTorsoJoint, first);

    first = mesh.indices.size();
    addSphere(mesh, static_cast<std::uint32_t>(kHeadJoint), glm::vec3(0.0f, 0.15f, 0.0f),
              0.22f, 20, 12);
    addSection(mesh, kHeadJoint, first);

    first = mesh.indices.size();
    addBox(mesh, static_cast<std::uint32_t>(kLeftUpperArmJoint),
           glm::vec3(0.0f, -0.16f, 0.0f), glm::vec3(0.075f, 0.18f, 0.075f));
    addSection(mesh, kLeftUpperArmJoint, first);

    first = mesh.indices.size();
    addBox(mesh, static_cast<std::uint32_t>(kLeftLowerArmJoint),
           glm::vec3(0.0f, -0.15f, 0.0f), glm::vec3(0.065f, 0.16f, 0.065f));
    addBox(mesh, static_cast<std::uint32_t>(kLeftLowerArmJoint),
           glm::vec3(0.0f, -0.34f, 0.0f), glm::vec3(0.085f, 0.075f, 0.07f));
    addSection(mesh, kLeftLowerArmJoint, first);

    first = mesh.indices.size();
    addBox(mesh, static_cast<std::uint32_t>(kRightUpperArmJoint),
           glm::vec3(0.0f, -0.16f, 0.0f), glm::vec3(0.075f, 0.18f, 0.075f));
    addSection(mesh, kRightUpperArmJoint, first);

    first = mesh.indices.size();
    addBox(mesh, static_cast<std::uint32_t>(kRightLowerArmJoint),
           glm::vec3(0.0f, -0.15f, 0.0f), glm::vec3(0.065f, 0.16f, 0.065f));
    addBox(mesh, static_cast<std::uint32_t>(kRightLowerArmJoint),
           glm::vec3(0.0f, -0.34f, 0.0f), glm::vec3(0.085f, 0.075f, 0.07f));
    addSection(mesh, kRightLowerArmJoint, first);

    if (injectInvalidIndex) {
        if (mesh.indices.empty()) {
            throw AppError(ErrorCode::SceneResourceInvalid,
                           "Cannot inject a resource failure into an empty index buffer");
        }
        mesh.indices[0] = static_cast<std::uint32_t>(mesh.vertices.size() + 17);
    }

    validateMesh(mesh);
    return mesh;
}

void validateMesh(const MeshData &mesh) {
    if (mesh.vertices.empty() || mesh.indices.empty()) {
        throw AppError(ErrorCode::SceneResourceInvalid,
                       "Procedural mesh resource is empty");
    }
    if (mesh.indices.size() % 3 != 0) {
        throw AppError(ErrorCode::SceneResourceInvalid,
                       "Procedural mesh index count is not triangle-aligned");
    }
    if (mesh.sections.empty()) {
        throw AppError(ErrorCode::SceneResourceInvalid,
                       "Procedural mesh has no joint mesh sections");
    }

    std::array<bool, kJointCount> covered{};
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        const Vertex &vertex = mesh.vertices[i];
        if (!finiteVec3(vertex.position) || !finiteVec3(vertex.normal) ||
            !std::isfinite(vertex.texCoord.x) || !std::isfinite(vertex.texCoord.y) ||
            !std::isfinite(vertex.jointIndex)) {
            throw AppError(ErrorCode::SceneResourceInvalid,
                           "Procedural mesh contains a non-finite vertex");
        }
        if (glm::length(vertex.normal) < 1.0e-5f) {
            throw AppError(ErrorCode::SceneResourceInvalid,
                           "Procedural mesh contains a zero-length normal");
        }
        if (vertex.jointIndex < 0.0f ||
            vertex.jointIndex > static_cast<float>(kJointCount - 1) ||
            std::abs(vertex.jointIndex - std::round(vertex.jointIndex)) > 1.0e-4f) {
            throw AppError(ErrorCode::SceneResourceInvalid,
                           "Procedural mesh contains an invalid joint index");
        }
        covered[static_cast<std::size_t>(std::lround(vertex.jointIndex))] = true;
    }

    for (std::size_t sectionIndex = 0; sectionIndex < mesh.sections.size();
         ++sectionIndex) {
        const MeshSection &section = mesh.sections[sectionIndex];
        if (section.jointIndex >= kJointCount || section.indexCount == 0 ||
            section.indexCount % 3 != 0 ||
            section.indexOffset + section.indexCount > mesh.indices.size()) {
            throw AppError(ErrorCode::SceneResourceInvalid,
                           "Procedural mesh section bounds are invalid");
        }
        for (std::size_t i = section.indexOffset;
             i < section.indexOffset + section.indexCount; ++i) {
            if (mesh.indices[i] >= mesh.vertices.size()) {
                std::ostringstream message;
                message << "Index " << mesh.indices[i] << " at position " << i
                        << " exceeds vertex count " << mesh.vertices.size();
                throw AppError(ErrorCode::SceneResourceInvalid, message.str());
            }
        }
    }

    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        if (!covered[joint]) {
            throw AppError(ErrorCode::SceneResourceInvalid,
                           "One or more joints have no associated mesh geometry");
        }
    }
}

} // namespace demo
