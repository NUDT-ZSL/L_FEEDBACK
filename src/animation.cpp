#include "animation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include <glm/gtc/matrix_inverse.hpp>

#include "error.h"

namespace sr {
namespace {

constexpr double kWeightEpsilon = 1.0e-6;
constexpr double kQuaternionEpsilon = 1.0e-12;

glm::dquat normalizedQuaternion(const glm::dquat& q) {
    const double length = glm::length(q);
    if (!std::isfinite(length) || length < kQuaternionEpsilon) {
        return glm::dquat(1.0, 0.0, 0.0, 0.0);
    }
    return q / length;
}

void validateSortedKeyframes(const std::vector<Keyframe>& keys) {
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (!std::isfinite(keys[i].time)) {
            fail(ExitCode::Scene, "E_SCENE_ANIMATION", "keyframe time is not finite");
        }
        if (i > 0 && !(keys[i - 1].time < keys[i].time)) {
            std::ostringstream message;
            message << "keyframe times must be strictly increasing at entries "
                    << (i - 1) << " and " << i;
            fail(ExitCode::Scene, "E_SCENE_ANIMATION", message.str());
        }
    }
}

} // namespace

void Bounds::expand(const glm::dvec3& p) {
    if (!valid) {
        min = p;
        max = p;
        valid = true;
    } else {
        min = glm::min(min, p);
        max = glm::max(max, p);
    }
}

void Bounds::expand(const Bounds& other) {
    if (!other.valid) {
        return;
    }
    expand(other.min);
    expand(other.max);
}

std::array<glm::dvec3, 8> Bounds::corners() const {
    std::array<glm::dvec3, 8> result{};
    result[0] = {min.x, min.y, min.z};
    result[1] = {max.x, min.y, min.z};
    result[2] = {min.x, max.y, min.z};
    result[3] = {max.x, max.y, min.z};
    result[4] = {min.x, min.y, max.z};
    result[5] = {max.x, min.y, max.z};
    result[6] = {min.x, max.y, max.z};
    result[7] = {max.x, max.y, max.z};
    return result;
}

glm::dvec3 Bounds::center() const {
    return (min + max) * 0.5;
}

glm::dvec3 Bounds::extent() const {
    return max - min;
}

Bounds meshBounds(const MeshData& mesh) {
    Bounds bounds;
    for (const SkinnedVertex& vertex : mesh.vertices) {
        bounds.expand(glm::dvec3(vertex.px, vertex.py, vertex.pz));
    }
    return bounds;
}

MeshParams meshParamsForVariant(int variant) {
    MeshParams params;
    params.variant = variant;
    if (variant == 1) {
        params.bodyWidth = 1.16;
        params.heightScale = 0.97;
        params.headScale = 1.08;
        params.armSpan = 1.14;
    } else if (variant == 2) {
        params.bodyWidth = 0.88;
        params.heightScale = 1.05;
        params.headScale = 0.93;
        params.armSpan = 0.90;
    } else {
        const double phase = static_cast<double>(variant);
        params.bodyWidth = 1.0 + 0.045 * std::sin(phase * 1.17);
        params.heightScale = 1.0 + 0.025 * std::cos(phase * 0.71);
        params.headScale = 1.0 + 0.04 * std::sin(phase * 0.43 + 0.6);
        params.armSpan = 1.0 + 0.05 * std::cos(phase * 0.93 + 0.2);
    }
    return params;
}

Camera evaluateCamera(double seconds) {
    Camera camera;
    const double orbit = 0.62 * std::sin(seconds * 0.73) +
                         0.19 * std::sin(seconds * 1.91 + 0.4) + 0.55;
    const double radius = 7.2 + 0.32 * std::sin(seconds * 0.47 + 1.1);
    const double height = 3.35 + 0.18 * std::cos(seconds * 0.63);
    camera.target = {0.05, 1.22, 0.0};
    camera.eye = camera.target +
                 glm::dvec3(std::cos(orbit) * radius, height, std::sin(orbit) * radius);
    camera.up = {0.0, 1.0, 0.0};
    camera.fovYDegrees = 43.0;
    camera.nearPlane = 0.45;
    camera.farPlane = 28.0;
    updateCameraMatrices(camera);
    return camera;
}

void updateCameraMatrices(Camera& camera) {
    camera.view = glm::lookAt(camera.eye, camera.target, camera.up);
    camera.projection = glm::perspective(glm::radians(camera.fovYDegrees),
                                         camera.aspect,
                                         camera.nearPlane,
                                         camera.farPlane);
    camera.viewProjection = camera.projection * camera.view;
}

Skeleton makeSkeleton() {
    Skeleton skeleton;
    skeleton.bones.resize(kSkeletonBoneCount);

    Bone& root = skeleton.bones[0];
    root.name = "hips";
    root.parent = -1;
    root.bindTranslation = {0.0, 0.95, 0.0};
    root.bindRotation = glm::dquat(1.0, 0.0, 0.0, 0.0);
    root.keyframes = {
        {0.00, {0.0, 0.950, 0.0}, glm::angleAxis(0.000, glm::dvec3(0, 1, 0))},
        {0.37, {0.015, 0.972, 0.0}, glm::angleAxis(0.105, glm::dvec3(0, 0, 1))},
        {1.20, {-0.018, 0.944, 0.012}, glm::angleAxis(-0.080, glm::dvec3(0.2, 0.3, 1))},
        {2.45, {0.023, 0.982, -0.010}, glm::angleAxis(0.135, glm::dvec3(-0.2, 0, 1))},
        {3.80, {0.0, 0.950, 0.0}, glm::angleAxis(0.420, glm::dvec3(0, 1, 0))},
    };

    Bone& mid = skeleton.bones[1];
    mid.name = "chest";
    mid.parent = 0;
    mid.bindTranslation = {0.0, 0.65, 0.0};
    mid.bindRotation = glm::dquat(1.0, 0.0, 0.0, 0.0);
    mid.keyframes = {
        {0.00, {0.0, 0.650, 0.0}, glm::angleAxis(0.000, glm::dvec3(1, 0, 0))},
        {0.19, {-0.012, 0.658, 0.010}, glm::angleAxis(0.115, glm::dvec3(1, 0, 0))},
        {0.83, {0.020, 0.642, -0.006}, glm::angleAxis(-0.150, glm::dvec3(0.4, 0, 1))},
        {2.90, {-0.025, 0.665, 0.0}, glm::angleAxis(0.190, glm::dvec3(0, 0.2, 1))},
        {3.80, {0.0, 0.650, 0.0}, glm::angleAxis(0.0, glm::dvec3(1, 0, 0))},
    };

    Bone& tip = skeleton.bones[2];
    tip.name = "head";
    tip.parent = 1;
    tip.bindTranslation = {0.0, 0.55, 0.0};
    tip.bindRotation = glm::dquat(1.0, 0.0, 0.0, 0.0);
    tip.keyframes = {
        {0.00, {0.0, 0.550, 0.0}, glm::angleAxis(0.000, glm::dvec3(1, 0, 0))},
        {0.51, {0.0, 0.548, 0.0}, glm::angleAxis(-0.120, glm::dvec3(1, 0, 0))},
        {1.55, {0.010, 0.552, 0.0}, glm::angleAxis(0.100, glm::dvec3(0.2, 1, 0.3))},
        {3.10, {-0.008, 0.550, 0.006}, glm::angleAxis(-0.080, glm::dvec3(1, 0.2, 0))},
        {3.80, {0.0, 0.550, 0.0}, glm::angleAxis(0.0, glm::dvec3(1, 0, 0))},
    };

    for (Bone& bone : skeleton.bones) {
        validateSortedKeyframes(bone.keyframes);
        for (Keyframe& key : bone.keyframes) {
            key.rotation = normalizedQuaternion(key.rotation);
        }
    }

    std::vector<glm::dmat4> bindWorld(kSkeletonBoneCount, glm::dmat4(1.0));
    skeleton.inverseBind.resize(kSkeletonBoneCount, glm::dmat4(1.0));
    for (int i = 0; i < kSkeletonBoneCount; ++i) {
        const Bone& bone = skeleton.bones[static_cast<std::size_t>(i)];
        const glm::dmat4 local =
            glm::translate(glm::dmat4(1.0), bone.bindTranslation) *
            glm::mat4_cast(normalizedQuaternion(bone.bindRotation));
        bindWorld[static_cast<std::size_t>(i)] =
            bone.parent >= 0
                ? bindWorld[static_cast<std::size_t>(bone.parent)] * local
                : local;
        skeleton.inverseBind[static_cast<std::size_t>(i)] =
            glm::inverse(bindWorld[static_cast<std::size_t>(i)]);
    }
    return skeleton;
}

glm::dquat shortestSlerp(const glm::dquat& a, const glm::dquat& b, double t) {
    glm::dquat q0 = normalizedQuaternion(a);
    glm::dquat q1 = normalizedQuaternion(b);
    double dot = glm::dot(q0, q1);

    if (dot < 0.0) {
        q1 = -q1;
        dot = -dot;
    }
    dot = std::clamp(dot, -1.0, 1.0);

    if (dot > 0.9995) {
        const glm::dquat mixed = q0 + (q1 - q0) * t;
        return normalizedQuaternion(mixed);
    }

    const double theta0 = std::acos(dot);
    const double theta = theta0 * t;
    const double sinTheta = std::sin(theta);
    const double sinTheta0 = std::sin(theta0);
    const double w0 = std::cos(theta) - dot * sinTheta / sinTheta0;
    const double w1 = sinTheta / sinTheta0;
    return normalizedQuaternion(q0 * w0 + q1 * w1);
}

PoseSample sampleKeyframes(const std::vector<Keyframe>& keys, double time) {
    if (keys.empty()) {
        fail(ExitCode::Scene, "E_SCENE_ANIMATION", "a bone has no keyframes");
    }

    PoseSample sample;
    if (time <= keys.front().time) {
        sample.translation = keys.front().translation;
        sample.rotation = normalizedQuaternion(keys.front().rotation);
        sample.segmentTime0 = keys.front().time;
        sample.segmentTime1 = keys.front().time;
        sample.segmentIndex = 0;
        sample.clampedBefore = true;
        return sample;
    }
    if (time >= keys.back().time) {
        sample.translation = keys.back().translation;
        sample.rotation = normalizedQuaternion(keys.back().rotation);
        sample.segmentTime0 = keys.back().time;
        sample.segmentTime1 = keys.back().time;
        sample.segmentIndex = static_cast<int>(keys.size()) - 1;
        sample.clampedAfter = true;
        return sample;
    }

    const auto upper = std::upper_bound(
        keys.begin(), keys.end(), time,
        [](double value, const Keyframe& key) { return value < key.time; });
    const std::size_t index1 = static_cast<std::size_t>(upper - keys.begin());
    const std::size_t index0 = index1 - 1;
    const Keyframe& k0 = keys[index0];
    const Keyframe& k1 = keys[index1];
    const double span = k1.time - k0.time;
    const double fraction = span > 0.0 ? (time - k0.time) / span : 0.0;

    sample.translation = glm::mix(k0.translation, k1.translation, fraction);
    sample.rotation = shortestSlerp(k0.rotation, k1.rotation, fraction);
    sample.segmentTime0 = k0.time;
    sample.segmentTime1 = k1.time;
    sample.fraction = fraction;
    sample.segmentIndex = static_cast<int>(index0);
    return sample;
}

PoseResult evaluatePose(const Skeleton& skeleton, double time) {
    if (skeleton.bones.size() != static_cast<std::size_t>(kSkeletonBoneCount) ||
        skeleton.inverseBind.size() != skeleton.bones.size()) {
        fail(ExitCode::Scene, "E_SCENE_RIG", "skeleton must contain exactly three valid bones");
    }
    if (!std::isfinite(time)) {
        fail(ExitCode::Scene, "E_SCENE_ANIMATION", "pose time is not finite");
    }

    PoseResult pose;
    pose.time = time;
    pose.worldMatrices.resize(skeleton.bones.size(), glm::dmat4(1.0));
    pose.skinMatrices.resize(skeleton.bones.size(), glm::dmat4(1.0));
    pose.localRotations.resize(skeleton.bones.size(), glm::dquat(1.0, 0.0, 0.0, 0.0));
    pose.samples.resize(skeleton.bones.size());

    for (std::size_t i = 0; i < skeleton.bones.size(); ++i) {
        const Bone& bone = skeleton.bones[i];
        if (bone.parent >= static_cast<int>(i)) {
            fail(ExitCode::Scene, "E_SCENE_RIG", "bone parents must precede their children");
        }
        pose.samples[i] = sampleKeyframes(bone.keyframes, time);
        pose.localRotations[i] = pose.samples[i].rotation;
        const glm::dmat4 local =
            glm::translate(glm::dmat4(1.0), pose.samples[i].translation) *
            glm::mat4_cast(pose.samples[i].rotation);
        pose.worldMatrices[i] =
            bone.parent >= 0
                ? pose.worldMatrices[static_cast<std::size_t>(bone.parent)] * local
                : local;
        pose.skinMatrices[i] = pose.worldMatrices[i] * skeleton.inverseBind[i];
    }
    return pose;
}

void normalizeVertexWeights(SkinnedVertex& vertex) {
    double sum = 0.0;
    for (int i = 0; i < kMaxBoneInfluences; ++i) {
        double weight = static_cast<double>(vertex.boneWeights[i]);
        if (!std::isfinite(weight) || weight < 0.0) {
            weight = 0.0;
        }
        vertex.boneWeights[i] = static_cast<float>(weight);
        sum += weight;
    }

    if (sum <= kWeightEpsilon) {
        for (int i = 0; i < kMaxBoneInfluences; ++i) {
            vertex.boneIndices[i] = 0;
            vertex.boneWeights[i] = (i == 0) ? 1.0f : 0.0f;
        }
        return;
    }

    for (float& weight : vertex.boneWeights) {
        weight = static_cast<float>(static_cast<double>(weight) / sum);
    }
}

void validateVertexWeights(const SkinnedVertex& vertex, const char* context) {
    double sum = 0.0;
    for (int i = 0; i < kMaxBoneInfluences; ++i) {
        if (vertex.boneIndices[i] >= static_cast<std::uint32_t>(kSkeletonBoneCount)) {
            fail(ExitCode::Scene, "E_SCENE_WEIGHTS",
                 std::string(context) + ": bone index exceeds the three-bone chain");
        }
        const double weight = static_cast<double>(vertex.boneWeights[i]);
        if (!std::isfinite(weight) || weight < 0.0) {
            fail(ExitCode::Scene, "E_SCENE_WEIGHTS",
                 std::string(context) + ": weight is negative or not finite");
        }
        sum += weight;
    }
    if (!(sum > kWeightEpsilon)) {
        fail(ExitCode::Scene, "E_SCENE_WEIGHTS",
             std::string(context) + ": every vertex must have a non-zero weight sum");
    }
    if (std::abs(sum - 1.0) > 1.0e-4) {
        fail(ExitCode::Scene, "E_SCENE_WEIGHTS",
             std::string(context) + ": vertex weights were not normalized");
    }
}

glm::dvec3 skinPosition(const SkinnedVertex& vertex,
                        const std::vector<glm::dmat4>& skinMatrices) {
    double sum = 0.0;
    for (float weight : vertex.boneWeights) {
        if (std::isfinite(weight) && weight > 0.0f) {
            sum += static_cast<double>(weight);
        }
    }

    glm::dvec4 result(0.0);
    if (sum <= kWeightEpsilon) {
        result = skinMatrices[0] * glm::dvec4(vertex.px, vertex.py, vertex.pz, 1.0);
    } else {
        const glm::dvec4 position(vertex.px, vertex.py, vertex.pz, 1.0);
        for (int i = 0; i < kMaxBoneInfluences; ++i) {
            const double weight = static_cast<double>(vertex.boneWeights[i]);
            if (weight > 0.0) {
                result += skinMatrices[vertex.boneIndices[i]] * position *
                          (weight / sum);
            }
        }
    }
    return glm::dvec3(result) / result.w;
}

glm::dvec3 skinNormal(const SkinnedVertex& vertex,
                      const std::vector<glm::dmat4>& skinMatrices) {
    double sum = 0.0;
    for (float weight : vertex.boneWeights) {
        if (std::isfinite(weight) && weight > 0.0f) {
            sum += static_cast<double>(weight);
        }
    }

    const glm::dvec3 normal(vertex.nx, vertex.ny, vertex.nz);
    glm::dvec3 result(0.0);
    if (sum <= kWeightEpsilon) {
        result = glm::inverseTranspose(glm::mat3(skinMatrices[0])) * normal;
    } else {
        for (int i = 0; i < kMaxBoneInfluences; ++i) {
            const double weight = static_cast<double>(vertex.boneWeights[i]);
            if (weight > 0.0) {
                const glm::dmat3 normalMatrix =
                    glm::inverseTranspose(glm::mat3(skinMatrices[vertex.boneIndices[i]]));
                result += normalMatrix * normal * (weight / sum);
            }
        }
    }
    const double length = glm::length(result);
    if (!std::isfinite(length) || length < 1.0e-12) {
        return glm::dvec3(0.0, 1.0, 0.0);
    }
    return result / length;
}

MeshRevisionInfo inspectMeshWeights(const MeshData& mesh) {
    MeshRevisionInfo info;
    info.vertexCount = static_cast<int>(mesh.vertices.size());
    info.triangleCount = static_cast<int>(mesh.indices.size() / 3);
    info.minWeightSum = std::numeric_limits<double>::infinity();
    info.maxWeightSum = 0.0;

    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        const SkinnedVertex& vertex = mesh.vertices[i];
        validateVertexWeights(vertex, mesh.name.c_str());
        int influences = 0;
        double sum = 0.0;
        for (int j = 0; j < kMaxBoneInfluences; ++j) {
            const double weight = static_cast<double>(vertex.boneWeights[j]);
            sum += weight;
            if (weight > 1.0e-7) {
                ++influences;
            }
        }
        info.maxInfluences = std::max(info.maxInfluences, influences);
        info.minWeightSum = std::min(info.minWeightSum, sum);
        info.maxWeightSum = std::max(info.maxWeightSum, sum);
    }
    if (mesh.vertices.empty()) {
        info.minWeightSum = 0.0;
    }
    return info;
}

} // namespace sr
