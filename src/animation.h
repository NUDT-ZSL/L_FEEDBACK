#pragma once

#include <vector>

#include "types.h"

namespace sr {

struct Keyframe {
    double time = 0.0;
    glm::dvec3 translation{0.0};
    glm::dquat rotation{1.0, 0.0, 0.0, 0.0};
};

struct Bone {
    std::string name;
    int parent = -1;
    glm::dvec3 bindTranslation{0.0};
    glm::dquat bindRotation{1.0, 0.0, 0.0, 0.0};
    std::vector<Keyframe> keyframes;
};

struct Skeleton {
    std::vector<Bone> bones;
    std::vector<glm::dmat4> inverseBind;
};

struct PoseSample {
    glm::dvec3 translation{0.0};
    glm::dquat rotation{1.0, 0.0, 0.0, 0.0};
    double segmentTime0 = 0.0;
    double segmentTime1 = 0.0;
    double fraction = 0.0;
    int segmentIndex = 0;
    bool clampedBefore = false;
    bool clampedAfter = false;
};

struct PoseResult {
    double time = 0.0;
    std::vector<glm::dmat4> worldMatrices;
    std::vector<glm::dmat4> skinMatrices;
    std::vector<glm::dquat> localRotations;
    std::vector<PoseSample> samples;
};

Skeleton makeSkeleton();
PoseSample sampleKeyframes(const std::vector<Keyframe>& keys, double time);
PoseResult evaluatePose(const Skeleton& skeleton, double time);

glm::dquat shortestSlerp(const glm::dquat& a, const glm::dquat& b, double t);
glm::dvec3 skinPosition(const SkinnedVertex& vertex,
                        const std::vector<glm::dmat4>& skinMatrices);
glm::dvec3 skinNormal(const SkinnedVertex& vertex,
                      const std::vector<glm::dmat4>& skinMatrices);
void normalizeVertexWeights(SkinnedVertex& vertex);
void validateVertexWeights(const SkinnedVertex& vertex, const char* context);
MeshRevisionInfo inspectMeshWeights(const MeshData& mesh);

} // namespace sr
