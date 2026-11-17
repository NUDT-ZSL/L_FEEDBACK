#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace sr {

constexpr int kMaxBoneInfluences = 4;
constexpr int kSkeletonBoneCount = 3;

struct SkinnedVertex {
    float px = 0.0f;
    float py = 0.0f;
    float pz = 0.0f;
    float nx = 0.0f;
    float ny = 1.0f;
    float nz = 0.0f;
    std::uint32_t boneIndices[kMaxBoneInfluences] = {0, 0, 0, 0};
    float boneWeights[kMaxBoneInfluences] = {1.0f, 0.0f, 0.0f, 0.0f};
};

static_assert(sizeof(SkinnedVertex) == 56, "SkinnedVertex must stay tightly packed");

struct MeshData {
    std::string name;
    std::vector<SkinnedVertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct Bounds {
    glm::dvec3 min{0.0};
    glm::dvec3 max{0.0};
    bool valid = false;

    void expand(const glm::dvec3& p);
    void expand(const Bounds& other);
    std::array<glm::dvec3, 8> corners() const;
    glm::dvec3 center() const;
    glm::dvec3 extent() const;
};

Bounds meshBounds(const MeshData& mesh);

struct MeshParams {
    int variant = 0;
    double bodyWidth = 1.0;
    double heightScale = 1.0;
    double headScale = 1.0;
    double armSpan = 1.0;
};

MeshParams meshParamsForVariant(int variant);

struct Camera {
    glm::dvec3 eye{0.0};
    glm::dvec3 target{0.0};
    glm::dvec3 up{0.0, 1.0, 0.0};
    double fovYDegrees = 45.0;
    double aspect = 16.0 / 9.0;
    double nearPlane = 0.5;
    double farPlane = 30.0;
    glm::dmat4 view{1.0};
    glm::dmat4 projection{1.0};
    glm::dmat4 viewProjection{1.0};
};

Camera evaluateCamera(double seconds);
void updateCameraMatrices(Camera& camera);

struct DirectionalLight {
    glm::dvec3 directionTowardScene{-0.45, -0.80, -0.35};
    glm::dvec3 color{1.0, 0.94, 0.84};
    double intensity = 2.3;
};

struct PointLight {
    glm::dvec3 position{0.0};
    glm::dvec3 color{1.0};
    double intensity = 1.0;
    double range = 8.0;
};

struct Lights {
    DirectionalLight directional;
    PointLight point0;
    PointLight point1;
    glm::dvec3 ambientColor{0.18, 0.22, 0.28};
    double ambientIntensity = 0.32;
};

struct Material {
    glm::dvec3 baseColor{1.0};
    double roughness = 0.55;
    double specular = 0.20;
    double checkerScale = 0.0;
};

struct MeshRevisionInfo {
    int revision = 0;
    int variant = 0;
    int vertexCount = 0;
    int triangleCount = 0;
    int maxInfluences = 0;
    double minWeightSum = 0.0;
    double maxWeightSum = 0.0;
};

inline glm::mat4 toMat4(const glm::dmat4& m) {
    return glm::mat4(m);
}

inline glm::vec4 toVec4(const glm::dvec3& v, double w = 0.0) {
    return glm::vec4(static_cast<float>(v.x), static_cast<float>(v.y),
                     static_cast<float>(v.z), static_cast<float>(w));
}

} // namespace sr
