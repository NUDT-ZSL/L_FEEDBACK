#pragma once

#include "cv_math.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cv {

constexpr int kMaxBoneInfluences = 4;

struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec4 tangent;  // xyz = T, w = bitangent handedness (+1 or -1)
    Vec2 uv;
    std::array<uint16_t, kMaxBoneInfluences> boneIndices{};
    std::array<float, kMaxBoneInfluences> boneWeights{};
};

struct Material {
    std::string name;
    Vec3 baseColor{1.0f, 1.0f, 1.0f};
    Vec3 emissive{0.0f, 0.0f, 0.0f};
    float roughness = 0.5f;
    float metallic = 0.0f;
    float opacity = 1.0f;
    float normalStrength = 1.0f;
    bool transparent = false;
    bool doubleSided = false;
    bool castsShadow = true;
    uint64_t stateKey = 0;
};

struct NormalTexture {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
};

struct Mesh {
    std::string name;
    bool skinned = false;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<uint16_t> triangleMaterial;
};

struct Bone {
    std::string name;
    int parent = -1;
    Vec3 localPosition;
    Vec3 bindEuler;
};

struct Skeleton {
    std::vector<Bone> bones;
    std::vector<Mat4> bindGlobal;
    std::vector<Mat4> inverseBind;
};

struct Scene {
    std::vector<Material> materials;
    std::vector<Mesh> meshes;
    Skeleton skeleton;
    NormalTexture normalMap;

    size_t totalTriangles() const;
    size_t totalVertices() const;
};

struct FrameVertex {
    Vec3 worldPosition;
    Vec3 worldNormal;
    Vec4 worldTangent;
    Vec2 uv;
};

struct Light {
    Vec3 position{};
    Vec3 color{1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    float radius = 0.08f;
};

struct Lighting {
    // Direction is the direction in which the directional light travels.
    Vec3 directionalDirection = normalize(Vec3{0.35f, -0.75f, -0.55f});
    Vec3 directionalColor{1.0f, 0.93f, 0.82f};
    float directionalIntensity = 1.45f;
    std::array<Light, 2> pointLights{{
        {{-2.35f, 2.15f, 1.95f}, {0.20f, 0.58f, 1.0f}, 2.8f, 0.07f},
        {{2.25f, 1.25f, -1.65f}, {1.0f, 0.34f, 0.12f}, 2.1f, 0.06f},
    }};
    Vec3 ambientSky{0.055f, 0.085f, 0.14f};
    Vec3 ambientGround{0.025f, 0.022f, 0.018f};
};

std::vector<Mat4> buildBoneMatrices(const Skeleton& skeleton, uint32_t tick);
std::vector<FrameVertex> skinVertices(const Mesh& mesh, const std::vector<Mat4>& boneMatrices);

struct AssetBundle {
    Scene scene;
    Lighting lighting;
};

AssetBundle makeProceduralScene();

}  // namespace cv

