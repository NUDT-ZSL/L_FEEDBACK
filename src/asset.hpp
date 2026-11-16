#pragma once

#include "error.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace skel {

constexpr std::uint32_t kAssetVersion = 1;
constexpr std::size_t kMaxAssetBytes = 512U * 1024U * 1024U;
constexpr std::uint32_t kMaxNodes = 65'536;
constexpr std::uint32_t kMaxSamples = 4'194'304;
constexpr std::uint32_t kMaxMeshes = 256;
constexpr std::uint32_t kMaxVerticesPerMesh = 4'000'000;
constexpr std::uint32_t kMaxIndicesPerMesh = 24'000'000;

enum class CompositionOrder : std::uint32_t {
    TRS = 0,
    TSR = 1,
    RTS = 2,
    RST = 3,
    STR = 4,
    SRT = 5,
};

struct AnimationSample {
    float time = 0.0F;
    glm::vec3 translation{0.0F};
    glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
    glm::vec3 scale{1.0F};
};

struct Node {
    std::string name;
    std::int32_t parent = -1;
    std::uint32_t firstSample = 0;
    std::uint32_t sampleCount = 0;
    glm::mat4 inverseBind{1.0F};
};

struct Vertex {
    glm::vec3 position{0.0F};
    glm::vec3 normal{0.0F};
    glm::vec3 color{1.0F};
    std::array<std::uint32_t, 4> boneIndices{0, 0, 0, 0};
    std::array<float, 4> weights{0.0F, 0.0F, 0.0F, 0.0F};
    std::uint32_t weightCount = 0;
};

struct Mesh {
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct Asset {
    std::uint32_t version = kAssetVersion;
    CompositionOrder compositionOrder = CompositionOrder::TRS;
    float framesPerSecond = 24.0F;
    std::uint32_t frameCount = 0;
    std::vector<Node> nodes;
    std::vector<AnimationSample> samples;
    std::vector<Mesh> meshes;

    std::size_t vertexCount() const;
    std::size_t triangleCount() const;
};

const char* compositionOrderName(CompositionOrder order);
bool isFinite(const glm::mat4& value);
bool isFinite(const glm::vec3& value);
bool isFinite(const glm::quat& value);

} // namespace skel

