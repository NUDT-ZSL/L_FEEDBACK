#pragma once

#include "terrain/cache.hpp"
#include "terrain/math.hpp"
#include "terrain/terrain.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace terrain {

struct NodeKey {
    uint8_t depth = 0;
    uint16_t ix = 0;
    uint16_t iz = 0;

    uint64_t encode() const {
        return (static_cast<uint64_t>(depth) << 58) |
               (static_cast<uint64_t>(iz) << 29) |
               static_cast<uint64_t>(ix);
    }
};

struct Camera {
    Vec3 position{256.0f, 210.0f, 520.0f};
    float yawDegrees = -12.0f;
    float pitchDegrees = -18.0f;
    float fovYDegrees = 60.0f;
    float nearPlane = 0.5f;
    float farPlane = 2400.0f;

    Vec3 forward() const;

    Mat4 viewMatrix() const;
    Mat4 projectionMatrix(float aspect) const;
    Mat4 viewProjectionMatrix(float aspect) const;
    Frustum frustum(float aspect) const;
};

struct NodeVisitStats {
    uint64_t nodesTested = 0;
    uint64_t nodesCulled = 0;
    uint64_t splits = 0;
};

struct LeafNode {
    NodeKey key{};
    float x0 = 0.0f;
    float z0 = 0.0f;
    float x1 = 0.0f;
    float z1 = 0.0f;
    float minHeight = 0.0f;
    float maxHeight = 0.0f;
    std::array<float, kPatchVertexCount * kPatchVertexCount> vertexY{};
};

struct SelectionResult {
    std::vector<LeafNode> leaves;
    std::array<uint32_t, 8> leavesByLod{};
    NodeVisitStats visits{};
    CacheStats cacheStats{};
    uint32_t drawCalls = 0;
    uint64_t triangles = 0;
    uint32_t height = 0;
    float aspect = 1.0f;
};

class LodSelector {
public:
    LodSelector(const Terrain& terrain, float sseThreshold);

    SelectionResult select(
        const Camera& camera,
        int viewportWidth,
        int viewportHeight,
        uint64_t frameNumber,
        NodeStateCache& cache) const;

private:
    void visit(
        uint8_t depth,
        uint16_t ix,
        uint16_t iz,
        const Camera& camera,
        const Frustum& frustum,
        float projectionScale,
        uint64_t frameNumber,
        NodeStateCache& cache,
        SelectionResult& result) const;

    const Terrain& terrain_;
    float sseThreshold_;
};

}  // namespace terrain

