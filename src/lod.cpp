#include "terrain/lod.hpp"

#include <algorithm>
#include <cmath>

namespace terrain {

namespace {

constexpr uint64_t kMinimumLodDwellFrames = 4;

float nodeSpan(const Terrain& terrain, uint8_t depth) {
    return terrain.extent() / static_cast<float>(1u << depth);
}

float nodeOrigin(const Terrain& terrain, uint8_t depth, uint16_t index) {
    return static_cast<float>(index) * nodeSpan(terrain, depth);
}

}  // namespace

LodSelector::LodSelector(const Terrain& terrain, float sseThreshold)
    : terrain_(terrain), sseThreshold_(std::max(sseThreshold, 1.0e-4f)) {}

SelectionResult LodSelector::select(
    const Camera& camera,
    int viewportWidth,
    int viewportHeight,
    uint64_t frameNumber,
    NodeStateCache& cache) const {
    SelectionResult result;
    result.height = static_cast<uint32_t>(viewportHeight);
    result.aspect = static_cast<float>(viewportWidth) /
                    static_cast<float>(std::max(viewportHeight, 1));

    const float aspect =
        static_cast<float>(viewportWidth) / static_cast<float>(std::max(viewportHeight, 1));
    const Frustum frustum = camera.frustum(aspect);
    const float tanHalfFov = std::tan(radians(camera.fovYDegrees) * 0.5f);
    const float projectionScale =
        static_cast<float>(viewportHeight) /
        std::max(2.0f * tanHalfFov, 1.0e-6f);

    cache.beginFrame();
    visit(
        0,
        0,
        0,
        camera,
        frustum,
        projectionScale,
        frameNumber,
        cache,
        result);
    cache.endFrame(frameNumber, result.cacheStats);

    for (uint32_t count : result.leavesByLod) {
        if (count > 0) {
            ++result.drawCalls;
        }
    }
    result.triangles =
        static_cast<uint64_t>(result.leaves.size()) *
        static_cast<uint64_t>(kPatchQuads) *
        static_cast<uint64_t>(kPatchQuads) * 2ull;
    return result;
}

void LodSelector::visit(
    uint8_t depth,
    uint16_t ix,
    uint16_t iz,
    const Camera& camera,
    const Frustum& frustum,
    float projectionScale,
    uint64_t frameNumber,
    NodeStateCache& cache,
    SelectionResult& result) const {
    ++result.visits.nodesTested;

    const float span = nodeSpan(terrain_, depth);
    const float x0 = nodeOrigin(terrain_, depth, ix);
    const float z0 = nodeOrigin(terrain_, depth, iz);
    const float x1 = std::min(x0 + span, terrain_.extent());
    const float z1 = std::min(z0 + span, terrain_.extent());
    const auto [minimumHeight, maximumHeight] =
        terrain_.heightBounds(x0, z0, x1, z1);
    const Aabb bounds{
        {x0, minimumHeight, z0},
        {x1, maximumHeight, z1},
    };

    const NodeKey key{depth, ix, iz};
    const uint64_t nodeId = key.encode();
    NodeCacheEntry& entry =
        cache.acquire(nodeId, terrain_.version(), frameNumber, result.cacheStats);

    if (!frustum.intersects(bounds)) {
        ++result.visits.nodesCulled;
        entry.selectedDepth = depth;
        entry.selectedAsLeaf = false;
        return;
    }

    const float cellSize = span / static_cast<float>(kPatchQuads);
    const float heightRange = std::max(0.0f, maximumHeight - minimumHeight);
    const float geometricError =
        cellSize * (1.0f + heightRange * 0.5f);
    const float distanceSquared =
        distanceSquaredToAabb(camera.position, bounds);
    const float distance = std::sqrt(distanceSquared);
    const float sse =
        geometricError * projectionScale / std::max(distance, 0.01f);

    bool split = false;
    if (depth < static_cast<uint8_t>(terrain_.maxDepth())) {
        const float hysteresisThreshold =
            entry.selectedAsLeaf ? sseThreshold_ * 1.15f : sseThreshold_ * 0.90f;
        split = sse > hysteresisThreshold;
        if (entry.hasSelectionHistory &&
            frameNumber - entry.lastStateChangeFrame < kMinimumLodDwellFrames) {
            split = !entry.selectedAsLeaf;
        }
    }

    const bool desiredLeaf = !split;
    if (entry.hasSelectionHistory && entry.selectedAsLeaf != desiredLeaf) {
        entry.lastStateChangeFrame = frameNumber;
    }
    entry.selectedDepth = depth;
    entry.selectedAsLeaf = desiredLeaf;
    entry.hasSelectionHistory = true;

    if (split) {
        ++result.visits.splits;
        const uint16_t childIx = static_cast<uint16_t>(ix * 2u);
        const uint16_t childIz = static_cast<uint16_t>(iz * 2u);
        visit(
            static_cast<uint8_t>(depth + 1),
            childIx,
            childIz,
            camera,
            frustum,
            projectionScale,
            frameNumber,
            cache,
            result);
        visit(
            static_cast<uint8_t>(depth + 1),
            static_cast<uint16_t>(childIx + 1),
            childIz,
            camera,
            frustum,
            projectionScale,
            frameNumber,
            cache,
            result);
        visit(
            static_cast<uint8_t>(depth + 1),
            childIx,
            static_cast<uint16_t>(childIz + 1),
            camera,
            frustum,
            projectionScale,
            frameNumber,
            cache,
            result);
        visit(
            static_cast<uint8_t>(depth + 1),
            static_cast<uint16_t>(childIx + 1),
            static_cast<uint16_t>(childIz + 1),
            camera,
            frustum,
            projectionScale,
            frameNumber,
            cache,
            result);
        return;
    }

    LeafNode leaf;
    leaf.key = key;
    leaf.x0 = x0;
    leaf.z0 = z0;
    leaf.x1 = x1;
    leaf.z1 = z1;
    leaf.minHeight = minimumHeight;
    leaf.maxHeight = maximumHeight;
    result.leaves.push_back(leaf);
    ++result.leavesByLod[depth];
}

}  // namespace terrain

