#include "terrain/seams.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace terrain {
namespace {

constexpr int kMaxOwnerGrid = 1 << 8;

float edgeParameter(const LeafNode& leaf, Edge edge, float value) {
    const float span = (edge == Edge::North || edge == Edge::South)
                           ? leaf.x1 - leaf.x0
                           : leaf.z1 - leaf.z0;
    const float origin = (edge == Edge::North || edge == Edge::South)
                             ? leaf.x0
                             : leaf.z0;
    if (span <= 0.0f) {
        return 0.0f;
    }
    return clampf((value - origin) / span, 0.0f, 1.0f);
}

Vec3 edgePoint(const LeafNode& leaf, Edge edge, float t, float height) {
    const float xSpan = leaf.x1 - leaf.x0;
    const float zSpan = leaf.z1 - leaf.z0;
    switch (edge) {
        case Edge::North:
            return {leaf.x0 + xSpan * t, height, leaf.z0};
        case Edge::East:
            return {leaf.x1, height, leaf.z0 + zSpan * t};
        case Edge::South:
            return {leaf.x0 + xSpan * t, height, leaf.z1};
        case Edge::West:
            return {leaf.x0, height, leaf.z0 + zSpan * t};
    }
    return {leaf.x0, height, leaf.z0};
}

int boundaryLocalX(Edge edge, int index) {
    switch (edge) {
        case Edge::North:
        case Edge::South:
            return index;
        case Edge::East:
            return kPatchQuads;
        case Edge::West:
            return 0;
    }
    return 0;
}

int boundaryLocalZ(Edge edge, int index) {
    switch (edge) {
        case Edge::West:
        case Edge::East:
            return index;
        case Edge::North:
            return 0;
        case Edge::South:
            return kPatchQuads;
    }
    return 0;
}

float lowEdgeHeight(
    const Terrain& terrain,
    const LeafNode& high,
    const LeafNode& low,
    Edge edge,
    float t) {
    const float lowStep = (low.x1 - low.x0) / static_cast<float>(kPatchQuads);
    const float lowZStep = (low.z1 - low.z0) / static_cast<float>(kPatchQuads);
    float along = 0.0f;
    float step = 0.0f;
    float fixedCoordinate = 0.0f;
    bool alongX = false;

    switch (edge) {
        case Edge::North:
        case Edge::South:
            along = high.x0 + (high.x1 - high.x0) * t;
            step = lowStep;
            fixedCoordinate = edge == Edge::North ? high.z0 : high.z1;
            alongX = true;
            break;
        case Edge::East:
        case Edge::West:
            along = high.z0 + (high.z1 - high.z0) * t;
            step = lowZStep;
            fixedCoordinate = edge == Edge::East ? high.x1 : high.x0;
            alongX = false;
            break;
    }

    const float lowOrigin = alongX ? low.x0 : low.z0;
    const float local = clampf(
        (along - lowOrigin) / std::max(step, 1.0e-6f),
        0.0f,
        static_cast<float>(kPatchQuads));
    const int i0 = std::clamp(
        static_cast<int>(std::floor(local)), 0, kPatchQuads);
    const int i1 = std::min(i0 + 1, kPatchQuads);
    const float fraction = local - static_cast<float>(i0);

    const float a = alongX
                        ? lowOrigin + static_cast<float>(i0) * step
                        : lowOrigin + static_cast<float>(i0) * step;
    const float b = alongX
                        ? lowOrigin + static_cast<float>(i1) * step
                        : lowOrigin + static_cast<float>(i1) * step;

    const float h0 = alongX
                         ? terrain.sampleHeight(a, fixedCoordinate)
                         : terrain.sampleHeight(fixedCoordinate, a);
    const float h1 = alongX
                         ? terrain.sampleHeight(b, fixedCoordinate)
                         : terrain.sampleHeight(fixedCoordinate, b);
    return h0 + (h1 - h0) * fraction;
}

const StitchSegment* findSegment(
    const std::vector<StitchSegment>& segments,
    float t) {
    for (const StitchSegment& segment : segments) {
        if (t >= segment.t0 - 1.0e-6f && t <= segment.t1 + 1.0e-6f) {
            return &segment;
        }
    }
    return nullptr;
}

float interpolateBoundaryHeight(
    const LeafNode& leaf,
    Edge edge,
    float t) {
    const float scaled = clampf(t, 0.0f, 1.0f) * static_cast<float>(kPatchQuads);
    const int i0 = std::clamp(
        static_cast<int>(std::floor(scaled)), 0, kPatchQuads);
    const int i1 = std::min(i0 + 1, kPatchQuads);
    const float fraction = scaled - static_cast<float>(i0);

    const int x0 = boundaryLocalX(edge, i0);
    const int z0 = boundaryLocalZ(edge, i0);
    const int x1 = boundaryLocalX(edge, i1);
    const int z1 = boundaryLocalZ(edge, i1);
    const float h0 =
        leaf.vertexY[static_cast<std::size_t>(z0) * kPatchVertexCount +
                     static_cast<std::size_t>(x0)];
    const float h1 =
        leaf.vertexY[static_cast<std::size_t>(z1) * kPatchVertexCount +
                     static_cast<std::size_t>(x1)];
    return h0 + (h1 - h0) * fraction;
}

}  // namespace

SeamSolver::SeamSolver(const Terrain& terrain)
    : terrain_(terrain) {}

std::vector<SeamResult> SeamSolver::solve(
    SelectionResult& selection,
    uint64_t frameNumber,
    NodeStateCache& cache) const {
    std::vector<SeamResult> results(selection.leaves.size());
    if (selection.leaves.empty()) {
        return results;
    }

    const int maxDepth = terrain_.maxDepth();
    if (maxDepth >= kMaxOwnerGrid) {
        return results;
    }
    const int ownerGridSize = 1 << maxDepth;
    std::vector<int> owner(
        static_cast<std::size_t>(ownerGridSize) * ownerGridSize,
        -1);

    for (std::size_t leafIndex = 0; leafIndex < selection.leaves.size(); ++leafIndex) {
        const LeafNode& leaf = selection.leaves[leafIndex];
        const int cellsPerNode = ownerGridSize >> leaf.key.depth;
        const int cellX0 = static_cast<int>(leaf.key.ix) * cellsPerNode;
        const int cellZ0 = static_cast<int>(leaf.key.iz) * cellsPerNode;
        for (int z = cellZ0; z < cellZ0 + cellsPerNode; ++z) {
            for (int x = cellX0; x < cellX0 + cellsPerNode; ++x) {
                owner[static_cast<std::size_t>(z) * ownerGridSize +
                      static_cast<std::size_t>(x)] = static_cast<int>(leafIndex);
            }
        }
    }

    const auto collectNeighbor = [&owner, ownerGridSize](
                                     int cellX, int cellZ) -> int {
        if (cellX < 0 || cellZ < 0 || cellX >= ownerGridSize ||
            cellZ >= ownerGridSize) {
            return -1;
        }
        return owner[static_cast<std::size_t>(cellZ) * ownerGridSize +
                     static_cast<std::size_t>(cellX)];
    };

    for (std::size_t leafIndex = 0; leafIndex < selection.leaves.size(); ++leafIndex) {
        LeafNode& leaf = selection.leaves[leafIndex];
        const int cellsPerNode = ownerGridSize >> leaf.key.depth;
        const int cellX0 = static_cast<int>(leaf.key.ix) * cellsPerNode;
        const int cellZ0 = static_cast<int>(leaf.key.iz) * cellsPerNode;
        const int cellX1 = cellX0 + cellsPerNode;
        const int cellZ1 = cellZ0 + cellsPerNode;

        std::vector<int> neighbors;
        const auto addNeighbor = [&neighbors](int value) {
            if (value >= 0) {
                neighbors.push_back(value);
            }
        };
        for (int x = cellX0; x < cellX1; ++x) {
            addNeighbor(collectNeighbor(x, cellZ0 - 1));
            addNeighbor(collectNeighbor(x, cellZ1));
        }
        for (int z = cellZ0; z < cellZ1; ++z) {
            addNeighbor(collectNeighbor(cellX0 - 1, z));
            addNeighbor(collectNeighbor(cellX1, z));
        }
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

        for (int neighborIndex : neighbors) {
            if (neighborIndex < 0 ||
                static_cast<std::size_t>(neighborIndex) >= selection.leaves.size()) {
                continue;
            }
            const LeafNode& other =
                selection.leaves[static_cast<std::size_t>(neighborIndex)];
            if (other.key.depth >= leaf.key.depth) {
                continue;
            }

            const bool touchesNorth = std::abs(other.z1 - leaf.z0) < 1.0e-4f;
            const bool touchesSouth = std::abs(other.z0 - leaf.z1) < 1.0e-4f;
            const bool touchesWest = std::abs(other.x1 - leaf.x0) < 1.0e-4f;
            const bool touchesEast = std::abs(other.x0 - leaf.x1) < 1.0e-4f;

            const auto addSegment = [&](Edge edge, float overlapStart, float overlapEnd) {
                if (overlapEnd <= overlapStart + 1.0e-5f) {
                    return;
                }
                StitchSegment segment;
                segment.edge = edge;
                segment.t0 = edgeParameter(leaf, edge, overlapStart);
                segment.t1 = edgeParameter(leaf, edge, overlapEnd);
                segment.neighborLeafIndex = static_cast<uint32_t>(neighborIndex);
                segment.neighborDepth = other.key.depth;
                segment.neighborMinX = other.x0;
                segment.neighborMinZ = other.z0;
                segment.neighborSpan = other.x1 - other.x0;
                results[leafIndex].edgeSegments[static_cast<std::size_t>(edge)]
                    .push_back(segment);
            };

            if (touchesNorth) {
                addSegment(
                    Edge::North,
                    std::max(leaf.x0, other.x0),
                    std::min(leaf.x1, other.x1));
            }
            if (touchesSouth) {
                addSegment(
                    Edge::South,
                    std::max(leaf.x0, other.x0),
                    std::min(leaf.x1, other.x1));
            }
            if (touchesWest) {
                addSegment(
                    Edge::West,
                    std::max(leaf.z0, other.z0),
                    std::min(leaf.z1, other.z1));
            }
            if (touchesEast) {
                addSegment(
                    Edge::East,
                    std::max(leaf.z0, other.z0),
                    std::min(leaf.z1, other.z1));
            }
        }

        for (auto& segments : results[leafIndex].edgeSegments) {
            std::sort(
                segments.begin(),
                segments.end(),
                [](const StitchSegment& a, const StitchSegment& b) {
                    return a.t0 < b.t0;
                });
        }

        std::array<float, kPatchVertexCount * kPatchVertexCount> rawHeights{};
        for (int localZ = 0; localZ < kPatchVertexCount; ++localZ) {
            for (int localX = 0; localX < kPatchVertexCount; ++localX) {
                const float worldX =
                    leaf.x0 + (leaf.x1 - leaf.x0) *
                                  (static_cast<float>(localX) /
                                   static_cast<float>(kPatchQuads));
                const float worldZ =
                    leaf.z0 + (leaf.z1 - leaf.z0) *
                                  (static_cast<float>(localZ) /
                                   static_cast<float>(kPatchQuads));
                const float height = terrain_.sampleHeight(worldX, worldZ);
                const std::size_t index =
                    static_cast<std::size_t>(localZ) * kPatchVertexCount +
                    static_cast<std::size_t>(localX);
                rawHeights[index] = height;
                leaf.vertexY[index] = height;
            }
        }

        const std::array<Edge, 4> priority{
            Edge::North, Edge::East, Edge::South, Edge::West};
        for (Edge edge : priority) {
            const std::vector<StitchSegment>& segments =
                results[leafIndex].edgeSegments[static_cast<std::size_t>(edge)];
            if (segments.empty()) {
                continue;
            }

            for (int localIndex = 0; localIndex <= kPatchQuads; ++localIndex) {
                const float t =
                    static_cast<float>(localIndex) /
                    static_cast<float>(kPatchQuads);
                const StitchSegment* segment = findSegment(segments, t);
                if (segment == nullptr) {
                    continue;
                }
                const LeafNode& low =
                    selection.leaves[segment->neighborLeafIndex];
                const float stitchedHeight =
                    lowEdgeHeight(terrain_, leaf, low, edge, t);
                const int localX = boundaryLocalX(edge, localIndex);
                const int localZ = boundaryLocalZ(edge, localIndex);
                const std::size_t index =
                    static_cast<std::size_t>(localZ) * kPatchVertexCount +
                    static_cast<std::size_t>(localX);
                leaf.vertexY[index] = stitchedHeight;
            }
        }

        std::vector<CachedStitchOffset> offsets;
        for (int localZ = 0; localZ < kPatchVertexCount; ++localZ) {
            for (int localX = 0; localX < kPatchVertexCount; ++localX) {
                const bool boundary =
                    localX == 0 || localX == kPatchQuads ||
                    localZ == 0 || localZ == kPatchQuads;
                if (!boundary) {
                    continue;
                }
                const std::size_t index =
                    static_cast<std::size_t>(localZ) * kPatchVertexCount +
                    static_cast<std::size_t>(localX);
                const float offset = leaf.vertexY[index] - rawHeights[index];
                if (std::abs(offset) > 1.0e-6f) {
                    offsets.push_back(
                        {static_cast<uint16_t>(index), offset});
                    ++results[leafIndex].stats.stitchVertices;
                }
            }
        }
        cache.storeStitchOffsets(leaf.key.encode(), offsets);

        for (Edge edge : priority) {
            const std::vector<StitchSegment>& segments =
                results[leafIndex].edgeSegments[static_cast<std::size_t>(edge)];
            for (const StitchSegment& segment : segments) {
                ++results[leafIndex].stats.seamSegments;
                for (int localIndex = 0; localIndex < kPatchQuads; ++localIndex) {
                    const float a =
                        static_cast<float>(localIndex) /
                        static_cast<float>(kPatchQuads);
                    const float b =
                        static_cast<float>(localIndex + 1) /
                        static_cast<float>(kPatchQuads);
                    const float start = std::max(a, segment.t0);
                    const float end = std::min(b, segment.t1);
                    if (end <= start + 1.0e-5f) {
                        continue;
                    }
                    const float startHeight =
                        interpolateBoundaryHeight(leaf, edge, start);
                    const float endHeight =
                        interpolateBoundaryHeight(leaf, edge, end);
                    results[leafIndex].seamLines.push_back(
                        {edgePoint(leaf, edge, start, startHeight + 0.025f)});
                    results[leafIndex].seamLines.push_back(
                        {edgePoint(leaf, edge, end, endHeight + 0.025f)});
                    ++results[leafIndex].stats.seamLineSegments;
                }
            }

            for (int localIndex = 0; localIndex <= kPatchQuads; ++localIndex) {
                const float t =
                    static_cast<float>(localIndex) /
                    static_cast<float>(kPatchQuads);
                const StitchSegment* segment = findSegment(segments, t);
                if (segment == nullptr) {
                    continue;
                }
                const int localX = boundaryLocalX(edge, localIndex);
                const int localZ = boundaryLocalZ(edge, localIndex);
                const std::size_t index =
                    static_cast<std::size_t>(localZ) * kPatchVertexCount +
                    static_cast<std::size_t>(localX);
                const LeafNode& low =
                    selection.leaves[segment->neighborLeafIndex];
                const float expected =
                    lowEdgeHeight(terrain_, leaf, low, edge, t);
                const float residual =
                    std::abs(leaf.vertexY[index] - expected);
                results[leafIndex].stats.maxResidual =
                    std::max(results[leafIndex].stats.maxResidual, residual);
            }
        }
    }

    (void)frameNumber;
    return results;
}

}  // namespace terrain

