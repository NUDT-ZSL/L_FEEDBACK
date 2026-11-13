#pragma once

#include "terrain/cache.hpp"
#include "terrain/lod.hpp"
#include "terrain/terrain.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace terrain {

enum class Edge : uint8_t {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
};

struct StitchSegment {
    Edge edge = Edge::North;
    float t0 = 0.0f;
    float t1 = 0.0f;
    uint32_t neighborLeafIndex = 0;
    uint8_t neighborDepth = 0;
    float neighborMinX = 0.0f;
    float neighborMinZ = 0.0f;
    float neighborSpan = 0.0f;
};

struct SeamLineVertex {
    Vec3 position;
};

struct SeamSolveStats {
    uint64_t stitchVertices = 0;
    uint64_t seamSegments = 0;
    uint64_t seamLineSegments = 0;
    float maxResidual = 0.0f;
};

struct SeamResult {
    std::array<std::vector<StitchSegment>, 4> edgeSegments;
    std::vector<SeamLineVertex> seamLines;
    SeamSolveStats stats;
};

class SeamSolver {
public:
    explicit SeamSolver(const Terrain& terrain);

    std::vector<SeamResult> solve(
        SelectionResult& selection,
        uint64_t frameNumber,
        NodeStateCache& cache) const;

private:
    const Terrain& terrain_;
};

}  // namespace terrain

