#pragma once

#include "terrain/cli.hpp"
#include "terrain/lod.hpp"
#include "terrain/seams.hpp"

#include <cstdint>
#include <vector>

namespace terrain {

struct RenderStats {
    uint64_t triangles = 0;
    uint32_t drawCalls = 0;
    uint32_t totalDrawCalls = 0;
    uint64_t stitchVertices = 0;
    uint64_t seamSegments = 0;
    uint64_t seamLineSegments = 0;
    float maxStitchResidual = 0.0f;
};

struct SceneFrame {
    Camera camera;
    SelectionResult selection;
    std::vector<SeamResult> seams;
    RenderStats stats;
    uint64_t frameNumber = 0;
    float time = 0.0f;
};

SceneFrame buildSceneFrame(
    const Terrain& terrain,
    LodSelector& selector,
    SeamSolver& seams,
    NodeStateCache& cache,
    const Camera& camera,
    int width,
    int height,
    DebugView view,
    uint64_t frameNumber,
    float time);

}  // namespace terrain

