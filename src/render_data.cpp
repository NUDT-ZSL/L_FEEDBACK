#include "terrain/render_data.hpp"

#include <algorithm>

namespace terrain {

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
    float time) {
    (void)terrain;
    SceneFrame frame;
    frame.camera = camera;
    frame.frameNumber = frameNumber;
    frame.time = time;
    frame.selection = selector.select(camera, width, height, frameNumber, cache);
    frame.seams = seams.solve(frame.selection, frameNumber, cache);

    frame.stats.triangles = frame.selection.triangles;
    frame.stats.drawCalls = frame.selection.drawCalls;
    frame.stats.totalDrawCalls = frame.selection.drawCalls;
    for (const SeamResult& result : frame.seams) {
        frame.stats.stitchVertices += result.stats.stitchVertices;
        frame.stats.seamSegments += result.stats.seamSegments;
        frame.stats.seamLineSegments += result.stats.seamLineSegments;
        frame.stats.maxStitchResidual =
            std::max(frame.stats.maxStitchResidual, result.stats.maxResidual);
    }

    if (view == DebugView::Seam && !frame.seams.empty() &&
        frame.stats.seamLineSegments > 0) {
        ++frame.stats.totalDrawCalls;
    }
    return frame;
}

}  // namespace terrain

