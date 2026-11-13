#include "app.hpp"

#include "terrain/cache.hpp"
#include "terrain/camera_path.hpp"
#include "terrain/lod.hpp"
#include "terrain/ppm.hpp"
#include "terrain/seams.hpp"
#include "terrain/software_renderer.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>

namespace terrain {
namespace {

PathLoadResult resolvePath(const Options& options) {
    if (options.cameraPath.empty()) {
        PathLoadResult result;
        result.ok = true;
        result.path = builtInCameraPath();
        return result;
    }
    return loadCameraPath(options.cameraPath);
}

std::string captureFilename(const std::string& directory, uint64_t frame) {
    std::ostringstream name;
    name << "frame_" << std::setw(6) << std::setfill('0') << frame << ".ppm";
    return (std::filesystem::path(directory) / name.str()).string();
}

}  // namespace

void printStatsJson(
    std::ostream& output,
    const SceneFrame& frame,
    const Terrain& terrain,
    DebugView view) {
    std::ostringstream line;
    line.imbue(std::locale::classic());
    line << std::fixed;
    line << "{"
         << "\"frame\":" << frame.frameNumber
         << ",\"time\":" << std::setprecision(6) << frame.time
         << ",\"seed\":" << terrain.parameters().seed
         << ",\"amplitude\":" << std::setprecision(6)
         << terrain.parameters().amplitude
         << ",\"frequency\":" << terrain.parameters().frequency
         << ",\"octaves\":" << terrain.parameters().octaves
         << ",\"grid\":" << terrain.gridSize()
         << ",\"maxDepth\":" << terrain.maxDepth()
         << ",\"view\":\"" << debugViewName(view) << "\""
         << ",\"visibleNodes\":" << frame.selection.leaves.size()
         << ",\"leavesByLod\":{";
    for (int depth = 0; depth <= terrain.maxDepth(); ++depth) {
        if (depth > 0) {
            line << ",";
        }
        line << "\"" << depth << "\":"
             << frame.selection.leavesByLod[static_cast<std::size_t>(depth)];
    }
    line << "}"
         << ",\"triangles\":" << frame.stats.triangles
         << ",\"drawCalls\":" << frame.stats.drawCalls
         << ",\"totalDrawCalls\":" << frame.stats.totalDrawCalls
         << ",\"seamDrawCalls\":" << (frame.stats.totalDrawCalls - frame.stats.drawCalls)
         << ",\"stitchVertices\":" << frame.stats.stitchVertices
         << ",\"seamSegments\":" << frame.stats.seamSegments
         << ",\"seamLineSegments\":" << frame.stats.seamLineSegments
         << ",\"nodesTested\":" << frame.selection.visits.nodesTested
         << ",\"nodesCulled\":" << frame.selection.visits.nodesCulled
         << ",\"splits\":" << frame.selection.visits.splits
         << ",\"cacheHits\":" << frame.selection.cacheStats.hits
         << ",\"cacheMisses\":" << frame.selection.cacheStats.misses
         << ",\"cacheEvictions\":" << frame.selection.cacheStats.evictions
         << ",\"cacheInvalidations\":"
         << frame.selection.cacheStats.versionInvalidations
         << ",\"maxStitchResidual\":" << std::setprecision(9)
         << frame.stats.maxStitchResidual
         << ",\"terrainVersion\":\"" << std::hex << terrain.version() << std::dec
         << "\",\"heightChecksum\":\"" << std::hex << terrain.heightChecksum()
         << std::dec << "\"}" << "\n";
    output << line.str();
}

int runSoftwareApp(const Options& options) {
    if (options.interactive) {
        std::cerr << "E_GL_INIT: --interactive requires the OpenGL backend\n";
        return 4;
    }

    PathLoadResult pathResult = resolvePath(options);
    if (!pathResult.ok) {
        std::cerr << "E_BAD_PATH: " << pathResult.error;
        if (pathResult.line > 0) {
            std::cerr << " at line " << pathResult.line;
        }
        std::cerr << "\n";
        return 3;
    }

    Terrain terrain(options.terrain);
    LodSelector selector(terrain, options.sseThreshold);
    SeamSolver seams(terrain);
    NodeStateCache cache(static_cast<std::size_t>(options.cacheCapacity));
    SoftwareRenderer renderer(terrain, options.width, options.height);

    const bool captureEveryFrame = !options.captureDirectory.empty();
    for (int frameIndex = 0; frameIndex < options.frames; ++frameIndex) {
        const float time =
            static_cast<float>(frameIndex) * options.timeStep;
        const Camera camera = pathResult.path.sample(
            time,
            static_cast<float>(options.width) /
                static_cast<float>(options.height));
        const SceneFrame frame = buildSceneFrame(
            terrain,
            selector,
            seams,
            cache,
            camera,
            options.width,
            options.height,
            options.debugView,
            static_cast<uint64_t>(frameIndex),
            time);
        printStatsJson(std::cout, frame, terrain, options.debugView);
        std::cout.flush();

        const bool finalFrame = frameIndex + 1 == options.frames;
        const bool shouldCapture = captureEveryFrame || finalFrame;
        if (shouldCapture &&
            (captureEveryFrame || !options.screenshot.empty())) {
            const ImageRgb image = renderer.render(frame, options.debugView);
            std::string filename;
            if (captureEveryFrame) {
                filename = captureFilename(
                    options.captureDirectory,
                    static_cast<uint64_t>(frameIndex));
            } else {
                filename = options.screenshot;
            }
            std::string error;
            if (!writePpm(filename, image, error)) {
                std::cerr << "E_RENDER_FAIL: " << error
                          << " (" << filename << ")\n";
                return 4;
            }
        }
    }
    return 0;
}

}  // namespace terrain

