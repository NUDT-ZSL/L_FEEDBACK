#pragma once

#include "terrain/camera_path.hpp"
#include "terrain/terrain.hpp"

#include <cstdint>
#include <string>

namespace terrain {

enum class DebugView {
    Normal = 0,
    Wireframe = 1,
    Lod = 2,
    Seam = 3,
};

enum class Backend {
    Auto,
    Gl,
    Software,
};

struct Options {
    TerrainParameters terrain;
    std::string cameraPath;
    std::string screenshot;
    std::string captureDirectory;
    float timeStep = 0.1f;
    int frames = 1;
    int width = 960;
    int height = 540;
    float sseThreshold = 16.0f;
    int cacheCapacity = 2048;
    DebugView debugView = DebugView::Normal;
    Backend backend = Backend::Auto;
    bool replay = false;
    bool headless = false;
    bool interactive = false;
    bool help = false;
};

struct ParseResult {
    bool ok = false;
    bool isPathError = false;
    Options options;
    std::string error;
};

ParseResult parseCommandLine(int argc, char** argv);
std::string debugViewName(DebugView view);
std::string usageText();

}  // namespace terrain

