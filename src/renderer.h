#pragma once

#include "core.h"
#include "scene.h"

#include <cstddef>
#include <string>
#include <vector>

enum class RenderMode { Full, Incremental };

struct ChangeLogEntry {
    int index = 0;
    std::string target;
    std::string kind;
    std::vector<int> affectedTiles;
    std::string frameHash;
};

struct TileHash {
    int tile = 0;
    std::string hash;
};

struct RockWorldCenter {
    std::string id;
    Vec3 position;
};

struct RenderStats {
    std::string status = "ok";
    std::string mode;
    std::string frameHash;
    std::string depthHash;
    std::string combinedHash;
    std::string fullReferenceFrameHash;
    std::string fullReferenceDepthHash;
    int width = 0;
    int height = 0;
    int tileSize = 0;
    int tileColumns = 0;
    int tileRows = 0;
    int depthBits = 0;
    uint32_t depthMax = 0;
    int stencilBits = 0;
    int changesApplied = 0;
    int affectedTileCount = 0;
    int renderedTileCount = 0;
    int initialFullTileCount = 0;
    std::vector<int> affectedTiles;
    std::vector<TileHash> tileHashes;
    std::vector<std::vector<int>> tileDependencies;
    std::vector<RockWorldCenter> rockWorldCenters;
    size_t shadowVolumeTriangles = 0;
    size_t stencilOverflowCount = 0;
    size_t stencilDepthPassTests = 0;
    size_t stencilDepthFailTests = 0;
    int maxTemplatePassCountAbs = 0;
    int maxTemplateFailCountAbs = 0;
    size_t nonBackgroundPixels = 0;
    size_t shadowMaskedBackgroundPixels = 0;
    size_t distinctDepthValues = 0;
    uint32_t minNonBackgroundDepth = 0;
    uint32_t maxNonBackgroundDepth = 0;
    bool divergenceChecked = false;
    bool framebufferHashesEqual = false;
    bool tileHashesEqual = false;
    int unaffectedTileHashMismatches = 0;
    std::vector<ChangeLogEntry> changeLog;

    double loadAndProceduralMs = 0.0;
    double rasterizationMs = 0.0;
    double shadowMs = 0.0;
    double shadingMs = 0.0;
    double cacheBuildMs = 0.0;
    double incrementalUpdateMs = 0.0;
    double fullComparisonMs = 0.0;
    double totalMs = 0.0;
};

struct RenderResult {
    bool ok = false;
    bool hasFrame = false;
    std::string code;
    std::string message;
    int exitCode = 0;
    Framebuffer frame;
    RenderStats stats;
};

RenderResult runRender(Scene scene, RenderMode mode);
