#include "renderer.h"
#include "scene.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::string scenePath;
    std::string outputDir = "build/out";
    RenderMode mode = RenderMode::Incremental;
};

void printUsage() {
    std::cout
        << "Usage: canyon_renderer --scene <scene.json> [--out <dir>] "
           "[--mode full|incremental]\n";
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--scene" && i + 1 < argc) {
            options.scenePath = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            options.outputDir = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            const std::string mode = argv[++i];
            if (mode == "full") options.mode = RenderMode::Full;
            else if (mode == "incremental") options.mode = RenderMode::Incremental;
            else throw RenderError("E_SCENE_INVALID", "invalid --mode value: " + mode);
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            std::exit(0);
        } else {
            throw RenderError("E_SCENE_INVALID", "unknown or incomplete argument: " + arg);
        }
    }
    if (options.scenePath.empty()) {
        throw RenderError("E_SCENE_INVALID", "missing required --scene argument");
    }
    return options;
}

void writePpm(const std::string& path, int width, int height,
              const std::vector<uint8_t>& pixels, bool rgb) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw RenderError("E_SCENE_INVALID", "cannot write " + path);
    out << "P6\n" << width << " " << height << "\n255\n";
    if (rgb) {
        out.write(reinterpret_cast<const char*>(pixels.data()),
                  static_cast<std::streamsize>(pixels.size()));
    } else {
        for (const uint8_t value : pixels) {
            const char bytes[3] = {static_cast<char>(value),
                                   static_cast<char>(value),
                                   static_cast<char>(value)};
            out.write(bytes, 3);
        }
    }
    if (!out) throw RenderError("E_SCENE_INVALID", "failed while writing " + path);
}

void writeColorAndDepth(const RenderResult& result, const std::string& outputDir) {
    if (!result.hasFrame) return;
    const Framebuffer& frame = result.frame;
    writePpm(outputDir + "/color.ppm", frame.width, frame.height, frame.color, true);

    std::vector<uint8_t> depthImage(frame.depth.size());
    const uint32_t maxDepth = result.stats.depthMax == 0 ? 1U : result.stats.depthMax;
    uint32_t minDepth = maxDepth;
    uint32_t maxGeometryDepth = 0;
    for (const uint32_t depth : frame.depth) {
        if (depth != maxDepth) {
            minDepth = std::min(minDepth, depth);
            maxGeometryDepth = std::max(maxGeometryDepth, depth);
        }
    }
    const uint32_t depthSpan = maxGeometryDepth > minDepth
        ? maxGeometryDepth - minDepth : 1U;
    for (size_t i = 0; i < frame.depth.size(); ++i) {
        if (frame.depth[i] == maxDepth) {
            depthImage[i] = 0U;
        } else {
            const uint32_t distance = maxGeometryDepth - frame.depth[i];
            const uint64_t scaled = static_cast<uint64_t>(distance) * 255ULL +
                                    static_cast<uint64_t>(depthSpan) / 2ULL;
            depthImage[i] = static_cast<uint8_t>(
                std::min<uint64_t>(255ULL, scaled / static_cast<uint64_t>(depthSpan)));
        }
    }
    writePpm(outputDir + "/depth.ppm", frame.width, frame.height, depthImage, false);

    std::ofstream raw(outputDir + "/depth_raw.bin", std::ios::binary);
    if (raw) {
        for (const uint32_t depth : frame.depth) {
            const uint32_t value = depth;
            const char bytes[4] = {
                static_cast<char>(value & 0xffU),
                static_cast<char>((value >> 8U) & 0xffU),
                static_cast<char>((value >> 16U) & 0xffU),
                0
            };
            raw.write(bytes, 4);
        }
    }
}

void writeIntArray(std::ostream& out, const std::vector<int>& values) {
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ",";
        out << values[i];
    }
    out << "]";
}

void writeStatsJson(const std::string& outputDir, const RenderStats& stats,
                    const std::string& code, const std::string& message) {
    std::ofstream out(outputDir + "/stats.json", std::ios::binary);
    if (!out) return;
    out << std::fixed << std::setprecision(3);
    out << "{\n";
    out << "  \"status\": \"" << (code.empty() ? "ok" : "error") << "\",\n";
    out << "  \"error\": ";
    if (code.empty()) {
        out << "null,\n";
    } else {
        const int exitCode = code == "E_INCREMENTAL_DIVERGED" ? 3 : 2;
        out << "{\"code\": \"" << jsonEscape(code) << "\", \"message\": \""
            << jsonEscape(message) << "\", \"exitCode\": " << exitCode << "},\n";
    }
    out << "  \"mode\": \"" << jsonEscape(stats.mode) << "\",\n";
    out << "  \"width\": " << stats.width << ",\n";
    out << "  \"height\": " << stats.height << ",\n";
    out << "  \"tileSize\": " << stats.tileSize << ",\n";
    out << "  \"tileColumns\": " << stats.tileColumns << ",\n";
    out << "  \"tileRows\": " << stats.tileRows << ",\n";
    out << "  \"depthBits\": " << stats.depthBits << ",\n";
    out << "  \"depthMax\": " << stats.depthMax << ",\n";
    out << "  \"stencilBits\": " << stats.stencilBits << ",\n";
    out << "  \"hashAlgorithm\": \"fnv1a64\",\n";
    out << "  \"frameHash\": \"" << stats.frameHash << "\",\n";
    out << "  \"depthHash\": \"" << stats.depthHash << "\",\n";
    out << "  \"combinedHash\": \"" << stats.combinedHash << "\",\n";
    out << "  \"fullReferenceFrameHash\": \"" << stats.fullReferenceFrameHash << "\",\n";
    out << "  \"fullReferenceDepthHash\": \"" << stats.fullReferenceDepthHash << "\",\n";
    out << "  \"changesApplied\": " << stats.changesApplied << ",\n";
    out << "  \"affectedTileCount\": " << stats.affectedTileCount << ",\n";
    out << "  \"renderedTileCount\": " << stats.renderedTileCount << ",\n";
    out << "  \"initialFullTileCount\": " << stats.initialFullTileCount << ",\n";
    out << "  \"affectedTiles\": ";
    writeIntArray(out, stats.affectedTiles);
    out << ",\n";
    out << "  \"tileHashes\": [";
    for (size_t i = 0; i < stats.tileHashes.size(); ++i) {
        if (i != 0) out << ",";
        out << "\n    {\"tile\": " << stats.tileHashes[i].tile << ", \"hash\": \""
            << stats.tileHashes[i].hash << "\"}";
    }
    if (!stats.tileHashes.empty()) out << "\n  ";
    out << "],\n";
    out << "  \"tileDependencies\": [";
    for (size_t i = 0; i < stats.tileDependencies.size(); ++i) {
        if (i != 0) out << ",";
        out << "\n    ";
        writeIntArray(out, stats.tileDependencies[i]);
    }
    if (!stats.tileDependencies.empty()) out << "\n  ";
    out << "],\n";
    out << "  \"rockWorldCenters\": [";
    for (size_t i = 0; i < stats.rockWorldCenters.size(); ++i) {
        if (i != 0) out << ",";
        out << "\n    {\"id\": \"" << jsonEscape(stats.rockWorldCenters[i].id)
            << "\", \"position\": ["
            << formatDouble(stats.rockWorldCenters[i].position.x, 6) << ", "
            << formatDouble(stats.rockWorldCenters[i].position.y, 6) << ", "
            << formatDouble(stats.rockWorldCenters[i].position.z, 6) << "]}";
    }
    if (!stats.rockWorldCenters.empty()) out << "\n  ";
    out << "],\n";
    out << "  \"changeLog\": [";
    for (size_t i = 0; i < stats.changeLog.size(); ++i) {
        if (i != 0) out << ",";
        const ChangeLogEntry& entry = stats.changeLog[i];
        out << "\n    {\"index\": " << entry.index << ", \"target\": \""
            << jsonEscape(entry.target) << "\", \"kind\": \"" << entry.kind
            << "\", \"affectedTiles\": ";
        writeIntArray(out, entry.affectedTiles);
        out << ", \"frameHash\": \"" << entry.frameHash << "\"}";
    }
    if (!stats.changeLog.empty()) out << "\n  ";
    out << "],\n";
    out << "  \"shadowVolumeTriangles\": " << stats.shadowVolumeTriangles << ",\n";
    out << "  \"stencilOverflowCount\": " << stats.stencilOverflowCount << ",\n";
    out << "  \"stencilDepthPassTests\": " << stats.stencilDepthPassTests << ",\n";
    out << "  \"stencilDepthFailTests\": " << stats.stencilDepthFailTests << ",\n";
    out << "  \"maxTemplatePassCountAbs\": " << stats.maxTemplatePassCountAbs << ",\n";
    out << "  \"maxTemplateFailCountAbs\": " << stats.maxTemplateFailCountAbs << ",\n";
    out << "  \"nonBackgroundPixels\": " << stats.nonBackgroundPixels << ",\n";
    out << "  \"shadowMaskedBackgroundPixels\": "
        << stats.shadowMaskedBackgroundPixels << ",\n";
    out << "  \"distinctDepthValues\": " << stats.distinctDepthValues << ",\n";
    out << "  \"minNonBackgroundDepth\": " << stats.minNonBackgroundDepth << ",\n";
    out << "  \"maxNonBackgroundDepth\": " << stats.maxNonBackgroundDepth << ",\n";
    out << "  \"incrementalVerification\": {\n";
    out << "    \"divergenceChecked\": " << (stats.divergenceChecked ? "true" : "false") << ",\n";
    out << "    \"framebufferHashesEqual\": "
        << (stats.framebufferHashesEqual ? "true" : "false") << ",\n";
    out << "    \"tileHashesEqual\": " << (stats.tileHashesEqual ? "true" : "false") << ",\n";
    out << "    \"unaffectedTileHashMismatches\": "
        << stats.unaffectedTileHashMismatches << "\n";
    out << "  },\n";
    out << "  \"timingsMs\": {\n";
    out << "    \"loadAndProcedural\": " << stats.loadAndProceduralMs << ",\n";
    out << "    \"rasterization\": " << stats.rasterizationMs << ",\n";
    out << "    \"shadowVolumes\": " << stats.shadowMs << ",\n";
    out << "    \"shading\": " << stats.shadingMs << ",\n";
    out << "    \"cacheBuild\": " << stats.cacheBuildMs << ",\n";
    out << "    \"incrementalUpdates\": " << stats.incrementalUpdateMs << ",\n";
    out << "    \"fullComparison\": " << stats.fullComparisonMs << ",\n";
    out << "    \"total\": " << stats.totalMs << "\n";
    out << "  },\n";
    out << "  \"outputs\": {\n";
    out << "    \"color\": \"color.ppm\",\n";
    out << "    \"depth\": \"depth.ppm\",\n";
    out << "    \"rawDepth\": \"depth_raw.bin\",\n";
    out << "    \"log\": \"render.log\"\n";
    out << "  }\n";
    out << "}\n";
}

void writeRenderLog(const std::string& outputDir, const RenderStats& stats,
                    const std::string& code, const std::string& message) {
    std::ofstream out(outputDir + "/render.log", std::ios::binary);
    if (!out) return;
    out << "status=" << (code.empty() ? "ok" : "error") << "\n";
    if (!code.empty()) out << "error=" << code << " message=" << message << "\n";
    out << "mode=" << stats.mode << "\n";
    out << "depth_bits=" << stats.depthBits << " depth_max=" << stats.depthMax << "\n";
    out << "stencil_bits=" << stats.stencilBits
        << " stencil_overflow=" << stats.stencilOverflowCount << "\n";
    out << "shadow_volume_triangles=" << stats.shadowVolumeTriangles << "\n";
    out << "affected_tiles=";
    writeIntArray(out, stats.affectedTiles);
    out << "\n";
    for (const ChangeLogEntry& entry : stats.changeLog) {
        out << "change[" << entry.index << "] target=" << entry.target
            << " kind=" << entry.kind << " affected=";
        writeIntArray(out, entry.affectedTiles);
        out << " frame_hash=" << entry.frameHash << "\n";
    }
    for (const TileHash& tile : stats.tileHashes) {
        out << "tile[" << tile.tile << "] color_hash=" << tile.hash << "\n";
    }
    out << "frame_hash=" << stats.frameHash << "\n";
    out << "depth_hash=" << stats.depthHash << "\n";
    out << "combined_hash=" << stats.combinedHash << "\n";
    if (stats.divergenceChecked) {
        out << "full_reference_frame_hash=" << stats.fullReferenceFrameHash << "\n";
        out << "full_reference_depth_hash=" << stats.fullReferenceDepthHash << "\n";
        out << "framebuffer_hashes_equal="
            << (stats.framebufferHashesEqual ? "true" : "false") << "\n";
        out << "tile_hashes_equal=" << (stats.tileHashesEqual ? "true" : "false") << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    std::string outputDir = "build/out";
    try {
        options = parseOptions(argc, argv);
        outputDir = options.outputDir;
        std::filesystem::create_directories(outputDir);

        const auto loadStart = std::chrono::steady_clock::now();
        Scene scene = loadScene(options.scenePath);
        const double loadMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - loadStart).count();
        RenderResult result = runRender(std::move(scene), options.mode);
        result.stats.loadAndProceduralMs = loadMs;
        if (result.hasFrame) writeColorAndDepth(result, outputDir);
        writeStatsJson(outputDir, result.stats, result.code, result.message);
        writeRenderLog(outputDir, result.stats, result.code, result.message);

        if (!result.ok) {
            std::cout << "status=error code=" << result.code
                      << " message=" << result.message << "\n";
            return result.exitCode;
        }
        std::cout << "status=ok mode=" << result.stats.mode
                  << " frame_hash=" << result.stats.frameHash
                  << " affected_tiles=" << result.stats.affectedTileCount
                  << " shadow_triangles=" << result.stats.shadowVolumeTriangles << "\n";
        return 0;
    } catch (const RenderError& error) {
        try {
            std::filesystem::create_directories(outputDir);
            RenderStats stats;
            stats.mode = options.mode == RenderMode::Full ? "full" : "incremental";
            stats.status = "error";
            writeStatsJson(outputDir, stats, error.code, error.what());
            writeRenderLog(outputDir, stats, error.code, error.what());
        } catch (...) {
        }
        std::cout << "status=error code=" << error.code
                  << " message=" << error.what() << "\n";
        return error.exitCode;
    } catch (const std::exception& error) {
        try {
            std::filesystem::create_directories(outputDir);
            RenderStats stats;
            stats.mode = options.mode == RenderMode::Full ? "full" : "incremental";
            stats.status = "error";
            writeStatsJson(outputDir, stats, "E_SCENE_INVALID", error.what());
            writeRenderLog(outputDir, stats, "E_SCENE_INVALID", error.what());
        } catch (...) {
        }
        std::cout << "status=error code=E_SCENE_INVALID message=" << error.what() << "\n";
        return 2;
    }
}
