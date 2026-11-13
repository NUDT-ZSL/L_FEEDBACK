#include "renderer.hpp"
#include "scene.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

using dungeon::BusinessError;
using dungeon::CameraConfig;
using dungeon::GeometryParams;
using dungeon::RenderReport;
using dungeon::RenderSettings;
using dungeon::Scene;

struct CliOptions {
    int width = 160;
    int height = 112;
    CameraConfig camera;
    GeometryParams geometry;
    double lightShift = 0.0;
    double bias = 0.015;
    int atlasColumns = 2;
    int atlasRows = 2;
    int shadowTileSize = 64;
    std::string cachePath;
    std::string outputPath = "render_summary.json";
};

struct ParseResult {
    bool ok = true;
    bool showHelp = false;
    std::string message;
};

bool parseLong(const std::string& text, long& value) {
    if (text.empty()) {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0') {
        return false;
    }
    value = parsed;
    return true;
}

bool parseInt(const std::string& text, int& value) {
    long parsed = 0;
    if (!parseLong(text, parsed) || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool parseDouble(const std::string& text, double& value) {
    if (text.empty()) {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(text.c_str(), &end);
    if (errno != 0 || end == text.c_str() || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

ParseResult parseArgs(int argc, char** argv, CliOptions& options) {
    auto requireValue = [&](int& index, const std::string& flag, std::string& value) -> bool {
        (void)flag;
        if (index + 1 >= argc) {
            return false;
        }
        value = argv[++index];
        return !value.empty();
    };

    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--help" || arg == "-h") {
            return {true, true, {}};
        }

        std::string value;
        if (arg == "--out") {
            if (!requireValue(index, arg, value)) {
                return {false, false, "missing value for --out"};
            }
            options.outputPath = value;
        } else if (arg == "--cache") {
            if (!requireValue(index, arg, value)) {
                return {false, false, "missing value for --cache"};
            }
            options.cachePath = value;
        } else if (arg == "--width") {
            if (!requireValue(index, arg, value) || !parseInt(value, options.width)) {
                return {false, false, "invalid integer for --width"};
            }
        } else if (arg == "--height") {
            if (!requireValue(index, arg, value) || !parseInt(value, options.height)) {
                return {false, false, "invalid integer for --height"};
            }
        } else if (arg == "--fov") {
            if (!requireValue(index, arg, value) || !parseDouble(value, options.camera.fovYDegrees)) {
                return {false, false, "invalid number for --fov"};
            }
        } else if (arg == "--near") {
            if (!requireValue(index, arg, value) || !parseDouble(value, options.camera.nearZ)) {
                return {false, false, "invalid number for --near"};
            }
        } else if (arg == "--far") {
            if (!requireValue(index, arg, value) || !parseDouble(value, options.camera.farZ)) {
                return {false, false, "invalid number for --far"};
            }
        } else if (arg == "--camera-x") {
            if (!requireValue(index, arg, value) || !parseDouble(value, options.camera.position.x)) {
                return {false, false, "invalid number for --camera-x"};
            }
        } else if (arg == "--camera-y") {
            if (!requireValue(index, arg, value) || !parseDouble(value, options.camera.position.y)) {
                return {false, false, "invalid number for --camera-y"};
            }
        } else if (arg == "--camera-z") {
            if (!requireValue(index, arg, value) || !parseDouble(value, options.camera.position.z)) {
                return {false, false, "invalid number for --camera-z"};
            }
        } else if (arg == "--yaw") {
            if (!requireValue(index, arg, value) || !parseDouble(value, options.camera.yawDegrees)) {
                return {false, false, "invalid number for --yaw"};
            }
        } else if (arg == "--pitch") {
            if (!requireValue(index, arg, value) || !parseDouble(value, options.camera.pitchDegrees)) {
                return {false, false, "invalid number for --pitch"};
            }
        } else if (arg == "--geometry-version") {
            if (!requireValue(index, arg, value) || !parseInt(value, options.geometry.version)) {
                return {false, false, "invalid integer for --geometry-version"};
            }
        } else if (arg == "--geometry-scale") {
            if (!requireValue(index, arg, value) || !parseDouble(value, options.geometry.scale)) {
                return {false, false, "invalid number for --geometry-scale"};
            }
        } else if (arg == "--light-shift") {
            if (!requireValue(index, arg, value) || !parseDouble(value, options.lightShift)) {
                return {false, false, "invalid number for --light-shift"};
            }
        } else if (arg == "--bias") {
            if (!requireValue(index, arg, value) || !parseDouble(value, options.bias)) {
                return {false, false, "invalid number for --bias"};
            }
        } else if (arg == "--atlas-cols") {
            if (!requireValue(index, arg, value) || !parseInt(value, options.atlasColumns)) {
                return {false, false, "invalid integer for --atlas-cols"};
            }
        } else if (arg == "--atlas-rows") {
            if (!requireValue(index, arg, value) || !parseInt(value, options.atlasRows)) {
                return {false, false, "invalid integer for --atlas-rows"};
            }
        } else if (arg == "--shadow-tile-size") {
            if (!requireValue(index, arg, value) || !parseInt(value, options.shadowTileSize)) {
                return {false, false, "invalid integer for --shadow-tile-size"};
            }
        } else {
            return {false, false, "unknown argument: " + arg};
        }
    }
    return {};
}

void printHelp() {
    std::cout
        << "dungeon_deferred - deterministic offline deferred shadow validator\n\n"
        << "Usage:\n"
        << "  dungeon_deferred [options]\n\n"
        << "Options:\n"
        << "  --out PATH                 Summary JSON path (default render_summary.json)\n"
        << "  --width N --height N       Positive screen resolution; edge tiles are clipped\n"
        << "  --fov DEG                  Vertical field of view in (0, 179)\n"
        << "  --near Z --far Z           0 < near < far\n"
        << "  --camera-x/y/z VALUE       Camera position\n"
        << "  --yaw DEG --pitch DEG      Camera orientation\n"
        << "  --geometry-version N       Non-negative deterministic geometry variant\n"
        << "  --geometry-scale S         Geometry scale in (0, 8]\n"
        << "  --light-shift S            Moves light id 2 to exercise selective tile rebuild\n"
        << "  --bias VALUE               Shadow compare bias in [0, 0.25]\n"
        << "  --atlas-cols N --atlas-rows N\n"
        << "                             Shadow atlas grid (preallocated fixed tiles)\n"
        << "  --shadow-tile-size N       Per-light atlas tile edge (8..256)\n"
        << "  --cache PATH               Persistent shadow tile cache for cross-run reuse\n"
        << "  --help                     Show this help\n\n"
        << "Exit codes: 0 success, 2 invalid camera/geometry/arguments, 3 E_BIAS_RANGE,\n"
        << "            4 output/cache write failure.\n";
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch)) << std::dec;
                } else {
                    out << ch;
                }
                break;
        }
    }
    return out.str();
}

std::string formatNumber(double value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(12) << value;
    return out.str();
}

std::string formatFixed(double value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(6) << value;
    return out.str();
}

void writeVec3Json(std::ostringstream& out, const dungeon::Vec3& value) {
    out << "[" << formatNumber(value.x) << ", " << formatNumber(value.y)
        << ", " << formatNumber(value.z) << "]";
}

std::vector<std::string> uniqueErrorCodes(const std::vector<BusinessError>& errors) {
    std::vector<std::string> codes;
    for (const BusinessError& error : errors) {
        if (std::find(codes.begin(), codes.end(), error.code) == codes.end()) {
            codes.push_back(error.code);
        }
    }
    return codes;
}

std::string buildSummaryJson(
    const CliOptions& options,
    const RenderReport& report,
    const std::string& status) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    int tileColumns = report.tileColumns;
    int tileRows = report.tileRows;
    if (tileColumns == 0 && options.width > 0) {
        tileColumns = (options.width + 15) / 16;
    }
    if (tileRows == 0 && options.height > 0) {
        tileRows = (options.height + 15) / 16;
    }

    out << "{\n";
    out << "  \"schemaVersion\": 1,\n";
    out << "  \"status\": \"" << jsonEscape(status) << "\",\n";
    out << "  \"mode\": \"offline-deferred-shadow-validation\",\n";
    out << "  \"config\": {\n";
    out << "    \"width\": " << options.width << ",\n";
    out << "    \"height\": " << options.height << ",\n";
    out << "    \"lightTileSize\": 16,\n";
    out << "    \"tileColumns\": " << tileColumns << ",\n";
    out << "    \"tileRows\": " << tileRows << ",\n";
    out << "    \"camera\": {\n";
    out << "      \"position\": ";
    writeVec3Json(out, options.camera.position);
    out << ",\n";
    out << "      \"yawDegrees\": " << formatNumber(options.camera.yawDegrees) << ",\n";
    out << "      \"pitchDegrees\": " << formatNumber(options.camera.pitchDegrees) << ",\n";
    out << "      \"fovYDegrees\": " << formatNumber(options.camera.fovYDegrees) << ",\n";
    out << "      \"nearZ\": " << formatNumber(options.camera.nearZ) << ",\n";
    out << "      \"farZ\": " << formatNumber(options.camera.farZ) << "\n";
    out << "    },\n";
    out << "    \"geometry\": {\n";
    out << "      \"version\": " << options.geometry.version << ",\n";
    out << "      \"scale\": " << formatNumber(options.geometry.scale) << ",\n";
    out << "      \"hash\": \"" << dungeon::hex64(report.sceneHash) << "\"\n";
    out << "    },\n";
    out << "    \"lightShift\": " << formatNumber(options.lightShift) << ",\n";
    out << "    \"shadowBias\": " << formatNumber(options.bias) << ",\n";
    out << "    \"atlas\": {\n";
    out << "      \"columns\": " << options.atlasColumns << ",\n";
    out << "      \"rows\": " << options.atlasRows << ",\n";
    out << "      \"tileSize\": " << options.shadowTileSize << ",\n";
    out << "      \"capacity\": " << options.atlasColumns * options.atlasRows << "\n";
    out << "    },\n";
    out << "    \"cache\": {\n";
    out << "      \"requested\": " << (report.cacheRequested ? "true" : "false") << ",\n";
    out << "      \"loaded\": " << (report.cacheLoaded ? "true" : "false") << ",\n";
    out << "      \"saved\": " << (report.cacheSaved ? "true" : "false") << "\n";
    out << "    }\n";
    out << "  },\n";

    const std::vector<std::string> errorCodes = uniqueErrorCodes(report.businessErrors);
    out << "  \"errorCodes\": [";
    for (std::size_t index = 0; index < errorCodes.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        out << "\"" << jsonEscape(errorCodes[index]) << "\"";
    }
    out << "],\n";

    out << "  \"businessErrors\": [";
    if (!report.businessErrors.empty()) {
        out << "\n";
    }
    for (std::size_t index = 0; index < report.businessErrors.size(); ++index) {
        const BusinessError& error = report.businessErrors[index];
        out << "    {\"code\": \"" << jsonEscape(error.code)
            << "\", \"message\": \"" << jsonEscape(error.message) << "\"";
        if (error.lightId >= 0) {
            out << ", \"lightId\": " << error.lightId;
        }
        out << "}";
        if (index + 1 != report.businessErrors.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"summary\": {\n";
    out << "    \"tileHits\": " << report.tileHits << ",\n";
    out << "    \"tileRebuilds\": " << report.tileRebuilds << ",\n";
    out << "    \"atlasAllocated\": " << report.atlasAllocated << ",\n";
    out << "    \"degradedLights\": " << report.degradedLights << ",\n";
    out << "    \"ambientOnlyTiles\": " << report.ambientOnlyTiles << ",\n";
    out << "    \"shadowComparisons\": " << report.shadowComparisons << ",\n";
    out << "    \"pixelLuminanceChecksum\": \"" << dungeon::hex64(report.pixelLuminanceChecksum)
        << "\"\n";
    out << "  },\n";

    out << "  \"gbuffer\": {\n";
    out << "    \"resolution\": [" << options.width << ", " << options.height << "],\n";
    out << "    \"coveredPixels\": " << report.gbufferCoveredPixels << ",\n";
    out << "    \"materialPixelCounts\": {";
    const std::size_t materialCount =
        std::min<std::size_t>(report.materialPixelCounts.size(), 5);
    for (std::size_t material = 1; material < materialCount; ++material) {
        if (material != 1) {
            out << ", ";
        }
        out << "\"" << material << "\": " << report.materialPixelCounts[material];
    }
    out << "}\n";
    out << "  },\n";

    out << "  \"tiles\": [";
    if (!report.tiles.empty()) {
        out << "\n";
    }
    for (std::size_t index = 0; index < report.tiles.size(); ++index) {
        const auto& tile = report.tiles[index];
        out << "    {\"x\": " << tile.x << ", \"y\": " << tile.y
            << ", \"width\": " << tile.width << ", \"height\": " << tile.height
            << ", \"lightCount\": " << tile.lightIds.size() << ", \"lightIds\": [";
        for (std::size_t lightIndex = 0; lightIndex < tile.lightIds.size(); ++lightIndex) {
            if (lightIndex != 0) {
                out << ", ";
            }
            out << tile.lightIds[lightIndex];
        }
        out << "], \"luminanceChecksum\": \"" << dungeon::hex64(tile.luminanceChecksum)
            << "\", \"averageLuminance\": " << formatFixed(tile.averageLuminance) << "}";
        if (index + 1 != report.tiles.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"lights\": [";
    if (!report.lights.empty()) {
        out << "\n";
    }
    for (std::size_t index = 0; index < report.lights.size(); ++index) {
        const auto& light = report.lights[index];
        const bool reused = light.action == "hit";
        const bool rebuilt = light.action == "rebuild";
        out << "    {\"id\": " << light.id << ", \"type\": \"" << light.type
            << "\", \"castsShadow\": " << (light.castsShadow ? "true" : "false")
            << ", \"tileIndex\": " << light.tileIndex
            << ", \"atlasX\": " << light.atlasX << ", \"atlasY\": " << light.atlasY
            << ", \"action\": \"" << light.action << "\""
            << ", \"reused\": " << (reused ? "true" : "false")
            << ", \"rebuilt\": " << (rebuilt ? "true" : "false")
            << ", \"transformHash\": \"" << dungeon::hex64(light.transformHash) << "\""
            << ", \"shadowTapAttempts\": " << light.shadowTapAttempts
            << ", \"shadowTapSuccesses\": " << light.shadowTapSuccesses
            << ", \"shadowTapShadowed\": " << light.shadowTapShadowed
            << ", \"shadowSampleSuccessRate\": " << formatFixed(light.sampleSuccessRate)
            << ", \"shadowedRate\": " << formatFixed(light.shadowedRate)
            << ", \"litPixels\": " << light.litPixels << "}";
        if (index + 1 != report.lights.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    return out.str();
}

bool writeTextAtomic(const std::string& path, const std::string& content) {
    try {
        const std::filesystem::path destination(path);
        if (destination.has_parent_path()) {
            std::filesystem::create_directories(destination.parent_path());
        }
        const std::filesystem::path temp = destination.string() + ".tmp";
        {
            std::ofstream out(temp, std::ios::binary | std::ios::trunc);
            if (!out) {
                return false;
            }
            out.write(content.data(), static_cast<std::streamsize>(content.size()));
            out.flush();
            if (!out) {
                out.close();
                std::filesystem::remove(temp);
                return false;
            }
        }
#ifdef _WIN32
        if (MoveFileExW(
                temp.wstring().c_str(),
                destination.wstring().c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
            std::filesystem::remove(temp);
            return false;
        }
#else
        std::error_code ec;
        std::filesystem::rename(temp, destination, ec);
        if (ec) {
            std::filesystem::remove(temp);
            return false;
        }
#endif
        return true;
    } catch (const std::filesystem::filesystem_error&) {
        return false;
    }
}

int writeErrorSummary(
    const CliOptions& options,
    const std::string& code,
    const std::string& message,
    int exitCode) {
    RenderReport report;
    report.width = options.width;
    report.height = options.height;
    report.tileColumns = options.width > 0 ? (options.width + 15) / 16 : 0;
    report.tileRows = options.height > 0 ? (options.height + 15) / 16 : 0;
    report.atlasColumns = options.atlasColumns;
    report.atlasRows = options.atlasRows;
    report.shadowTileSize = options.shadowTileSize;
    report.cacheRequested = !options.cachePath.empty();
    report.businessErrors.push_back({code, message, -1});
    const std::string json = buildSummaryJson(options, report, "error");
    if (!writeTextAtomic(options.outputPath, json)) {
        std::cerr << "failed to write error summary to " << options.outputPath << "\n";
        return 4;
    }
    return exitCode;
}

bool validAtlasArguments(const CliOptions& options) {
    if (options.atlasColumns < 1 || options.atlasColumns > 64 ||
        options.atlasRows < 1 || options.atlasRows > 64 ||
        options.shadowTileSize < 8 || options.shadowTileSize > 256) {
        return false;
    }
    const int64_t capacity =
        static_cast<int64_t>(options.atlasColumns) * options.atlasRows;
    const int64_t texelsPerTile =
        static_cast<int64_t>(options.shadowTileSize) * options.shadowTileSize;
    return capacity * texelsPerTile <= 16LL * 1024LL * 1024LL;
}

}  // namespace

int main(int argc, char** argv) {
    CliOptions options;
    const ParseResult parseResult = parseArgs(argc, argv, options);
    if (parseResult.showHelp) {
        printHelp();
        return 0;
    }
    if (!parseResult.ok) {
        return writeErrorSummary(options, "E_ARGS_INVALID", parseResult.message, 2);
    }

    dungeon::ValidationIssue issue;
    if (!dungeon::validateCamera(options.camera, options.width, options.height, issue)) {
        return writeErrorSummary(options, issue.code, issue.message, 2);
    }
    if (!dungeon::validateGeometry(options.geometry, issue)) {
        return writeErrorSummary(options, issue.code, issue.message, 2);
    }
    if (!std::isfinite(options.lightShift) || options.lightShift < -4.0 || options.lightShift > 4.0) {
        return writeErrorSummary(
            options,
            "E_GEOMETRY_INVALID",
            "light shift must be finite and in [-4, 4]",
            2);
    }
    if (!validAtlasArguments(options)) {
        return writeErrorSummary(
            options,
            "E_ATLAS_CONFIG_INVALID",
            "atlas requires 1..64 columns/rows, tile size 8..256, and <=16M atlas texels",
            2);
    }
    if (!std::isfinite(options.bias) || options.bias < 0.0 || options.bias > 0.25) {
        return writeErrorSummary(
            options,
            "E_BIAS_RANGE",
            "shadow bias must be in [0, 0.25] world units",
            3);
    }

    const dungeon::Camera camera = dungeon::makeCamera(options.camera, options.width, options.height);
    const Scene scene = dungeon::buildDungeonScene(options.geometry, options.lightShift);

    RenderSettings settings;
    settings.width = options.width;
    settings.height = options.height;
    settings.camera = camera;
    settings.geometryVersion = options.geometry.version;
    settings.shadowBias = options.bias;
    settings.atlasColumns = options.atlasColumns;
    settings.atlasRows = options.atlasRows;
    settings.shadowTileSize = options.shadowTileSize;
    settings.cachePath = options.cachePath;

    const RenderReport report = dungeon::renderDeferred(scene, settings);
    const std::string json = buildSummaryJson(options, report, "ok");
    if (!writeTextAtomic(options.outputPath, json)) {
        std::cerr << "failed to write summary to " << options.outputPath << "\n";
        return 4;
    }

    std::cout << "rendered " << options.width << "x" << options.height
              << ", tiles=" << report.tiles.size()
              << ", light-tile hits=" << report.tileHits
              << ", rebuilds=" << report.tileRebuilds
              << ", checksum=" << dungeon::hex64(report.pixelLuminanceChecksum) << "\n";
    return 0;
}
