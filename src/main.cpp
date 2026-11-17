#include "error.h"
#include "json.h"
#include "renderer.h"
#include "sample_scene.h"
#include "scene.h"
#include "sha256.h"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using softraster::ErrorCode;
using softraster::JsonValue;
using softraster::RenderError;

void reportError(const RenderError& error) {
    std::cerr << softraster::formatErrorJson(error);
}

int fail(const RenderError& error) {
    reportError(error);
    return static_cast<int>(error.code);
}

bool writeBytes(const std::string& path, const std::vector<std::uint8_t>& bytes,
                RenderError& error) {
    const std::string temporary = path + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            error = softraster::makeError(ErrorCode::OutputWrite,
                                          "unable to create temporary output file", path);
            return false;
        }
        if (!bytes.empty()) {
            output.write(reinterpret_cast<const char*>(bytes.data()),
                         static_cast<std::streamsize>(bytes.size()));
        }
        output.flush();
        if (!output) {
            output.close();
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            error = softraster::makeError(ErrorCode::OutputWrite,
                                          "unable to write output file", path);
            return false;
        }
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::rename(temporary, path, ignored);
    if (ignored) {
        std::filesystem::remove(temporary, ignored);
        error = softraster::makeError(
            ErrorCode::OutputWrite,
            "unable to atomically replace the output file: " + ignored.message(), path);
        return false;
    }
    return true;
}

bool writeText(const std::string& path, std::string_view text, RenderError& error) {
    const std::vector<std::uint8_t> bytes(
        reinterpret_cast<const std::uint8_t*>(text.data()),
        reinterpret_cast<const std::uint8_t*>(text.data()) + text.size());
    return writeBytes(path, bytes, error);
}

JsonValue number(double value) { return JsonValue(value); }

JsonValue vec3(int x, int y, int width, int height) {
    JsonValue::Object object;
    object.emplace("height", number(height));
    object.emplace("width", number(width));
    object.emplace("x", number(x));
    object.emplace("y", number(y));
    return JsonValue(std::move(object));
}

std::string makeStatsJson(const softraster::RenderedFrame& frame,
                          const std::string& outputSha256) {
    JsonValue::Object stats;
    stats.emplace("alphaBlendedFragments",
                  number(static_cast<double>(frame.stats.alphaBlendedFragments)));
    stats.emplace("clippedTriangles",
                  number(static_cast<double>(frame.stats.clippedTriangles)));
    stats.emplace("height", number(frame.height));
    stats.emplace("opaqueFragments",
                  number(static_cast<double>(frame.stats.opaqueFragments)));
    stats.emplace("outputImageSha256", JsonValue(outputSha256));
    stats.emplace("peakMemoryEstimatedBytes",
                  number(static_cast<double>(frame.stats.peakMemoryEstimatedBytes)));
    stats.emplace("ppmBytes", number(static_cast<double>(frame.stats.ppmBytes)));
    stats.emplace("rasterizedTriangles",
                  number(static_cast<double>(frame.stats.rasterizedTriangles)));
    stats.emplace("shadowEnabled", JsonValue(frame.stats.shadowEnabled));
    stats.emplace("shadowMapSize", number(frame.stats.shadowMapSize));
    stats.emplace("shadowSampleCount",
                  number(static_cast<double>(frame.stats.shadowSampleCount)));
    stats.emplace("transparentDepthWrites",
                  number(static_cast<double>(frame.stats.transparentDepthWrites)));
    stats.emplace("transparentFragments",
                  number(static_cast<double>(frame.stats.transparentFragments)));
    stats.emplace("transparentSortSwaps",
                  number(static_cast<double>(frame.stats.transparentSortSwaps)));
    stats.emplace("transparentTriangles",
                  number(static_cast<double>(frame.stats.transparentTriangles)));
    stats.emplace("viewport",
                  vec3(frame.stats.viewportX, frame.stats.viewportY,
                       frame.stats.viewportWidth, frame.stats.viewportHeight));
    stats.emplace("width", number(frame.width));
    return softraster::writeJson(JsonValue(std::move(stats)), true);
}

int runGenerateSample(const std::string& path) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        return fail(softraster::makeError(ErrorCode::OutputWrite,
                                          "unable to create sample scene file", path));
    }
    const std::string json = softraster::buildSampleSceneJson();
    output.write(json.data(), static_cast<std::streamsize>(json.size()));
    output.flush();
    if (!output) {
        return fail(softraster::makeError(ErrorCode::OutputWrite,
                                          "unable to write sample scene file", path));
    }
    return 0;
}

int runRender(const std::string& scenePath, const std::string& outputPath,
              const std::string& statsPath) {
    softraster::SceneLoadResult loaded = softraster::loadScene(scenePath);
    if (!loaded.scene.has_value()) {
        return fail(loaded.error.value_or(
            softraster::makeError(ErrorCode::Io, "unknown scene loading failure", scenePath)));
    }
    softraster::RenderResult rendered = softraster::renderScene(*loaded.scene);
    if (!rendered.frame.has_value()) {
        return fail(rendered.error.value_or(
            softraster::makeError(ErrorCode::InvalidValue, "unknown rendering failure",
                                  scenePath)));
    }

    const std::string hash = softraster::sha256Hex(rendered.frame->ppm);
    const std::string statsJson = makeStatsJson(*rendered.frame, hash);
    RenderError writeError;
    if (!writeBytes(outputPath, rendered.frame->ppm, writeError)) {
        return fail(writeError);
    }
    if (!writeText(statsPath, statsJson, writeError)) {
        return fail(writeError);
    }
    return 0;
}

void printUsage() {
    std::cout
        << "Offline deterministic software rasterizer\n"
        << "Usage:\n"
        << "  softraster <scene.json> <output.ppm> [stats.json]\n"
        << "  softraster --generate-sample <scene.json>\n"
        << "If stats.json is omitted, <output.ppm>.stats.json is used.\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 &&
            (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
            printUsage();
            return 0;
        }
        if (argc == 3 && std::string_view(argv[1]) == "--generate-sample") {
            return runGenerateSample(argv[2]);
        }
        if (argc != 3 && argc != 4) {
            return fail(softraster::makeError(
                ErrorCode::InvalidValue,
                "expected <scene.json> <output.ppm> [stats.json] or "
                "--generate-sample <scene.json>; use --help for usage",
                "argv"));
        }
        const std::string statsPath =
            argc == 4 ? std::string(argv[3]) : std::string(argv[2]) + ".stats.json";
        return runRender(argv[1], argv[2], statsPath);
    } catch (const std::bad_alloc&) {
        return fail(softraster::makeError(ErrorCode::MemoryLimitExceeded,
                                          "allocation failed during rendering", "$"));
    } catch (const std::exception& exception) {
        return fail(softraster::makeError(
            ErrorCode::Io, std::string("unhandled exception: ") + exception.what(), "$"));
    }
}
