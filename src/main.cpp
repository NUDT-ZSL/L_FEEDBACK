#include "cli.h"
#include "image_io.h"
#include "renderer.h"
#include "scene.h"
#include "snapshot.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using dr::FrameTiming;
using dr::ImageDiff;
using dr::Light;
using dr::LightSet;
using dr::LightType;
using dr::Options;
using dr::Renderer;
using dr::RunMode;
using dr::Scene;
using dr::SceneAssets;
using dr::SnapshotData;
using dr::SnapshotError;

void glfwErrorCallback(int code, const char* description) {
    std::cerr << json{{"level", "error"},
                      {"event", "glfw"},
                      {"code", "ERR_GLFW"},
                      {"glfw_code", code},
                      {"message", description}}
                         .dump()
              << '\n';
}

void emitError(const std::string& code, const std::string& message,
               const json& extra = json::object()) {
    json diagnostic = extra;
    diagnostic["level"] = "error";
    diagnostic["event"] = "runtime";
    diagnostic["code"] = code;
    diagnostic["message"] = message;
    std::cerr << diagnostic.dump() << '\n';
}

void emitSnapshotError(const SnapshotError& error) {
    std::cerr << json{{"level", "error"},
                      {"event", "snapshot_load"},
                      {"code", error.code},
                      {"message", error.message},
                      {"path", error.path},
                      {"format_version", error.formatVersion},
                      {"light_version", error.lightVersion},
                      {"supported_format_version", dr::kSnapshotFormatVersion},
                      {"fallback", "realtime"}}
                     .dump()
              << '\n';
}

class GlfwSession {
public:
    GlfwSession() : initialized_(glfwInit() == GLFW_TRUE) {}
    ~GlfwSession() {
        if (initialized_) {
            glfwTerminate();
        }
    }
    GlfwSession(const GlfwSession&) = delete;
    GlfwSession& operator=(const GlfwSession&) = delete;
    bool valid() const { return initialized_; }

private:
    bool initialized_ = false;
};

struct WindowDeleter {
    void operator()(GLFWwindow* window) const {
        if (window != nullptr) {
            glfwDestroyWindow(window);
        }
    }
};

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    if (values.size() % 2U == 0U) {
        return (values[middle - 1U] + values[middle]) * 0.5;
    }
    return values[middle];
}

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    return sum / static_cast<double>(values.size());
}

bool ensureParentDirectory(const std::string& path, std::string& error) {
    const std::filesystem::path output(path);
    if (!output.has_parent_path()) {
        return true;
    }
    std::error_code filesystemError;
    std::filesystem::create_directories(output.parent_path(), filesystemError);
    if (filesystemError) {
        error = "cannot create output directory: " + filesystemError.message();
        return false;
    }
    return true;
}

bool applyLightModifications(LightSet& lights, const Options& options, std::string& error) {
    for (const std::string& id : options.removeLightIds) {
        const auto found = std::find_if(lights.lights.begin(), lights.lights.end(),
                                        [&id](const Light& light) { return light.id == id; });
        if (found == lights.lights.end()) {
            error = "light not found: " + id;
            return false;
        }
        lights.lights.erase(found);
    }

    std::set<std::string> ids;
    for (const Light& light : lights.lights) {
        ids.insert(light.id);
    }
    for (const std::string& specification : options.addLightSpecs) {
        Light light;
        std::string parseError;
        if (!dr::parseAddLight(specification, light, parseError)) {
            error = parseError;
            return false;
        }
        if (!ids.insert(light.id).second) {
            error = "duplicate light id: " + light.id;
            return false;
        }
        lights.lights.push_back(std::move(light));
    }

    if (!dr::validateLights(lights.lights, error)) {
        return false;
    }
    return true;
}

void printTiming(const FrameTiming& timing) {
    std::cout << json{{"event", "frame"},
                      {"mode", timing.mode},
                      {"frame", timing.frame},
                      {"cpu_ms", timing.cpuMs},
                      {"gpu_ms", timing.gpuMs},
                      {"point_lights", timing.pointLights},
                      {"directional_lights", timing.directionalLights},
                      {"total_lights", timing.pointLights + timing.directionalLights},
                      {"light_version", timing.lightVersion}}
                     .dump()
              << '\n';
}

void printSummary(const std::vector<FrameTiming>& timings, const std::string& mode,
                  std::uint64_t lightVersion) {
    std::vector<double> cpu;
    std::vector<double> gpu;
    cpu.reserve(timings.size());
    gpu.reserve(timings.size());
    for (const FrameTiming& timing : timings) {
        cpu.push_back(timing.cpuMs);
        gpu.push_back(timing.gpuMs);
    }
    const std::size_t pointLights = timings.empty() ? 0U : timings.front().pointLights;
    const std::size_t directionalLights = timings.empty() ? 0U : timings.front().directionalLights;
    std::cout << json{{"event", "summary"},
                      {"mode", mode},
                      {"frames", timings.size()},
                      {"cpu_ms_median", median(cpu)},
                      {"cpu_ms_mean", mean(cpu)},
                      {"gpu_ms_median", median(gpu)},
                      {"gpu_ms_mean", mean(gpu)},
                      {"point_lights", pointLights},
                      {"directional_lights", directionalLights},
                      {"total_lights", pointLights + directionalLights},
                      {"light_version", lightVersion}}
                     .dump()
              << '\n';
}

bool compareWithReference(const Options& options, const std::vector<std::uint8_t>& actual,
                          int actualWidth, int actualHeight) {
    if (options.comparePath.empty()) {
        return true;
    }

    std::vector<std::uint8_t> reference;
    int referenceWidth = 0;
    int referenceHeight = 0;
    std::string error;
    if (!dr::readPng(options.comparePath, referenceWidth, referenceHeight, reference, error)) {
        emitError("ERR_IMAGE_READ", error, {{"path", options.comparePath}});
        return false;
    }

    ImageDiff diff;
    if (!dr::compareImages(actual, actualWidth, actualHeight, reference, referenceWidth,
                           referenceHeight, diff, error)) {
        emitError("ERR_IMAGE_COMPARE", error, {{"path", options.comparePath}});
        return false;
    }

    std::cout << json{{"event", "image_comparison"},
                      {"reference", options.comparePath},
                      {"width", diff.width},
                      {"height", diff.height},
                      {"max_channel_difference", diff.maxChannelDifference},
                      {"different_pixels", diff.differentPixels},
                      {"mean_absolute_difference", diff.meanAbsoluteDifference},
                      {"threshold", options.maxChannelDifference},
                      {"passed", diff.maxChannelDifference <= options.maxChannelDifference}}
                     .dump()
              << '\n';

    if (diff.maxChannelDifference > options.maxChannelDifference) {
        emitError("ERR_IMAGE_MISMATCH", "image difference exceeds threshold",
                  {{"max_channel_difference", diff.maxChannelDifference},
                   {"threshold", options.maxChannelDifference}});
        return false;
    }
    return true;
}

bool writeOutputImage(const std::string& outputPath, const std::vector<std::uint8_t>& pixels) {
    std::string error;
    if (!ensureParentDirectory(outputPath, error)) {
        emitError("ERR_OUTPUT_IO", error, {{"path", outputPath}});
        return false;
    }
    if (!dr::writePng(outputPath, dr::kRenderWidth, dr::kRenderHeight, pixels, error)) {
        emitError("ERR_OUTPUT_IO", "cannot write PNG: " + error, {{"path", outputPath}});
        return false;
    }
    std::cout << json{{"event", "image_written"},
                      {"path", outputPath},
                      {"width", dr::kRenderWidth},
                      {"height", dr::kRenderHeight}}
                     .dump()
              << '\n';
    return true;
}

bool saveSnapshotWithMetadata(const std::string& path, SnapshotData& snapshot,
                              const std::string& sceneId) {
    snapshot.sceneId = sceneId;
    SnapshotError error;
    if (!dr::saveSnapshot(path, snapshot, error)) {
        std::cerr << json{{"level", "error"},
                          {"event", "snapshot_save"},
                          {"code", error.code},
                          {"message", error.message},
                          {"path", error.path},
                          {"format_version", error.formatVersion},
                          {"light_version", error.lightVersion}}
                         .dump()
                  << '\n';
        return false;
    }
    std::size_t pointLights = 0;
    std::size_t directionalLights = 0;
    for (const Light& light : snapshot.lights) {
        if (light.type == LightType::Point) {
            ++pointLights;
        } else {
            ++directionalLights;
        }
    }
    std::cout << json{{"event", "snapshot_saved"},
                      {"path", path},
                      {"format_version", snapshot.formatVersion},
                      {"light_version", snapshot.lightVersion},
                      {"point_lights", pointLights},
                      {"directional_lights", directionalLights},
                      {"width", snapshot.width},
                      {"height", snapshot.height}}
                     .dump()
              << '\n';
    return true;
}

bool renderAndReport(Renderer& renderer, const Options& options, const std::string& mode,
                     const std::string& outputPath, SnapshotData* snapshotToSave) {
    for (int frame = 0; frame < options.warmupFrames; ++frame) {
        renderer.renderFrame(-frame - 1, mode);
    }

    std::vector<FrameTiming> timings;
    timings.reserve(static_cast<std::size_t>(options.frames));
    for (int frame = 1; frame <= options.frames; ++frame) {
        const FrameTiming timing = renderer.renderFrame(frame, mode);
        timings.push_back(timing);
        printTiming(timing);
    }
    printSummary(timings, mode, renderer.lightVersion());

    if (snapshotToSave != nullptr) {
        std::string captureError;
        if (!renderer.captureSnapshot(*snapshotToSave, captureError)) {
            emitError("ERR_SNAPSHOT_CAPTURE", captureError);
            return false;
        }
    }

    std::vector<std::uint8_t> pixels;
    std::string readError;
    if (!renderer.readResultImage(pixels, readError)) {
        emitError("ERR_OUTPUT_READBACK", readError);
        return false;
    }
    if (!writeOutputImage(outputPath, pixels)) {
        return false;
    }

    return compareWithReference(options, pixels, dr::kRenderWidth, dr::kRenderHeight);
}

enum class SnapshotRunResult {
    Success,
    InitializationFailure,
    RuntimeFailure,
};

SnapshotRunResult runSnapshotRestore(const Options& options, const SnapshotData& loaded,
                                     const LightSet& lights, SnapshotData* modifiedSnapshot) {
    {
        Renderer renderer;
        std::string error;
        if (!renderer.initializeSnapshot(loaded, lights, error)) {
            emitError("ERR_RENDERER_INIT", error, {{"mode", "snapshot_restore"}});
            return SnapshotRunResult::InitializationFailure;
        }
        const bool ok = renderAndReport(renderer, options, "snapshot_restore", options.outputPath,
                                        modifiedSnapshot);
        return ok ? SnapshotRunResult::Success : SnapshotRunResult::RuntimeFailure;
    }
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    std::string argumentError;
    bool showHelp = false;
    if (!dr::parseArguments(argc, argv, options, argumentError, showHelp)) {
        std::cerr << json{{"level", "error"},
                          {"event", "arguments"},
                          {"code", "ERR_ARGUMENTS"},
                          {"message", argumentError}}
                         .dump()
                  << '\n';
        dr::printUsage(std::cerr);
        return 2;
    }
    if (showHelp) {
        dr::printUsage(std::cout);
        return 0;
    }

    Scene scene;
    std::string sceneError;
    if (!dr::loadScene(options.scenePath, scene, sceneError)) {
        emitError("ERR_SCENE_PARSE", sceneError, {{"path", options.scenePath}});
        return 1;
    }

    glfwSetErrorCallback(glfwErrorCallback);
    GlfwSession glfwSession;
    if (!glfwSession.valid()) {
        emitError("ERR_GLFW_INIT", "glfwInit failed");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
    std::unique_ptr<GLFWwindow, WindowDeleter> window(
        glfwCreateWindow(dr::kRenderWidth, dr::kRenderHeight, "deferred-snapshot", nullptr, nullptr));
    if (!window) {
        emitError("ERR_GLFW_WINDOW", "cannot create OpenGL window");
        return 1;
    }
    glfwMakeContextCurrent(window.get());
    if (gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)) == 0) {
        emitError("ERR_GLAD", "cannot load OpenGL functions");
        return 1;
    }

    std::cout << json{{"event", "startup"},
                      {"renderer", reinterpret_cast<const char*>(glGetString(GL_RENDERER))},
                      {"version", reinterpret_cast<const char*>(glGetString(GL_VERSION))},
                      {"width", dr::kRenderWidth},
                      {"height", dr::kRenderHeight}}
                     .dump()
              << '\n';

    if (options.mode == RunMode::SnapshotLoad) {
        SnapshotData loaded;
        SnapshotError loadError;
        if (!dr::loadSnapshot(options.snapshotPath, loaded, loadError)) {
            emitSnapshotError(loadError);
        } else {
            LightSet lights{loaded.lights, loaded.lightVersion};
            const bool modified = !options.addLightSpecs.empty() || !options.removeLightIds.empty();
            if (modified) {
                std::string modificationError;
                if (!applyLightModifications(lights, options, modificationError)) {
                    emitError("ERR_LIGHT_UPDATE", modificationError);
                    return 1;
                }
                const std::uint64_t version =
                    dr::nextAvailableLightVersion(loaded.lightVersion + 1, options.snapshotPath);
                lights.version = version;
            }

            const std::string outputSnapshotPath =
                modified ? dr::versionedSnapshotPath(options.snapshotPath, lights.version)
                         : std::string{};
            if (modified) {
                SnapshotData updated;
                updated.formatVersion = dr::kSnapshotFormatVersion;
                // The original G-Buffer is captured from the restored GPU textures.
                const SnapshotRunResult result =
                    runSnapshotRestore(options, loaded, lights, &updated);
                if (result == SnapshotRunResult::Success) {
                    if (!saveSnapshotWithMetadata(outputSnapshotPath, updated, loaded.sceneId)) {
                        return 1;
                    }
                    return 0;
                }
                if (result == SnapshotRunResult::RuntimeFailure) {
                    return 1;
                }
                emitError("ERR_RENDERER_INIT", "snapshot renderer initialization failed; falling back",
                          {{"fallback", "realtime"}, {"light_version", loaded.lightVersion}});
            } else {
                const SnapshotRunResult result =
                    runSnapshotRestore(options, loaded, lights, nullptr);
                if (result == SnapshotRunResult::Success) {
                    return 0;
                }
                if (result == SnapshotRunResult::RuntimeFailure) {
                    return 1;
                }
                emitError("ERR_RENDERER_INIT", "snapshot renderer initialization failed; falling back",
                          {{"fallback", "realtime"}, {"light_version", loaded.lightVersion}});
            }
        }
    }

    const SceneAssets assets = dr::buildSceneAssets(scene);
    LightSet lights = dr::lightSetFromScene(scene);
    std::string snapshotPathToWrite;
    SnapshotData snapshotToSave;
    const bool saveSnapshot = options.mode == RunMode::SnapshotSave;
    if (saveSnapshot) {
        const std::uint64_t version =
            dr::nextAvailableLightVersion(scene.lightVersion, options.snapshotPath);
        lights.version = version;
        snapshotPathToWrite = dr::versionedSnapshotPath(options.snapshotPath, version);
        snapshotToSave.formatVersion = dr::kSnapshotFormatVersion;
    }

    {
        Renderer renderer;
        std::string initializationError;
        if (!renderer.initializeLive(scene, assets, lights, initializationError)) {
            emitError("ERR_RENDERER_INIT", initializationError, {{"mode", "realtime"}});
            return 1;
        }
        const std::string mode =
            options.mode == RunMode::SnapshotLoad ? "realtime_fallback" : "realtime";
        if (!renderAndReport(renderer, options, mode, options.outputPath,
                             saveSnapshot ? &snapshotToSave : nullptr)) {
            return 1;
        }
    }

    if (saveSnapshot) {
        if (!saveSnapshotWithMetadata(snapshotPathToWrite, snapshotToSave, scene.sceneId)) {
            return 1;
        }
    }
    return 0;
}
