#include "config.h"
#include "density.h"
#include "errors.h"
#include "image.h"
#include "json.h"
#include "log.h"
#include "renderer.h"
#include "snapshot.h"
#include "thread_pool.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <vector>

namespace cloud {
namespace {

std::string frameFileBase(const AppConfig& config, std::uint32_t frameIndex) {
    std::ostringstream name;
    name << config.outputDirectory << '/' << config.outputName << '_' << std::setw(4)
         << std::setfill('0') << frameIndex;
    return name.str();
}

void ensureWritableDirectory(const std::string& directory) {
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        throw BusinessError("E_OUTPUT", kExitOutput, "cannot create output directory",
                            json::Value::Object{{"path", directory}});
    }
    const std::string probePath = directory + "/.write_probe";
    {
        std::ofstream probe(probePath, std::ios::binary | std::ios::trunc);
        if (!probe) {
            throw BusinessError("E_OUTPUT", kExitOutput, "output directory is not writable",
                                json::Value::Object{{"path", directory}});
        }
        probe << "ok";
        probe.flush();
        if (!probe) {
            throw BusinessError("E_OUTPUT", kExitOutput, "output directory write probe failed",
                                json::Value::Object{{"path", directory}});
        }
    }
    std::filesystem::remove(probePath, error);
}

SnapshotData makeSnapshotData(const RenderState& state, const AppConfig& config,
                              const DensityField& density, const Light& nextLight) {
    SnapshotData data;
    data.seed = density.seed();
    data.cloudType = static_cast<std::uint32_t>(density.type());
    data.densityResolution = static_cast<std::uint32_t>(density.resolution());
    data.frameIndex = state.frameIndex;
    data.width = static_cast<std::uint32_t>(config.width);
    data.height = static_cast<std::uint32_t>(config.height);
    data.historyLength = static_cast<std::uint32_t>(config.maxHistoryLength);
    data.hasLight = true;
    data.nextLight = nextLight;
    data.view = state.previousView;
    data.projection = state.previousProjection;
    data.accumulatedColor = state.accumulatedColor;
    data.depth = state.depth;
    data.counts = state.counts;
    return data;
}

json::Value statsToJson(const FrameStats& stats, double psnrDb) {
    json::Value::Object object;
    object["frameIndex"] = stats.frameIndex;
    object["reprojectionAttempts"] = stats.reprojectionAttempts;
    object["reprojectionHits"] = stats.reprojectionHits;
    object["reprojectionMisses"] = stats.reprojectionMisses;
    object["reprojectionHitRate"] = stats.reprojectionHitRate;
    object["minimumCount"] = stats.minimumCount;
    object["maximumCount"] = stats.maximumCount;
    object["countsStrictlyMonotonic"] = stats.countsStrictlyMonotonic;
    object["upsamplePsnrDb"] = std::isfinite(psnrDb) ? psnrDb : 999.0;
    return json::Value(std::move(object));
}

int run(const AppConfig& inputConfig, Logger& logger, WorkerPool* pool) {
    AppConfig config = inputConfig;
    bool loadedSnapshot = false;
    SnapshotData snapshot;

    if (!config.snapshot.loadPath.empty()) {
        snapshot = loadSnapshot(config.snapshot.loadPath);
        loadedSnapshot = true;
        config.width = static_cast<int>(snapshot.width);
        config.height = static_cast<int>(snapshot.height);
        config.seed = snapshot.seed;
        config.cloudType = static_cast<CloudType>(snapshot.cloudType);
        config.maxHistoryLength = static_cast<int>(snapshot.historyLength);
        if (snapshot.version == 1) {
            logger.warning(
                "W_SNAPSHOT_UPGRADED",
                "v1 snapshot upgraded to the v2 in-memory layout using documented defaults",
                json::Value::Object{{"sourceVersion", 1},
                                    {"targetVersion", 2},
                                    {"defaultHistoryLength", snapshot.historyLength},
                                    {"defaultSunDirection", json::Value::Array{0.45, 0.82, 0.25}},
                                    {"defaultAmbient", 0.16}});
        }
        logger.info("snapshot_loaded",
                    "render state restored; continuation starts at the next frame",
                    json::Value::Object{{"path", config.snapshot.loadPath},
                                        {"version", snapshot.version},
                                        {"frameIndex", snapshot.frameIndex},
                                        {"width", snapshot.width},
                                        {"height", snapshot.height},
                                        {"densityResolution", snapshot.densityResolution},
                                        {"historyLength", snapshot.historyLength},
                                        {"upgradedFromV1", snapshot.upgradedFromV1}});
    }

    std::uint32_t densityResolution = snapshot.densityResolution;
    std::string resolutionReason;
    if (!loadedSnapshot) {
        try {
            densityResolution = static_cast<std::uint32_t>(
                chooseDensityResolution(config.memory.availableMB, &resolutionReason));
        } catch (const std::exception&) {
            throw BusinessError(
                "E_OUT_OF_MEMORY", kExitOutOfMemory,
                "available memory is insufficient even for the 32^3 density level",
                json::Value::Object{{"availableMB", config.memory.availableMB},
                                    {"minimumRequiredMB", 32}});
        }
    } else {
        const int requiredMB = densityResolutionRequiredMB(static_cast<int>(densityResolution));
        if (config.memory.availableMB < static_cast<double>(requiredMB)) {
            throw BusinessError(
                "E_OUT_OF_MEMORY", kExitOutOfMemory,
                "available memory cannot hold the snapshot density level",
                json::Value::Object{{"availableMB", config.memory.availableMB},
                                    {"snapshotDensityResolution", densityResolution},
                                    {"requiredMB", requiredMB}});
        }
        resolutionReason = "snapshot density level retained for deterministic continuation";
    }

    if (densityResolution != 128) {
        logger.warning("W_DENSITY_DOWNSAMPLED", resolutionReason,
                       json::Value::Object{{"densityResolution", densityResolution},
                                           {"fullResolution", 128},
                                           {"availableMB", config.memory.availableMB},
                                           {"reason", resolutionReason}});
    } else {
        logger.info("density_budget", resolutionReason,
                    json::Value::Object{{"densityResolution", densityResolution},
                                        {"availableMB", config.memory.availableMB}});
    }

    const auto densityStart = std::chrono::steady_clock::now();
    DensityField density =
        DensityField::build(config.seed, config.cloudType, static_cast<int>(densityResolution), pool,
                            config.asynchronous && pool != nullptr);
    const auto densityEnd = std::chrono::steady_clock::now();
    const double densityMilliseconds =
        std::chrono::duration<double, std::milli>(densityEnd - densityStart).count();
    logger.info("density_ready", "deterministic density field generated",
                json::Value::Object{{"seed", density.seed()},
                                    {"cloudType", cloudTypeName(density.type())},
                                    {"resolution", density.resolution()},
                                    {"bytes", density.data().size()},
                                    {"minimumDensity", density.minimumDensity()},
                                    {"maximumDensity", density.maximumDensity()},
                                    {"meanDensity", density.meanDensity()},
                                    {"async", config.asynchronous && pool != nullptr},
                                    {"workers", pool == nullptr ? 0 : pool->size()},
                                    {"mainThreadSpinMilliseconds", 0.0},
                                    {"waitEvent", "condition_variable"},
                                    {"elapsedMilliseconds", densityMilliseconds}});

    RenderSettings settings = config.renderSettings();
    Renderer renderer(settings, density);
    RenderState state;
    state.configure(config.width, config.height);

    Light restoredLight = snapshot.nextLight;
    bool useRestoredLight = false;
    std::uint32_t currentFrame =
        loadedSnapshot ? snapshot.frameIndex + 1u : static_cast<std::uint32_t>(config.startFrame);

    if (loadedSnapshot) {
        if (snapshot.width != static_cast<std::uint32_t>(config.width) ||
            snapshot.height != static_cast<std::uint32_t>(config.height)) {
            throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                                "snapshot dimensions changed while restoring");
        }
        const std::size_t pixelCount =
            static_cast<std::size_t>(config.width) * static_cast<std::size_t>(config.height);
        if (snapshot.accumulatedColor.size() != pixelCount || snapshot.depth.size() != pixelCount ||
            snapshot.counts.size() != pixelCount) {
            throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                                "snapshot buffers do not match restored dimensions");
        }
        state.hasHistory = true;
        state.frameIndex = snapshot.frameIndex;
        state.previousView = snapshot.view;
        state.previousProjection = snapshot.projection;
        try {
            state.previousCameraPosition =
                snapshot.view.inverse().transformPoint(Vec3{0.0f, 0.0f, 0.0f});
        } catch (const std::exception& error) {
            throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                                std::string("snapshot camera matrix is singular: ") + error.what());
        }
        if (!finite(state.previousCameraPosition)) {
            throw BusinessError("E_SNAPSHOT_FORMAT", kExitSnapshotFormat,
                                "snapshot camera matrix is not finite");
        }
        state.accumulatedColor = snapshot.accumulatedColor;
        state.depth = snapshot.depth;
        state.counts = snapshot.counts;
        useRestoredLight = true;
    }

    logger.info("render_start", "starting deterministic CPU volume rendering",
                json::Value::Object{{"startFrame", currentFrame},
                                    {"endFrameExclusive", config.endFrame},
                                    {"width", config.width},
                                    {"height", config.height},
                                    {"renderScale", config.renderScale},
                                    {"maxSteps", config.maxSteps},
                                    {"lightSteps", config.lightSteps},
                                    {"snapshotVersion", config.snapshot.version}});

    json::Value::Array frameStatsArray;
    std::uint32_t lastFrame = currentFrame;
    double minimumPsnr = std::numeric_limits<double>::infinity();
    bool stopRequested = false;
    std::uint64_t renderedFrames = 0;

    for (std::uint64_t frame64 = currentFrame; frame64 < config.endFrame; ++frame64) {
        const std::uint32_t frame = static_cast<std::uint32_t>(frame64);
        const Camera camera = config.camera.cameraAt(frame, config.width, config.height);
        Light light = config.sun.lightAt(frame);
        if (useRestoredLight) {
            light = restoredLight;
            useRestoredLight = false;
        }

        SampleFrame sample = renderer.renderSample(camera, light, config.renderScale);
        double psnrDb = std::numeric_limits<double>::infinity();
        if (config.quality.compareAgainstFullResolution) {
            const SampleFrame fullReference = renderer.renderSample(camera, light, 1);
            const std::vector<std::uint8_t> upsampledRgba = renderer.encodeRGBA(sample.color);
            const std::vector<std::uint8_t> fullRgba = renderer.encodeRGBA(fullReference.color);
            psnrDb = Renderer::psnrRGB8(upsampledRgba, fullRgba);
            minimumPsnr = std::min(minimumPsnr, psnrDb);
            if (config.quality.writeFullResolutionReference) {
                writeRGBA8(frameFileBase(config, frame) + "_full.rgba", fullRgba);
            }
            logger.info("upsample_quality", "half-resolution depth-aware upsample metric",
                        json::Value::Object{{"frameIndex", frame},
                                            {"psnrDb", std::isfinite(psnrDb) ? psnrDb : 999.0},
                                            {"thresholdDb", 30.0}});
        }

        const FrameStats stats = renderer.accumulate(state, sample, camera, frame);
        if (stats.reprojectionHits + stats.reprojectionMisses != stats.reprojectionAttempts) {
            throw BusinessError("E_INTERNAL", kExitInternal,
                                "reprojection statistics are internally inconsistent");
        }
        if (!stats.countsStrictlyMonotonic) {
            throw BusinessError("E_INTERNAL", kExitInternal,
                                "per-pixel accumulation counts are not strictly monotonic");
        }

        const std::vector<std::uint8_t> rgba = renderer.encodeRGBA(state.accumulatedColor);
        const std::string base = frameFileBase(config, frame);
        writeRGBA8(base + ".rgba", rgba);
        writePPM(base + ".ppm", config.width, config.height, rgba);
        writeU32(base + ".counts.u32", state.counts);
        writeF32(base + ".depth.f32", state.depth);
        writeRGBA8(config.outputDirectory + '/' + config.outputName + ".rgba", rgba);
        writeU32(config.outputDirectory + '/' + config.outputName + ".counts.u32", state.counts);
        writeF32(config.outputDirectory + '/' + config.outputName + ".depth.f32", state.depth);

        const json::Value frameStatsJson = statsToJson(stats, psnrDb);
        frameStatsArray.push_back(frameStatsJson);
        logger.info("frame_rendered", "frame rendered and accumulated",
                    json::Value::Object{{"frameIndex", frame},
                                        {"reprojectionAttempts", stats.reprojectionAttempts},
                                        {"reprojectionHits", stats.reprojectionHits},
                                        {"reprojectionMisses", stats.reprojectionMisses},
                                        {"reprojectionHitRate", stats.reprojectionHitRate},
                                        {"minimumCount", stats.minimumCount},
                                        {"maximumCount", stats.maximumCount},
                                        {"countsStrictlyMonotonic", stats.countsStrictlyMonotonic},
                                        {"psnrDb", std::isfinite(psnrDb) ? psnrDb : 999.0}});

        ++renderedFrames;
        lastFrame = frame;

        if ((frame + 1u) % config.snapshot.saveEveryFrames == 0u) {
            const Light nextLight = config.sun.lightAt(frame + 1u);
            SnapshotData snapshotOut = makeSnapshotData(state, config, density, nextLight);
            snapshotOut.version = config.snapshot.version;
            const std::uint32_t snapshotCrc =
                saveSnapshot(config.snapshot.savePath, snapshotOut, config.snapshot.version);
            std::ostringstream crcHex;
            crcHex << std::hex << snapshotCrc;
            std::error_code sizeError;
            const std::uintmax_t fileSize =
                std::filesystem::file_size(config.snapshot.savePath, sizeError);
            logger.info("snapshot_saved", "complete render state serialized",
                        json::Value::Object{{"path", config.snapshot.savePath},
                                            {"version", config.snapshot.version},
                                            {"frameIndex", state.frameIndex},
                                            {"densityResolution", density.resolution()},
                                            {"bytes", sizeError ? 0 : fileSize},
                                            {"crc32Hex", crcHex.str()}});
            if (config.snapshot.stopAfterSave) {
                stopRequested = true;
                break;
            }
        }
    }

    if (renderedFrames == 0) {
        logger.warning("W_NO_FRAMES_RENDERED",
                       "snapshot already reached the configured end frame; state was not re-saved",
                       json::Value::Object{{"nextFrame", currentFrame},
                                           {"endFrameExclusive", config.endFrame}});
    }

    json::Value::Object summary;
    summary["status"] = "ok";
    summary["renderedFrames"] = renderedFrames;
    summary["lastFrameIndex"] = lastFrame;
    summary["stoppedAfterSnapshot"] = stopRequested;
    summary["densitySeed"] = density.seed();
    summary["densityResolution"] = density.resolution();
    summary["cloudType"] = cloudTypeName(density.type());
    summary["width"] = config.width;
    summary["height"] = config.height;
    summary["reprojectionHitRate"] =
        frameStatsArray.empty()
            ? 0.0
            : frameStatsArray.back().find("reprojectionHitRate")->asNumber();
    summary["minimumCount"] = frameStatsArray.empty()
                                  ? 0
                                  : frameStatsArray.back().find("minimumCount")->asUInt64();
    summary["maximumCount"] = frameStatsArray.empty()
                                  ? 0
                                  : frameStatsArray.back().find("maximumCount")->asUInt64();
    summary["minimumUpsamplePsnrDb"] =
        std::isfinite(minimumPsnr) ? minimumPsnr : 0.0;
    summary["frames"] = json::Value(std::move(frameStatsArray));
    summary["snapshotLoadPath"] = config.snapshot.loadPath;
    summary["snapshotSavePath"] = config.snapshot.savePath;
    const std::string summaryText = json::stringify(json::Value(std::move(summary)), true) + "\n";
    const std::vector<std::uint8_t> summaryBytes(summaryText.begin(), summaryText.end());
    writeBytes(config.outputDirectory + "/run_stats.json", summaryBytes);

    logger.info("render_complete", "run completed",
                json::Value::Object{{"renderedFrames", renderedFrames},
                                    {"lastFrame", lastFrame},
                                    {"stoppedAfterSnapshot", stopRequested}});
    return kExitOk;
}

}  // namespace
}  // namespace cloud

int main(int argc, char** argv) {
    using namespace cloud;
    std::string configPath;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--config" && index + 1 < argc) {
            configPath = argv[++index];
        } else if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: cloud_render --config <config.json>\n";
            return kExitOk;
        } else {
            writeErrorToStderr("E_CONFIG", kExitConfig, "unknown command-line argument: " + argument);
            return kExitConfig;
        }
    }
    if (configPath.empty()) {
        writeErrorToStderr("E_CONFIG", kExitConfig, "missing required --config <config.json>");
        return kExitConfig;
    }

    std::unique_ptr<Logger> logger;
    AppConfig config;
    try {
        config = loadConfig(configPath);
        ensureWritableDirectory(config.outputDirectory);
        logger = std::make_unique<Logger>(config.outputDirectory + "/render.log");
        logger->info("config_loaded", "JSON configuration loaded",
                     json::Value::Object{{"path", configPath},
                                         {"outputDir", config.outputDirectory},
                                         {"availableMB", config.memory.availableMB},
                                         {"async", config.asynchronous}});

        std::unique_ptr<WorkerPool> pool;
        if (config.asynchronous) {
            pool = std::make_unique<WorkerPool>(config.workerThreads);
        } else {
            logger->warning("W_SYNC_MODE", "asynchronous density generation disabled",
                            json::Value::Object{{"mainThreadSpinMilliseconds", 0.0}});
        }
        const int result = run(config, *logger, pool.get());
        pool.reset();
        logger->info("thread_cleanup", "all background worker threads joined",
                     json::Value::Object{{"remainingThreads", 0}});
        return result;
    } catch (const BusinessError& error) {
        if (logger) {
            logger->error(error.code(), error.exitCode(), error.what(), error.fields());
        } else {
            writeErrorToStderr(error.code(), error.exitCode(), error.what());
        }
        return error.exitCode();
    } catch (const std::bad_alloc&) {
        if (logger) {
            logger->error("E_OUT_OF_MEMORY", kExitOutOfMemory,
                          "host allocation failed and the active downsample level could not fit");
        } else {
            writeErrorToStderr("E_OUT_OF_MEMORY", kExitOutOfMemory,
                               "host allocation failed");
        }
        return kExitOutOfMemory;
    } catch (const std::exception& error) {
        if (logger) {
            logger->error("E_INTERNAL", kExitInternal, error.what());
        } else {
            writeErrorToStderr("E_INTERNAL", kExitInternal, error.what());
        }
        return kExitInternal;
    }
}

