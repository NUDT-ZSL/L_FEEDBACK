#include "pipeline.hpp"

#include "mesh_io.hpp"
#include "render.hpp"
#include "sha256.hpp"
#include "skinning.hpp"
#include "snapshot.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace fs = std::filesystem;

namespace rigskin {
namespace {

volatile std::sig_atomic_t g_interruptFlag = 0;

struct SharedRunState {
    std::mutex mutex;
    Manifest manifest;
    std::uint32_t framesRecomputed = 0;
    std::vector<std::string> errorCodes;
    std::vector<std::string> errorDetails;
    std::string fatalMessage;
    bool fatal = false;
};

std::string hexInputHash(const std::string& modelText, const PipelineConfig& config) {
    Sha256 context;
    context.update(modelText);
    context.update("\n");
    context.update(pipelineDescriptor(config));
    return Sha256::hex(context.finish());
}

std::string reasonReport(
    const std::string& reason,
    const std::vector<std::pair<std::string, std::string>>& values) {
    std::ostringstream output;
    output << "FMSNAP_REBUILD_REPORT\n";
    output << "reason=" << reason << '\n';
    for (const auto& item : values) {
        output << item.first << '=' << item.second << '\n';
    }
    output << "action=archive_existing_payload_and_rebuild_snapshot\n";
    return output.str();
}

std::string recoveryReport(const std::vector<std::string>& details) {
    std::ostringstream output;
    output << "FMSNAP_FRAME_RECOVERY_REPORT\n";
    for (const std::string& detail : details) {
        output << detail << '\n';
    }
    output << "policy=recompute_corrupt_frame_and_all_subsequent_frames\n";
    return output.str();
}

std::string zeroWeightReport(
    const Model& model,
    const std::string& inputHash,
    const PipelineConfig& config) {
    std::ostringstream output;
    output << "FMSNAP_ZERO_WEIGHT_REPORT\n";
    output << "input_hash=" << inputHash << '\n';
    output << "vertex_count=" << model.zeroWeightVertices.size() << '\n';
    output << "frame_count=" << config.frameCount << '\n';
    for (const std::uint32_t vertex : model.zeroWeightVertices) {
        output << "vertex=" << vertex
               << " action=retain_bind_pose frames=all\n";
    }
    return output.str();
}

bool snapshotHasPayload(const fs::path& directory) {
    std::error_code code;
    if (!fs::exists(directory, code)) {
        return false;
    }
    for (fs::recursive_directory_iterator iterator(
             directory, fs::directory_options::skip_permission_denied, code);
         iterator != fs::recursive_directory_iterator();
         iterator.increment(code)) {
        if (code) {
            return true;
        }
        if (iterator->is_regular_file(code) && !code) {
            return true;
        }
        code.clear();
    }
    return false;
}

void clearTemporaryDirectory(const fs::path& directory) {
    std::error_code code;
    fs::remove_all(directory, code);
    fs::create_directories(directory, code);
    if (code) {
        throw std::runtime_error(
            "cannot prepare temporary directory: " + code.message());
    }
}

std::string statusText(RunStatus status) {
    switch (status) {
        case RunStatus::Complete:
            return "complete";
        case RunStatus::Interrupted:
            return "interrupted";
        case RunStatus::Failed:
            return "failed";
    }
    return "failed";
}

std::vector<std::string> sortedUnique(std::vector<std::string> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

std::string joinCodes(const std::vector<std::string>& codes) {
    if (codes.empty()) {
        return "none";
    }
    std::ostringstream output;
    for (std::size_t index = 0; index < codes.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << codes[index];
    }
    return output.str();
}

bool writeSummary(
    const fs::path& snapshotDirectory,
    const Manifest& manifest,
    RunStatus status,
    std::uint32_t framesSuccess,
    std::uint32_t framesReused,
    std::uint32_t framesRecomputed,
    const std::vector<std::string>& codes,
    const std::string& additionalMessage) {
    std::ostringstream output;
    output << "FMSNAP_SUMMARY\n";
    output << "manifest_version=" << kManifestVersion << '\n';
    output << "input_hash=" << manifest.inputHash << '\n';
    output << "status=" << statusText(status) << '\n';
    output << "frames_total=" << manifest.frameCount << '\n';
    output << "frames_success=" << framesSuccess << '\n';
    output << "frames_reused=" << framesReused << '\n';
    output << "frames_recomputed=" << framesRecomputed << '\n';
    output << "zero_weight_vertices=" << manifest.zeroWeightVertices << '\n';
    output << "error_codes=" << joinCodes(codes) << '\n';
    if (!additionalMessage.empty()) {
        output << "message=" << additionalMessage << '\n';
    }
    output << "summary_end\n";
    std::string error;
    return writeFileAtomic(snapshotDirectory / "summary.txt", output.str(), error);
}

FrameEntry computeFrame(
    const Model& model,
    const Camera& camera,
    const PipelineConfig& config,
    const std::string& inputHash,
    const fs::path& snapshotDirectory,
    std::uint32_t frameIndex) {
    const double time = frameTime(model.duration, static_cast<int>(frameIndex), config.frameCount);
    const std::vector<Mat4> globals = evaluatePoseGlobals(model, time);
    const DeformedMesh deformed = skinMesh(model, globals);
    const std::vector<std::uint8_t> meshBytes =
        encodeMeshBinary(model, frameIndex, deformed);

    const RenderedImage image =
        renderMesh(model, deformed, camera, config.width, config.height);
    const std::vector<std::uint8_t> ppmBytes =
        encodePpm(image.width, image.height, image.rgb);

    FrameEntry entry;
    entry.frame = frameIndex;
    entry.manifestVersion = kManifestVersion;
    entry.inputHash = inputHash;
    entry.meshSha256 = Sha256::digestHex(meshBytes);
    entry.ppmSha256 = Sha256::digestHex(ppmBytes);
    entry.outputChecksum = combinedChecksum(meshBytes, ppmBytes);
    entry.meshBytes = meshBytes.size();
    entry.ppmBytes = ppmBytes.size();
    entry.vertexCount = static_cast<std::uint32_t>(deformed.positions.size());
    entry.triangleCount = static_cast<std::uint32_t>(model.triangles.size());
    entry.degenerateTriangles = deformed.degenerateTriangleCount;
    entry.zeroWeightVertices =
        static_cast<std::uint32_t>(model.zeroWeightVertices.size());

    std::string ioError;
    if (!writeFileAtomic(frameMeshPath(snapshotDirectory, frameIndex), meshBytes, ioError)) {
        throw std::runtime_error(ioError);
    }
    if (!writeFileAtomic(framePpmPath(snapshotDirectory, frameIndex), ppmBytes, ioError)) {
        throw std::runtime_error(ioError);
    }
    return entry;
}

}  // namespace

void requestInterrupt() noexcept {
    g_interruptFlag = 1;
}

bool interruptionRequested() noexcept {
    return g_interruptFlag != 0;
}

RunResult runPipeline(const RunOptions& options) {
    RunResult result;
    const PipelineConfig& config = options.config;
    const fs::path& snapshotDirectory = options.snapshotDirectory;

    try {
        if (config.frameCount <= 0) {
            throw std::runtime_error("frame count must be positive");
        }
        if (config.width <= 0 || config.height <= 0) {
            throw std::runtime_error("render dimensions must be positive");
        }
        if (options.threads <= 0 || options.threads > 4096) {
            throw std::runtime_error("thread count must be between 1 and 4096");
        }
        if (options.snapshotDirectory.empty()) {
            throw std::runtime_error("snapshot directory must not be empty");
        }
        if (options.commitDelayMilliseconds < 0) {
            throw std::runtime_error("commit delay must not be negative");
        }

        const std::string modelText = generateAsciiSample("preview");
        const Model model = parseAsciiModel(modelText);
        const std::string inputHash = hexInputHash(modelText, config);

        const bool hadPayload = snapshotHasPayload(snapshotDirectory);
        const ManifestReadResult manifestRead =
            readManifest(snapshotDirectory / "manifest.txt");

        SharedRunState state;
        state.manifest.version = kManifestVersion;
        state.manifest.inputHash = inputHash;
        state.manifest.frameCount = static_cast<std::uint32_t>(config.frameCount);
        state.manifest.complete = false;
        state.manifest.zeroWeightVertices =
            static_cast<std::uint32_t>(model.zeroWeightVertices.size());

        bool fullRebuild = false;
        bool frameRecovery = false;
        std::uint32_t recomputeFrom =
            static_cast<std::uint32_t>(config.frameCount);
        std::vector<FrameEntry> reusedEntries;

        if (manifestRead.status == ManifestReadStatus::UnsupportedVersion) {
            fullRebuild = true;
            state.errorCodes.push_back("E1003");
            state.errorDetails.push_back(
                "E1003 unsupported_manifest_version detected=" +
                std::to_string(manifestRead.detectedVersion) +
                " expected=" + std::to_string(kManifestVersion));
        } else if (manifestRead.status == ManifestReadStatus::Corrupt) {
            fullRebuild = true;
            state.errorCodes.push_back("E1001");
            state.errorDetails.push_back(
                "E1001 manifest_parse_error detail=" + manifestRead.error);
        } else if (manifestRead.status == ManifestReadStatus::Missing) {
            if (hadPayload) {
                fullRebuild = true;
                state.errorCodes.push_back("E1002");
                state.errorDetails.push_back(
                    "E1002 manifest_missing_with_payload");
            } else {
                recomputeFrom = 0;
            }
        } else if (
            manifestRead.manifest.inputHash != inputHash ||
            manifestRead.manifest.frameCount !=
                static_cast<std::uint32_t>(config.frameCount)) {
            fullRebuild = true;
            state.errorCodes.push_back("E1004");
            state.errorDetails.push_back(
                "E1004 input_hash_changed old=" +
                manifestRead.manifest.inputHash + " new=" + inputHash);
        } else {
            std::map<std::uint32_t, FrameEntry> entryMap;
            for (const FrameEntry& entry : manifestRead.manifest.entries) {
                entryMap.emplace(entry.frame, entry);
            }
            std::uint32_t firstInvalid =
                static_cast<std::uint32_t>(config.frameCount);
            for (std::uint32_t frame = 0;
                 frame < static_cast<std::uint32_t>(config.frameCount);
                 ++frame) {
                bool reusable = true;
                const auto iterator = entryMap.find(frame);
                if (iterator == entryMap.end()) {
                    reusable = false;
                    state.errorCodes.push_back("E2101");
                    state.errorDetails.push_back(
                        "E2101 manifest_entry_missing frame=" +
                        std::to_string(frame));
                } else {
                    const ArtifactValidation validation = validateFrameArtifacts(
                        snapshotDirectory,
                        iterator->second,
                        inputHash,
                        config.width,
                        config.height);
                    if (!validation.valid) {
                        reusable = false;
                        state.errorCodes.push_back(validation.errorCode);
                        state.errorDetails.push_back(
                            validation.errorCode + " frame=" +
                            std::to_string(frame) +
                            " detail=" + validation.detail);
                    }
                }
                if (!reusable && firstInvalid ==
                                     static_cast<std::uint32_t>(config.frameCount)) {
                    firstInvalid = frame;
                }
                if (reusable && firstInvalid ==
                                    static_cast<std::uint32_t>(config.frameCount)) {
                    reusedEntries.push_back(iterator->second);
                }
            }
            if (firstInvalid < static_cast<std::uint32_t>(config.frameCount)) {
                recomputeFrom = firstInvalid;
                frameRecovery = true;
            }
        }

        if (fullRebuild) {
            std::string reason;
            std::string report;
            if (!state.errorCodes.empty() && state.errorCodes.back() == "E1003") {
                reason = "invalid_manifest_version";
                report = reasonReport(
                    reason,
                    {{"detected_version", std::to_string(manifestRead.detectedVersion)},
                     {"expected_version", std::to_string(kManifestVersion)},
                     {"new_input_hash", inputHash}});
            } else if (
                !state.errorCodes.empty() && state.errorCodes.back() == "E1004") {
                reason = "input_hash_changed";
                report = reasonReport(
                    reason,
                    {{"old_input_hash", manifestRead.manifest.inputHash},
                     {"new_input_hash", inputHash},
                     {"old_frame_count",
                      std::to_string(manifestRead.manifest.frameCount)},
                     {"new_frame_count", std::to_string(config.frameCount)}});
            } else {
                reason = "corrupt_manifest";
                report = reasonReport(
                    reason,
                    {{"new_input_hash", inputHash},
                     {"detail", manifestRead.error}});
            }

            fs::path archivePath;
            std::string archiveError;
            if (!archiveSnapshotForRebuild(
                    snapshotDirectory,
                    reason,
                    report,
                    archivePath,
                    archiveError)) {
                throw std::runtime_error(archiveError);
            }
            reusedEntries.clear();
            recomputeFrom = 0;
        } else if (frameRecovery) {
            std::string reportError;
            if (!writeFileAtomic(
                    snapshotDirectory / "errors" / "frame_recovery.txt",
                    recoveryReport(state.errorDetails),
                    reportError)) {
                throw std::runtime_error(reportError);
            }
        }

        std::error_code code;
        fs::create_directories(snapshotDirectory / "frames", code);
        if (code) {
            throw std::runtime_error(
                "cannot create frames directory: " + code.message());
        }
        clearTemporaryDirectory(snapshotDirectory / ".tmp");

        std::string ioError;
        if (!writeFileAtomic(snapshotDirectory / "input_model.txt", modelText, ioError) ||
            !writeFileAtomic(
                snapshotDirectory / "config.txt",
                pipelineDescriptor(config),
                ioError) ||
            !writeFileAtomic(
                snapshotDirectory / "zero_weight_report.txt",
                zeroWeightReport(model, inputHash, config),
                ioError)) {
            throw std::runtime_error(ioError);
        }

        state.manifest.entries = reusedEntries;
        if (!writeManifest(snapshotDirectory / "manifest.txt", state.manifest, ioError)) {
            throw std::runtime_error(ioError);
        }

        const Camera camera = makeCamera(model, config.width, config.height);
        std::vector<std::uint32_t> jobs;
        for (std::uint32_t frame = recomputeFrom;
             frame < static_cast<std::uint32_t>(config.frameCount);
             ++frame) {
            jobs.push_back(frame);
        }

        std::atomic<std::size_t> nextJob{0};
        std::atomic<bool> localAbort{false};
        const std::size_t workerCount =
            std::min<std::size_t>(static_cast<std::size_t>(options.threads), jobs.size());
        std::vector<std::thread> workers;
        workers.reserve(workerCount);

        for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex) {
            (void)workerIndex;
            workers.emplace_back([&]() {
                while (!interruptionRequested() && !localAbort.load()) {
                    const std::size_t jobIndex =
                        nextJob.fetch_add(1, std::memory_order_relaxed);
                    if (jobIndex >= jobs.size()) {
                        break;
                    }
                    const std::uint32_t frame = jobs[jobIndex];
                    try {
                        const FrameEntry entry = computeFrame(
                            model,
                            camera,
                            config,
                            inputHash,
                            snapshotDirectory,
                            frame);
                        {
                            std::lock_guard<std::mutex> lock(state.mutex);
                            state.manifest.entries.erase(
                                std::remove_if(
                                    state.manifest.entries.begin(),
                                    state.manifest.entries.end(),
                                    [frame](const FrameEntry& existing) {
                                        return existing.frame == frame;
                                    }),
                                state.manifest.entries.end());
                            state.manifest.entries.push_back(entry);
                            ++state.framesRecomputed;
                            std::string manifestError;
                            if (!writeManifest(
                                    snapshotDirectory / "manifest.txt",
                                    state.manifest,
                                    manifestError)) {
                                state.fatal = true;
                                state.fatalMessage = manifestError;
                                localAbort.store(true);
                                break;
                            }
                        }
                        if (options.commitDelayMilliseconds > 0) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(
                                options.commitDelayMilliseconds));
                        }
                    } catch (const std::exception& exception) {
                        std::lock_guard<std::mutex> lock(state.mutex);
                        state.errorCodes.push_back("E8001");
                        state.errorDetails.push_back(
                            "E8001 frame_compute_failure frame=" +
                            std::to_string(frame) + " detail=" + exception.what());
                        state.fatal = true;
                        state.fatalMessage = exception.what();
                        localAbort.store(true);
                        break;
                    }
                }
            });
        }
        for (std::thread& worker : workers) {
            worker.join();
        }

        std::sort(
            state.manifest.entries.begin(),
            state.manifest.entries.end(),
            [](const FrameEntry& lhs, const FrameEntry& rhs) {
                return lhs.frame < rhs.frame;
            });
        const std::uint32_t successFrames =
            static_cast<std::uint32_t>(state.manifest.entries.size());
        state.manifest.complete =
            !state.fatal &&
            successFrames == static_cast<std::uint32_t>(config.frameCount);
        if (!writeManifest(snapshotDirectory / "manifest.txt", state.manifest, ioError)) {
            state.fatal = true;
            state.fatalMessage = ioError;
        }

        RunStatus status = RunStatus::Complete;
        if (state.fatal) {
            status = RunStatus::Failed;
        } else if (
            interruptionRequested() ||
            successFrames != static_cast<std::uint32_t>(config.frameCount)) {
            status = RunStatus::Interrupted;
        }
        if (status == RunStatus::Interrupted) {
            state.errorCodes.push_back("E9001");
            state.errorDetails.push_back(
                "E9001 interrupted_before_all_frames_completed");
        }
        if (status == RunStatus::Failed) {
            std::string reportError;
            writeFileAtomic(
                snapshotDirectory / "errors" / "fatal.txt",
                "FMSNAP_FATAL\nmessage=" + state.fatalMessage + "\n",
                reportError);
        }

        if (!state.errorDetails.empty() && status != RunStatus::Failed) {
            std::string reportError;
            writeFileAtomic(
                snapshotDirectory / "errors" / "frame_recovery.txt",
                recoveryReport(state.errorDetails),
                reportError);
        }

        const std::vector<std::string> codes = sortedUnique(state.errorCodes);
        const bool summaryWritten = writeSummary(
            snapshotDirectory,
            state.manifest,
            status,
            successFrames,
            static_cast<std::uint32_t>(reusedEntries.size()),
            state.framesRecomputed,
            codes,
            state.fatalMessage);
        if (!summaryWritten && !state.fatal) {
            result.message = "final summary could not be written";
        }

        result.status = status;
        result.exitCode =
            status == RunStatus::Complete ? 0 : (status == RunStatus::Interrupted ? 130 : 2);
        result.framesSuccess = successFrames;
        result.framesReused = static_cast<std::uint32_t>(reusedEntries.size());
        result.framesRecomputed = state.framesRecomputed;
        result.errorCodes = codes;
        if (result.message.empty()) {
            result.message = state.fatalMessage;
        }
        return result;
    } catch (const std::exception& exception) {
        result.status = RunStatus::Failed;
        result.exitCode = 2;
        result.message = exception.what();
        result.errorCodes.push_back("E9000");

        if (!snapshotDirectory.empty()) {
            std::error_code code;
            fs::create_directories(snapshotDirectory, code);
            Manifest manifest;
            manifest.inputHash = "unavailable";
            manifest.frameCount = static_cast<std::uint32_t>(
                std::max(0, options.config.frameCount));
            writeSummary(
                snapshotDirectory,
                manifest,
                RunStatus::Failed,
                0,
                0,
                0,
                result.errorCodes,
                exception.what());
        }
        return result;
    }
}

}  // namespace rigskin
