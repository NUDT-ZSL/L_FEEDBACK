#include "skinbake/bake.hpp"

#include "skinbake/error.hpp"
#include "skinbake/json.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

namespace skinbake {

namespace {

struct BlockResult {
    std::vector<FrameRecord> frames;
    std::shared_ptr<BusinessError> error;
};

void append_event(const std::string& path,
                  const std::vector<JsonField>& fields) {
    if (path.empty()) {
        return;
    }
    std::ofstream output(path, std::ios::binary | std::ios::app);
    if (!output) {
        throw_error(ErrorCode::SnapshotIoError,
                    "unable to open event log: " + path,
                    {{"path", path}});
    }
    output << json_object(fields) << '\n';
    output.flush();
    if (!output) {
        throw_error(ErrorCode::SnapshotIoError,
                    "unable to write event log: " + path,
                    {{"path", path}});
    }
}

FrameRecord make_frame_record(const TimelineFrame& frame,
                              std::uint32_t frameIndex,
                              std::uint32_t fps,
                              std::uint32_t boneCount,
                              std::uint32_t vertexCount) {
    FrameRecord record;
    record.frameIndex = frameIndex;
    record.timeMs = static_cast<std::uint32_t>(
        rounded_frame_time_ms(frameIndex, fps));
    record.boneMatrices.resize(
        static_cast<std::size_t>(boneCount) * 12u, 0.0f);
    for (std::size_t bone = 0; bone < frame.pose.boneWorld.size(); ++bone) {
        const auto values = matrix_to_float12(frame.pose.boneWorld[bone]);
        std::copy(values.begin(), values.end(),
                  record.boneMatrices.begin() +
                      static_cast<std::ptrdiff_t>(bone * 12u));
    }
    record.vertices.resize(
        static_cast<std::size_t>(vertexCount) * 2u, 0.0f);
    for (std::size_t vertex = 0; vertex < frame.pose.vertices.size(); ++vertex) {
        record.vertices[vertex * 2u] =
            static_cast<float>(frame.pose.vertices[vertex].x);
        record.vertices[vertex * 2u + 1u] =
            static_cast<float>(frame.pose.vertices[vertex].y);
    }
    return record;
}

Snapshot metadata_snapshot(const BakeOptions& options,
                           const TimelineEvaluator& timeline,
                           bool complete) {
    Snapshot snapshot;
    snapshot.version = options.version;
    snapshot.fps = timeline.fps();
    snapshot.fingerprint = timeline.fingerprint();
    snapshot.totalFrames = options.totalFrames;
    snapshot.boneCount =
        static_cast<std::uint32_t>(timeline.prepared().scene.bones.size());
    snapshot.vertexCount =
        static_cast<std::uint32_t>(timeline.prepared().scene.vertices.size());
    snapshot.complete = complete;
    return snapshot;
}

void ensure_parent_directory(const std::string& path) {
    const std::filesystem::path file(path);
    const std::filesystem::path parent = file.parent_path();
    if (!parent.empty()) {
        std::error_code error;
        std::filesystem::create_directories(parent, error);
        if (error) {
            throw_error(ErrorCode::SnapshotIoError,
                        "unable to create output directory: " +
                            parent.string(),
                        {{"path", parent.string()},
                         {"error", error.message()}});
        }
    }
}

std::uint32_t bounded_worker_count(std::uint32_t requested,
                                   std::uint32_t logical,
                                   std::uint32_t workUnits) {
    std::uint32_t count = std::max<std::uint32_t>(requested, 1u);
    count = std::min(count, std::max<std::uint32_t>(logical, 1u));
    count = std::min(count, std::max<std::uint32_t>(workUnits, 1u));
    return std::max<std::uint32_t>(count, 1u);
}

}

std::string progress_path_for(const std::string& outputPath) {
    return outputPath + ".progress";
}

BakeReport bake(const TimelineEvaluator& timeline, const BakeOptions& options) {
    if (options.outputPath.empty()) {
        throw_error(ErrorCode::InvalidArgument, "bake output path is empty");
    }
    if (options.totalFrames == 0) {
        throw_error(ErrorCode::InvalidArgument,
                    "bake frame count must be positive");
    }
    if (options.requestedWorkers == 0) {
        throw_error(ErrorCode::InvalidArgument,
                    "worker count must be positive");
    }
    if (options.blockFrames == 0) {
        throw_error(ErrorCode::InvalidArgument,
                    "block frame count must be positive");
    }
    if (options.outputPath == progress_path_for(options.outputPath)) {
        throw_error(ErrorCode::InvalidArgument,
                    "output and progress paths must differ");
    }
    ensure_parent_directory(options.outputPath);

    const std::uint32_t logicalCores =
        std::max<std::uint32_t>(1u, std::thread::hardware_concurrency());
    const std::string progressPath = progress_path_for(options.outputPath);

    BakeReport report;
    report.version = options.version;
    report.clip = timeline.prepared().scene.defaultClip;
    report.frames = options.totalFrames;
    report.fps = timeline.fps();
    report.fingerprint = timeline.fingerprint();
    report.requestedWorkers = options.requestedWorkers;
    report.logicalCores = logicalCores;
    report.outputPath = options.outputPath;

    Snapshot base = metadata_snapshot(options, timeline, false);
    std::optional<Snapshot> oldProgress;
    bool progressExists = false;
    std::error_code existsError;
    progressExists = std::filesystem::exists(progressPath, existsError);
    if (existsError) {
        throw_error(ErrorCode::SnapshotIoError,
                    "unable to inspect progress path",
                    {{"path", progressPath}, {"error", existsError.message()}});
    }

    std::string discardReason;
    if (progressExists) {
        if (!options.resume) {
            discardReason = "resume-disabled";
        } else {
            try {
                oldProgress = load_snapshot(progressPath);
                if (oldProgress->version != options.version) {
                    discardReason = "version";
                } else if (oldProgress->fingerprint !=
                           timeline.fingerprint()) {
                    discardReason = "fingerprint";
                } else if (oldProgress->fps != timeline.fps() ||
                           oldProgress->totalFrames != options.totalFrames ||
                           oldProgress->boneCount != base.boneCount ||
                           oldProgress->vertexCount != base.vertexCount) {
                    discardReason = "format";
                }
            } catch (const BusinessError& error) {
                switch (error.code()) {
                    case ErrorCode::SnapshotChecksumMismatch:
                        discardReason = "checksum";
                        break;
                    case ErrorCode::SnapshotMagicMismatch:
                        discardReason = "magic";
                        break;
                    case ErrorCode::SnapshotVersionMismatch:
                        discardReason = "version";
                        break;
                    case ErrorCode::SnapshotIoError:
                        throw;
                    default:
                        discardReason = "corrupt";
                        break;
                }
            }
        }
    }

    if (progressExists && !discardReason.empty()) {
        const std::vector<JsonField> event{
            {"event", json_string("discard")},
            {"reason", json_string(discardReason)},
            {"oldVersion",
             oldProgress.has_value()
                 ? json_u64(oldProgress->version)
                 : json_string("unknown")},
            {"newVersion", json_u64(options.version)},
            {"oldFingerprint",
             oldProgress.has_value()
                 ? json_string(std::to_string(oldProgress->fingerprint))
                 : json_string("unknown")},
            {"newFingerprint", json_string(std::to_string(timeline.fingerprint()))},
        };
        append_event(options.eventLogPath, event);
        report.discardedProgress = true;
        report.discardReason = discardReason;
        write_snapshot_atomic(progressPath, base);
        oldProgress.reset();
    }

    Snapshot progress = oldProgress.has_value() ? *oldProgress : base;
    if (progress.frames.size() > progress.totalFrames) {
        throw_error(ErrorCode::SnapshotDataInvalid,
                    "progress contains more frames than requested");
    }
    report.resumeFrame = static_cast<std::uint32_t>(progress.frames.size());
    report.firstNewFrame = report.resumeFrame;
    report.resumed = oldProgress.has_value();

    if (oldProgress.has_value()) {
        const std::vector<JsonField> event{
            {"event", json_string("resume")},
            {"frame", json_u64(report.resumeFrame)},
            {"version", json_u64(options.version)},
        };
        append_event(options.eventLogPath, event);
    }

    if (progress.frames.size() < options.totalFrames) {
        const std::uint32_t blockFrames = options.blockFrames;
        const std::uint32_t totalBlocks =
            (options.totalFrames + blockFrames - 1u) / blockFrames;
        const std::uint32_t firstBlock =
            static_cast<std::uint32_t>(progress.frames.size()) / blockFrames;
        std::vector<BlockResult> blocks(totalBlocks);
        std::atomic<std::size_t> nextBlock{firstBlock};

        std::uint32_t blocksToCompute = 0;
        for (std::uint32_t block = firstBlock; block < totalBlocks; ++block) {
            const std::uint32_t end =
                std::min(options.totalFrames, (block + 1u) * blockFrames);
            if (end > progress.frames.size()) {
                ++blocksToCompute;
            }
        }
        const std::uint32_t usedWorkers = bounded_worker_count(
            options.requestedWorkers, logicalCores, blocksToCompute);
        report.usedWorkers = usedWorkers;

        auto compute_block = [&](std::uint32_t block) {
            const std::uint32_t begin = block * blockFrames;
            const std::uint32_t end =
                std::min(options.totalFrames, begin + blockFrames);
            const std::uint32_t start =
                std::max(begin, static_cast<std::uint32_t>(progress.frames.size()));
            try {
                blocks[block].frames.reserve(
                    static_cast<std::size_t>(end - start));
                for (std::uint32_t frame = start; frame < end; ++frame) {
                    const TimelineFrame sampled = timeline.evaluate(frame);
                    blocks[block].frames.push_back(make_frame_record(
                        sampled, frame, timeline.fps(), base.boneCount,
                        base.vertexCount));
                }
            } catch (const BusinessError& error) {
                blocks[block].error =
                    std::make_shared<BusinessError>(error);
            } catch (const std::exception& error) {
                blocks[block].error = std::make_shared<BusinessError>(
                    ErrorCode::InternalError,
                    std::string("frame evaluation failed: ") + error.what());
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(usedWorkers);
        for (std::uint32_t worker = 0; worker < usedWorkers; ++worker) {
            workers.emplace_back([&]() {
                while (true) {
                    const std::size_t block = nextBlock.fetch_add(1);
                    if (block >= totalBlocks) {
                        break;
                    }
                    compute_block(static_cast<std::uint32_t>(block));
                }
            });
        }
        for (std::thread& worker : workers) {
            worker.join();
        }

        std::shared_ptr<BusinessError> firstError;
        for (std::size_t block = 0; block < blocks.size(); ++block) {
            if (blocks[block].error) {
                firstError = blocks[block].error;
                break;
            }
        }
        if (firstError) {
            throw *firstError;
        }

        for (std::uint32_t frame = report.resumeFrame;
             frame < options.totalFrames; ++frame) {
            const std::uint32_t block = frame / blockFrames;
            const std::uint32_t blockBegin =
                std::max(block * blockFrames, report.resumeFrame);
            const std::size_t local = frame - blockBegin;
            if (local >= blocks[block].frames.size() ||
                blocks[block].frames[local].frameIndex != frame) {
                throw_error(ErrorCode::InternalError,
                            "parallel frame result is missing or misordered",
                            {{"frame", std::to_string(frame)}});
            }
            FrameRecord record = std::move(blocks[block].frames[local]);
            progress.frames.push_back(std::move(record));
            progress.complete = false;
            write_snapshot_atomic(progressPath, progress);
            ++report.newFramesCommitted;
            append_event(options.eventLogPath,
                         {{"event", json_string("commit")},
                          {"frame", json_u64(frame)},
                          {"version", json_u64(options.version)}});
            if (options.onFrameCommitted) {
                options.onFrameCommitted(progress.frames.back());
            }
            if (options.crashAfterCommitFrame.has_value() &&
                frame == *options.crashAfterCommitFrame) {
                std::_Exit(137);
            }
        }
    }

    progress.complete = true;
    write_snapshot_atomic(progressPath, progress);
    write_snapshot_atomic(options.outputPath, progress);
    append_event(options.eventLogPath,
                 {{"event", json_string("complete")},
                  {"frames", json_u64(options.totalFrames)},
                  {"version", json_u64(options.version)},
                  {"resumedFrom", json_u64(report.resumeFrame)}});
    return report;
}

}

