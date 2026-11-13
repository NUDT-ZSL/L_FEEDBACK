#include "skinbake/verify.hpp"

#include "skinbake/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace skinbake {

namespace {

bool float_bits_equal(float a, float b) {
    std::uint32_t first = 0;
    std::uint32_t second = 0;
    std::memcpy(&first, &a, sizeof(first));
    std::memcpy(&second, &b, sizeof(second));
    return first == second;
}

FrameRecord reference_frame(const TimelineEvaluator& timeline,
                            std::uint32_t frameIndex,
                            std::uint32_t boneCount,
                            std::uint32_t vertexCount) {
    const TimelineFrame sampled = timeline.evaluate(frameIndex);
    FrameRecord record;
    record.frameIndex = frameIndex;
    record.timeMs = static_cast<std::uint32_t>(
        rounded_frame_time_ms(frameIndex, timeline.fps()));
    record.boneMatrices.resize(
        static_cast<std::size_t>(boneCount) * 12u, 0.0f);
    for (std::size_t bone = 0; bone < sampled.pose.boneWorld.size(); ++bone) {
        const auto values = matrix_to_float12(sampled.pose.boneWorld[bone]);
        std::copy(values.begin(), values.end(),
                  record.boneMatrices.begin() +
                      static_cast<std::ptrdiff_t>(bone * 12u));
    }
    record.vertices.resize(
        static_cast<std::size_t>(vertexCount) * 2u, 0.0f);
    for (std::size_t vertex = 0; vertex < sampled.pose.vertices.size();
         ++vertex) {
        record.vertices[vertex * 2u] =
            static_cast<float>(sampled.pose.vertices[vertex].x);
        record.vertices[vertex * 2u + 1u] =
            static_cast<float>(sampled.pose.vertices[vertex].y);
    }
    return record;
}

}

VerificationReport verify_snapshot(const TimelineEvaluator& timeline,
                                   const std::string& path,
                                   double tolerance,
                                   std::optional<std::uint32_t> expectedVersion) {
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw_error(ErrorCode::InvalidArgument,
                    "verification tolerance must be finite and non-negative");
    }
    VerificationReport report;
    report.tolerance = tolerance;

    TimelineDefinition freshDefinition;
    freshDefinition.scene = timeline.prepared().scene;
    freshDefinition.transitions = timeline.transitions();
    const auto freshTimeline = prepare_timeline(freshDefinition);
    const Snapshot snapshot =
        load_snapshot(path, expectedVersion, timeline.fingerprint());

    report.frames = static_cast<std::uint32_t>(snapshot.frames.size());
    bool metadataMatches = true;
    metadataMatches = metadataMatches && snapshot.complete;
    metadataMatches = metadataMatches && snapshot.fps == timeline.fps();
    metadataMatches =
        metadataMatches &&
        snapshot.boneCount ==
            static_cast<std::uint32_t>(timeline.prepared().scene.bones.size());
    metadataMatches =
        metadataMatches &&
        snapshot.vertexCount ==
            static_cast<std::uint32_t>(timeline.prepared().scene.vertices.size());
    metadataMatches = metadataMatches && snapshot.fingerprint == timeline.fingerprint();

    for (std::size_t index = 0; index < snapshot.frames.size(); ++index) {
        const FrameRecord& actual = snapshot.frames[index];
        const FrameRecord expected = reference_frame(
            *freshTimeline, static_cast<std::uint32_t>(index),
            snapshot.boneCount, snapshot.vertexCount);
        bool mismatch = false;
        bool bytesEqual = actual.frameIndex == expected.frameIndex &&
                          actual.timeMs == expected.timeMs &&
                          actual.boneMatrices.size() ==
                              expected.boneMatrices.size() &&
                          actual.vertices.size() == expected.vertices.size();
        if (actual.frameIndex != expected.frameIndex ||
            actual.timeMs != expected.timeMs) {
            mismatch = true;
        }
        auto compare_values = [&](const std::vector<float>& a,
                                  const std::vector<float>& b) {
            if (a.size() != b.size()) {
                mismatch = true;
                report.byteIdentical = false;
                return;
            }
            for (std::size_t i = 0; i < a.size(); ++i) {
                const double error =
                    std::fabs(static_cast<double>(a[i]) -
                              static_cast<double>(b[i]));
                if (error > report.maxAbsError) {
                    report.maxAbsError = error;
                    report.maxErrorFrame =
                        static_cast<std::uint32_t>(index);
                }
                if (error > tolerance) {
                    mismatch = true;
                }
                if (!float_bits_equal(a[i], b[i])) {
                    bytesEqual = false;
                }
            }
        };
        compare_values(actual.boneMatrices, expected.boneMatrices);
        compare_values(actual.vertices, expected.vertices);
        if (!bytesEqual) {
            report.byteIdentical = false;
        }
        if (mismatch) {
            report.mismatchFrames.push_back(
                static_cast<std::uint32_t>(index));
        }
    }

    report.passed = metadataMatches && report.mismatchFrames.empty();
    return report;
}

}

