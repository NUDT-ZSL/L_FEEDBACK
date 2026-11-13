#pragma once

#include "skinbake/scene.hpp"

#include <cstdint>
#include <string>

namespace skinbake {

struct Pose {
    std::vector<Mat3> boneWorld;
    std::vector<Vec2> vertices;
};

struct PoseSnapshot {
    std::string clipName;
    std::int64_t playbackTimeMs = 0;
    Pose pose;
};

Pose evaluate_clip_pose(const PreparedScene& prepared, std::size_t clipIndex,
                        std::int64_t playbackTimeMs);

struct TransitionSpec {
    std::uint32_t switchFrame = 0;
    std::string toClip;
    std::int64_t blendWindowMs = 0;
};

struct TimelineDefinition {
    Scene scene;
    std::vector<TransitionSpec> transitions;
};

struct TimelineFrame {
    Pose pose;
    std::string clipName;
    std::int64_t playbackTimeMs = 0;
    double blendWeight = 0.0;
    bool blending = false;
};

struct TimedPose {
    Pose pose;
    std::string clipName;
    std::int64_t playbackTimeMs = 0;
    double blendWeight = 0.0;
};

class TimelineEvaluator {
public:
    TimelineEvaluator(std::shared_ptr<const PreparedScene> prepared,
                      std::vector<TransitionSpec> transitions,
                      std::uint32_t fps);

    TimelineFrame evaluate(std::uint32_t frameIndex) const;

    const PreparedScene& prepared() const noexcept { return *prepared_; }
    std::shared_ptr<const PreparedScene> prepared_shared() const noexcept {
        return prepared_;
    }
    const std::vector<TransitionSpec>& transitions() const noexcept {
        return transitions_;
    }
    std::uint32_t fps() const noexcept { return fps_; }
    std::uint64_t fingerprint() const noexcept { return fingerprint_; }

private:
    struct Segment {
        std::uint32_t startFrame = 0;
        std::int64_t startTimeMs = 0;
        std::string clipName;
        bool hasPreviousSnapshot = false;
        PoseSnapshot previousSnapshot;
        std::int64_t blendWindowMs = 0;
    };

    std::shared_ptr<const PreparedScene> prepared_;
    std::vector<TransitionSpec> transitions_;
    std::vector<Segment> segments_;
    std::uint32_t fps_ = 60;
    std::uint64_t fingerprint_ = 0;
};

std::shared_ptr<const TimelineEvaluator> prepare_timeline(
    const TimelineDefinition& definition);

class StateMachine {
public:
    StateMachine(std::shared_ptr<const PreparedScene> prepared,
                 std::string initialClip);

    void switch_clip(const std::string& clipName, std::int64_t blendWindowMs);
    void advance(std::int64_t deltaMs);
    void advance_rounded(double deltaMs);

    TimelineFrame sample() const;
    const std::string& current_clip() const noexcept { return currentClip_; }
    std::int64_t playback_time_ms() const noexcept { return playbackTimeMs_; }
    std::int64_t propagation_time_ms() const noexcept { return propagationTimeMs_; }
    bool blending() const noexcept {
        return hasSnapshot_ && blendElapsedMs_ < blendWindowMs_;
    }
    double blend_weight() const noexcept;
    const PoseSnapshot* pose_snapshot() const noexcept {
        return hasSnapshot_ ? &snapshot_ : nullptr;
    }

private:
    std::shared_ptr<const PreparedScene> prepared_;
    std::string currentClip_;
    std::size_t currentClipIndex_ = 0;
    std::int64_t playbackTimeMs_ = 0;
    std::int64_t propagationTimeMs_ = 0;
    bool hasSnapshot_ = false;
    PoseSnapshot snapshot_;
    std::int64_t blendWindowMs_ = 0;
    std::int64_t blendElapsedMs_ = 0;
};

}

