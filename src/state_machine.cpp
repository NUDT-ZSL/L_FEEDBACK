#include "skinbake/state_machine.hpp"

#include "skinbake/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

namespace skinbake {

namespace {

std::uint64_t mix_u64(std::uint64_t hash, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        hash ^= static_cast<std::uint8_t>((value >> (i * 8)) & 0xffu);
        hash *= 1099511628211ull;
    }
    return hash;
}

std::uint64_t mix_string(std::uint64_t hash, const std::string& value) {
    hash = mix_u64(hash, value.size());
    for (unsigned char ch : value) {
        hash ^= ch;
        hash *= 1099511628211ull;
    }
    return hash;
}

std::uint64_t timeline_fingerprint(
    const PreparedScene& prepared,
    const std::vector<TransitionSpec>& transitions) {
    std::uint64_t hash = mix_u64(1469598103934665603ull,
                                 prepared.fingerprint);
    hash = mix_u64(hash, transitions.size());
    for (const TransitionSpec& transition : transitions) {
        hash = mix_u64(hash, transition.switchFrame);
        hash = mix_string(hash, transition.toClip);
        hash = mix_u64(hash, static_cast<std::uint64_t>(transition.blendWindowMs));
    }
    return hash;
}

const Clip& clip_for(const PreparedScene& prepared, std::size_t index) {
    return prepared.scene.clips[index];
}

double blend_ratio(std::int64_t elapsedMs, std::int64_t windowMs) {
    if (windowMs <= 0) {
        return 1.0;
    }
    if (elapsedMs <= 0) {
        return 0.0;
    }
    if (elapsedMs >= windowMs) {
        return 1.0;
    }
    return static_cast<double>(elapsedMs) / static_cast<double>(windowMs);
}

TimelineFrame evaluate_segment_frame(
    const PreparedScene& prepared, const std::string& clipName,
    std::int64_t clipTimeMs) {
    const std::size_t index = clip_index(prepared, clipName);
    const Clip& clip = clip_for(prepared, index);
    const std::int64_t playback =
        wrap_playback_time(clipTimeMs, clip.durationMs, clip.loop);
    TimelineFrame frame;
    frame.pose = evaluate_clip_pose(prepared, index, playback);
    frame.clipName = clipName;
    frame.playbackTimeMs = playback;
    return frame;
}

}

Pose evaluate_clip_pose(const PreparedScene& prepared,
                        std::size_t clipIndexValue,
                        std::int64_t playbackTimeMs) {
    const Scene& scene = prepared.scene;
    const Clip& clip = scene.clips.at(clipIndexValue);
    const std::size_t boneCount = scene.bones.size();
    std::vector<Mat3> world(boneCount);

    for (int boneValue : prepared.topologicalOrder) {
        const std::size_t bone = static_cast<std::size_t>(boneValue);
        const BoneCurves& curves = clip.bones[bone];
        Transform local;
        local.translation.x =
            curves.channels[TranslationX].evaluate(
                static_cast<double>(playbackTimeMs));
        local.translation.y =
            curves.channels[TranslationY].evaluate(
                static_cast<double>(playbackTimeMs));
        local.rotationRadians =
            curves.channels[Rotation].evaluate(
                static_cast<double>(playbackTimeMs)) *
            kPi / 180.0;
        local.scale.x = curves.channels[ScaleX].evaluate(
            static_cast<double>(playbackTimeMs));
        local.scale.y = curves.channels[ScaleY].evaluate(
            static_cast<double>(playbackTimeMs));
        if (!is_finite(local.translation) || !std::isfinite(local.rotationRadians) ||
            !is_finite(local.scale)) {
            throw_error(ErrorCode::TransformNonFinite,
                        "evaluated bone transform is not finite",
                        {{"clip", clip.name},
                         {"boneIndex", std::to_string(bone)}});
        }
        const Mat3 localMatrix = local.to_matrix();
        const int parent = scene.bones[bone].parent;
        world[bone] = parent < 0
                          ? localMatrix
                          : world[static_cast<std::size_t>(parent)] * localMatrix;
        if (!is_finite(world[bone])) {
            throw_error(ErrorCode::TransformNonFinite,
                        "accumulated bone matrix is not finite",
                        {{"clip", clip.name},
                         {"boneIndex", std::to_string(bone)}});
        }
    }

    std::vector<Vec2> vertices(scene.vertices.size());
    for (std::size_t vertexIndex = 0; vertexIndex < scene.vertices.size();
         ++vertexIndex) {
        const Vertex& vertex = scene.vertices[vertexIndex];
        Vec2 accumulated{0.0, 0.0};
        for (std::uint32_t influence = 0;
             influence < vertex.influenceCount; ++influence) {
            const std::size_t bone = static_cast<std::size_t>(
                vertex.boneIndices[influence]);
            const Vec2 transformed = world[bone].transform_point(vertex.position);
            accumulated =
                accumulated + transformed * vertex.weights[influence];
        }
        vertices[vertexIndex] = accumulated;
    }

    return Pose{std::move(world), std::move(vertices)};
}

TimelineEvaluator::TimelineEvaluator(
    std::shared_ptr<const PreparedScene> prepared,
    std::vector<TransitionSpec> transitions, std::uint32_t fps)
    : prepared_(std::move(prepared)),
      transitions_(std::move(transitions)),
      fps_(fps) {
    if (!prepared_) {
        throw_error(ErrorCode::StateMachineInvalid,
                    "timeline requires a prepared scene");
    }
    if (fps_ == 0) {
        throw_error(ErrorCode::InvalidArgument, "timeline fps must be positive");
    }

    std::sort(transitions_.begin(), transitions_.end(),
              [](const TransitionSpec& a, const TransitionSpec& b) {
                  return a.switchFrame < b.switchFrame;
              });
    for (std::size_t i = 0; i < transitions_.size(); ++i) {
        const TransitionSpec& transition = transitions_[i];
        if (i > 0 && transition.switchFrame <= transitions_[i - 1].switchFrame) {
            throw_error(ErrorCode::TransitionInvalid,
                        "transition switch frames must be strictly increasing",
                        {{"switchFrame", std::to_string(transition.switchFrame)}});
        }
        if (transition.switchFrame == 0) {
            throw_error(ErrorCode::TransitionInvalid,
                        "transition switch frame must be greater than zero",
                        {{"switchFrame", "0"}});
        }
        if (transition.blendWindowMs < 0) {
            throw_error(ErrorCode::TransitionInvalid,
                        "transition blend window cannot be negative",
                        {{"blendWindowMs",
                          std::to_string(transition.blendWindowMs)}});
        }
        (void)clip_index(*prepared_, transition.toClip);
    }

    segments_.clear();
    Segment initial;
    initial.startFrame = 0;
    initial.startTimeMs = 0;
    initial.clipName = prepared_->scene.defaultClip;
    segments_.push_back(initial);

    for (const TransitionSpec& transition : transitions_) {
        const Segment& previous = segments_.back();
        const std::int64_t switchTime =
            rounded_frame_time_ms(transition.switchFrame, fps_);
        TimelineFrame oldFrame = evaluate_segment_frame(
            *prepared_, previous.clipName,
            switchTime - previous.startTimeMs);
        if (previous.hasPreviousSnapshot && previous.blendWindowMs > 0) {
            const double weight = blend_ratio(
                switchTime - previous.startTimeMs, previous.blendWindowMs);
            if (weight < 1.0) {
                for (std::size_t i = 0; i < oldFrame.pose.boneWorld.size(); ++i) {
                    oldFrame.pose.boneWorld[i] = lerp(
                        previous.previousSnapshot.pose.boneWorld[i],
                        oldFrame.pose.boneWorld[i], weight);
                }
                for (std::size_t i = 0; i < oldFrame.pose.vertices.size(); ++i) {
                    oldFrame.pose.vertices[i] = lerp(
                        previous.previousSnapshot.pose.vertices[i],
                        oldFrame.pose.vertices[i], weight);
                }
            }
        }

        Segment segment;
        segment.startFrame = transition.switchFrame;
        segment.startTimeMs = switchTime;
        segment.clipName = transition.toClip;
        segment.hasPreviousSnapshot = true;
        segment.previousSnapshot.clipName = previous.clipName;
        segment.previousSnapshot.playbackTimeMs = oldFrame.playbackTimeMs;
        segment.previousSnapshot.pose = std::move(oldFrame.pose);
        segment.blendWindowMs = transition.blendWindowMs;
        segments_.push_back(std::move(segment));
    }

    fingerprint_ = timeline_fingerprint(*prepared_, transitions_);
}

TimelineFrame TimelineEvaluator::evaluate(std::uint32_t frameIndex) const {
    auto it = std::upper_bound(
        segments_.begin(), segments_.end(), frameIndex,
        [](std::uint32_t frame, const Segment& segment) {
            return frame < segment.startFrame;
        });
    if (it == segments_.begin()) {
        throw_error(ErrorCode::StateMachineInvalid,
                    "timeline has no active segment",
                    {{"frame", std::to_string(frameIndex)}});
    }
    --it;
    const Segment& segment = *it;
    const std::int64_t frameTime = rounded_frame_time_ms(frameIndex, fps_);
    const std::int64_t elapsed = frameTime - segment.startTimeMs;
    TimelineFrame frame = evaluate_segment_frame(
        *prepared_, segment.clipName, elapsed);

    if (segment.hasPreviousSnapshot && segment.blendWindowMs > 0) {
        const double weight = blend_ratio(elapsed, segment.blendWindowMs);
        frame.blendWeight = weight;
        frame.blending = weight < 1.0;
        if (frame.blending) {
            for (std::size_t i = 0; i < frame.pose.boneWorld.size(); ++i) {
                frame.pose.boneWorld[i] = lerp(
                    segment.previousSnapshot.pose.boneWorld[i],
                    frame.pose.boneWorld[i], weight);
            }
            for (std::size_t i = 0; i < frame.pose.vertices.size(); ++i) {
                frame.pose.vertices[i] = lerp(
                    segment.previousSnapshot.pose.vertices[i],
                    frame.pose.vertices[i], weight);
            }
        }
    }
    return frame;
}

std::shared_ptr<const TimelineEvaluator> prepare_timeline(
    const TimelineDefinition& definition) {
    auto prepared = prepare_scene(definition.scene);
    return std::make_shared<TimelineEvaluator>(
        prepared, definition.transitions, definition.scene.fps);
}

StateMachine::StateMachine(std::shared_ptr<const PreparedScene> prepared,
                           std::string initialClip)
    : prepared_(std::move(prepared)),
      currentClip_(std::move(initialClip)) {
    if (!prepared_) {
        throw_error(ErrorCode::StateMachineInvalid,
                    "state machine requires a prepared scene");
    }
    currentClipIndex_ = clip_index(*prepared_, currentClip_);
}

void StateMachine::switch_clip(const std::string& clipName,
                               std::int64_t blendWindowMs) {
    if (blendWindowMs < 0) {
        throw_error(ErrorCode::StateMachineInvalid,
                    "blend window cannot be negative");
    }
    const std::size_t nextIndex = clip_index(*prepared_, clipName);
    const TimelineFrame current = sample();
    snapshot_.clipName = currentClip_;
    snapshot_.playbackTimeMs = playbackTimeMs_;
    snapshot_.pose = current.pose;
    hasSnapshot_ = true;
    blendWindowMs_ = blendWindowMs;
    blendElapsedMs_ = 0;
    currentClip_ = clipName;
    currentClipIndex_ = nextIndex;
    playbackTimeMs_ = 0;
}

void StateMachine::advance(std::int64_t deltaMs) {
    if (deltaMs < 0) {
        throw_error(ErrorCode::StateMachineInvalid,
                    "state machine time cannot move backwards");
    }
    propagationTimeMs_ += deltaMs;
    playbackTimeMs_ += deltaMs;
    const Clip& clip = prepared_->scene.clips[currentClipIndex_];
    playbackTimeMs_ =
        wrap_playback_time(playbackTimeMs_, clip.durationMs, clip.loop);
    if (hasSnapshot_) {
        blendElapsedMs_ += deltaMs;
    }
}

void StateMachine::advance_rounded(double deltaMs) {
    if (!std::isfinite(deltaMs)) {
        throw_error(ErrorCode::StateMachineInvalid,
                    "state machine delta is not finite");
    }
    advance(static_cast<std::int64_t>(std::llround(deltaMs)));
}

double StateMachine::blend_weight() const noexcept {
    if (!hasSnapshot_ || blendWindowMs_ <= 0) {
        return 0.0;
    }
    return blend_ratio(blendElapsedMs_, blendWindowMs_);
}

TimelineFrame StateMachine::sample() const {
    TimelineFrame frame;
    frame.pose =
        evaluate_clip_pose(*prepared_, currentClipIndex_, playbackTimeMs_);
    frame.clipName = currentClip_;
    frame.playbackTimeMs = playbackTimeMs_;
    if (hasSnapshot_ && blendWindowMs_ > 0) {
        const double weight = blend_weight();
        frame.blendWeight = weight;
        frame.blending = weight < 1.0;
        if (frame.blending) {
            for (std::size_t i = 0; i < frame.pose.boneWorld.size(); ++i) {
                frame.pose.boneWorld[i] = lerp(
                    snapshot_.pose.boneWorld[i], frame.pose.boneWorld[i], weight);
            }
            for (std::size_t i = 0; i < frame.pose.vertices.size(); ++i) {
                frame.pose.vertices[i] = lerp(
                    snapshot_.pose.vertices[i], frame.pose.vertices[i], weight);
            }
        }
    }
    return frame;
}

}

