#include "lightcluster/json_log.hpp"

#include <iomanip>
#include <sstream>
#include <string>

namespace lightcluster {

namespace {

std::ostringstream makeStream() {
    std::ostringstream stream;
    stream << std::setprecision(6);
    return stream;
}

void writeVec3(std::ostringstream& stream, Vec3 value) {
    stream << '[' << value.x << ',' << value.y << ',' << value.z << ']';
}

const char* boolText(bool value) {
    return value ? "true" : "false";
}

std::string escapeJson(std::string_view text) {
    std::string escaped;
    escaped.reserve(text.size());
    constexpr char kHex[] = "0123456789abcdef";
    for (const unsigned char character : text) {
        switch (character) {
            case '"':
                escaped += "\\\"";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            case '\b':
                escaped += "\\b";
                break;
            case '\f':
                escaped += "\\f";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                if (character < 0x20U) {
                    escaped += "\\u00";
                    escaped += kHex[character >> 4U];
                    escaped += kHex[character & 0x0fU];
                } else {
                    escaped += static_cast<char>(character);
                }
                break;
        }
    }
    return escaped;
}

}  // namespace

std::string_view toString(RunEventKind kind) noexcept {
    switch (kind) {
        case RunEventKind::LightAdded:
            return "light_added";
        case RunEventKind::LightRemoved:
            return "light_removed";
        case RunEventKind::AnimationQueued:
            return "animation_queued";
        case RunEventKind::AnimationRejected:
            return "animation_rejected";
        case RunEventKind::SnapshotSaved:
            return "snapshot_saved";
        case RunEventKind::SnapshotRestored:
            return "snapshot_restored";
    }
    return "unknown";
}

std::string formatEventJson(const RunEvent& event) {
    auto stream = makeStream();
    stream << "{\"event\":\"" << toString(event.kind)
           << "\",\"frame\":" << event.frame;

    switch (event.kind) {
        case RunEventKind::LightAdded:
            stream << ",\"light_id\":" << event.lightId
                   << ",\"light_type\":\"" << toString(event.lightType)
                   << "\",\"version_before\":" << event.versionBefore
                   << ",\"version\":" << event.versionAfter
                   << ",\"active_lights\":" << event.activeLights;
            break;
        case RunEventKind::LightRemoved:
            stream << ",\"light_id\":" << event.lightId
                   << ",\"version_before\":" << event.versionBefore
                   << ",\"version\":" << event.versionAfter
                   << ",\"active_lights\":" << event.activeLights;
            break;
        case RunEventKind::AnimationQueued:
            stream << ",\"task_id\":" << event.taskId
                   << ",\"light_id\":" << event.lightId
                   << ",\"duration_frames\":" << event.durationFrames
                   << ",\"start_frame\":" << event.startFrame
                   << ",\"end_frame\":" << event.endFrame
                   << ",\"animate_position\":"
                   << boolText(event.animatePosition)
                   << ",\"animate_intensity\":"
                   << boolText(event.animateIntensity)
                   << ",\"target_position\":";
            writeVec3(stream, event.targetPosition);
            stream << ",\"target_intensity\":" << event.targetIntensity
                   << ",\"active_tasks\":" << event.activeTasks
                   << ",\"pool_capacity\":" << event.poolCapacity
                   << ",\"scene_version\":" << event.versionAfter;
            break;
        case RunEventKind::AnimationRejected:
            stream << ",\"business_code\":\"" << event.businessCode
                   << "\",\"light_id\":" << event.lightId
                   << ",\"active_tasks\":" << event.activeTasks
                   << ",\"pool_capacity\":" << event.poolCapacity
                   << ",\"rejected_total\":" << event.rejectedTotal
                   << ",\"scene_version\":" << event.versionAfter;
            break;
        case RunEventKind::SnapshotSaved:
            stream << ",\"snapshot_version\":" << event.restoredSnapshotVersion
                   << ",\"version\":" << event.versionAfter
                   << ",\"active_lights\":" << event.activeLights;
            break;
        case RunEventKind::SnapshotRestored:
            stream << ",\"restored_snapshot_version\":"
                   << event.restoredSnapshotVersion
                   << ",\"version_before\":" << event.versionBefore
                   << ",\"version\":" << event.versionAfter
                   << ",\"version_strictly_increased\":"
                   << boolText(event.versionAfter > event.versionBefore)
                   << ",\"active_lights\":" << event.activeLights;
            break;
    }

    stream << '}';
    return stream.str();
}

std::string formatFrameJson(
    const FrameTick& tick,
    const RunConfig& config,
    const DebugFrameStatus& debug) {
    const LightCounts counts = tick.frontend.counts();
    auto stream = makeStream();
    stream << "{\"event\":\"frame\""
           << ",\"frame\":" << tick.frame
           << ",\"sync_mode\":\"" << toString(config.syncMode)
           << "\",\"delay_frames\":" << config.delayFrames
           << ",\"seed\":" << config.seed
           << ",\"scene_version\":" << tick.frontend.version
           << ",\"gpu_version\":" << tick.sync.gpuVersion
           << ",\"active_lights\":" << counts.total
           << ",\"point_lights\":" << counts.point
           << ",\"directional_lights\":" << counts.directional
           << ",\"uploaded\":" << boolText(tick.sync.uploaded)
           << ",\"upload_reason\":\"" << tick.sync.reason << '"'
           << ",\"buffer_capacity\":" << tick.sync.capacityLights
           << ",\"buffer_grew\":" << boolText(tick.sync.grew)
           << ",\"gpu_light_count\":" << tick.sync.gpuLightCount
           << ",\"gpu_point_count\":" << tick.sync.gpuPointCount
           << ",\"gpu_directional_count\":" << tick.sync.gpuDirectionalCount
           << ",\"gpu_source_frame\":" << tick.sync.sourceFrame
           << ",\"snapshot_age_frames\":" << tick.sync.snapshotAgeFrames
           << ",\"pending_snapshots\":" << tick.sync.pendingSnapshots
           << ",\"animation_active\":" << tick.animationActiveAfterQueue
           << ",\"animations_applied\":" << tick.animations.applied
           << ",\"animations_completed\":" << tick.animations.completed
           << ",\"animations_cancelled\":" << tick.animations.cancelled
           << ",\"animation_rejected_this_frame\":"
           << tick.animationRejectedThisFrame
           << ",\"animation_rejected_total\":" << tick.animationRejectedTotal
           << ",\"debug_checked\":" << boolText(debug.checked)
           << ",\"debug_ok\":" << boolText(debug.ok)
           << ",\"gpu_readback_lights\":" << debug.readbackLights
           << ",\"gpu_readback_point\":" << debug.readbackPoint
           << ",\"gpu_readback_directional\":"
           << debug.readbackDirectional
           << '}';
    return stream.str();
}

std::string formatBufferInitJson(
    const SyncOutcome& sync,
    const SceneSnapshot& snapshot,
    const RunConfig& config) {
    const LightCounts counts = snapshot.counts();
    auto stream = makeStream();
    stream << "{\"event\":\"buffer_init\""
           << ",\"sync_mode\":\"" << toString(config.syncMode)
           << "\",\"delay_frames\":" << config.delayFrames
           << ",\"version\":" << snapshot.version
           << ",\"active_lights\":" << counts.total
           << ",\"point_lights\":" << counts.point
           << ",\"directional_lights\":" << counts.directional
           << ",\"buffer_capacity\":" << sync.capacityLights
           << ",\"uploaded\":true"
           << ",\"used_bytes\":" << sync.usedBytes
           << '}';
    return stream.str();
}

std::string formatDebugErrorJson(
    std::uint32_t frame,
    const BufferValidation& validation) {
    auto stream = makeStream();
    stream << "{\"event\":\"debug_error\""
           << ",\"frame\":" << frame
           << ",\"business_code\":\"" << validation.code << '"'
           << ",\"expected_lights\":" << validation.expectedLights
           << ",\"actual_lights\":" << validation.actualLights
           << ",\"expected_point\":" << validation.expectedPoint
           << ",\"actual_point\":" << validation.actualPoint
           << ",\"expected_directional\":"
           << validation.expectedDirectional
           << ",\"actual_directional\":" << validation.actualDirectional
           << ",\"expected_version\":" << validation.expectedVersion
           << ",\"actual_version\":" << validation.actualVersion
           << '}';
    return stream.str();
}

std::string formatFatalJson(
    std::string_view businessCode,
    std::string_view message) {
    auto stream = makeStream();
    stream << "{\"event\":\"fatal\",\"business_code\":\""
           << escapeJson(businessCode) << "\",\"message\":\""
           << escapeJson(message) << "\"}";
    return stream.str();
}

}  // namespace lightcluster
