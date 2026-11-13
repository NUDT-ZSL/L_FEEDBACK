#include "skinbake/error.hpp"

#include "skinbake/json.hpp"

#include <sstream>

namespace skinbake {

const char* error_code_name(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::None: return "NONE";
        case ErrorCode::CurveTimeNotIncreasing: return "CURVE_TIME_NOT_INCREASING";
        case ErrorCode::BezierDenominatorZero: return "BEZIER_DENOMINATOR_ZERO";
        case ErrorCode::CurveEmpty: return "CURVE_EMPTY";
        case ErrorCode::CurveNonFinite: return "CURVE_NON_FINITE";
        case ErrorCode::TangentInvalid: return "TANGENT_INVALID";
        case ErrorCode::SkeletonParentInvalid: return "SKELETON_PARENT_INVALID";
        case ErrorCode::SkeletonCycle: return "SKELETON_CYCLE";
        case ErrorCode::VertexBoneIndexInvalid: return "VERTEX_BONE_INDEX_INVALID";
        case ErrorCode::VertexTooManyInfluences: return "VERTEX_TOO_MANY_INFLUENCES";
        case ErrorCode::WeightSumZero: return "WEIGHT_SUM_ZERO";
        case ErrorCode::WeightSumNotOne: return "WEIGHT_SUM_NOT_ONE";
        case ErrorCode::WeightInvalid: return "WEIGHT_INVALID";
        case ErrorCode::TransformNonFinite: return "TRANSFORM_NON_FINITE";
        case ErrorCode::MeshInvalid: return "MESH_INVALID";
        case ErrorCode::ClipNotFound: return "CLIP_NOT_FOUND";
        case ErrorCode::ClipDurationInvalid: return "CLIP_DURATION_INVALID";
        case ErrorCode::TransitionInvalid: return "TRANSITION_INVALID";
        case ErrorCode::StateMachineInvalid: return "STATE_MACHINE_INVALID";
        case ErrorCode::SnapshotMagicMismatch: return "SNAPSHOT_MAGIC_MISMATCH";
        case ErrorCode::SnapshotTruncated: return "SNAPSHOT_TRUNCATED";
        case ErrorCode::SnapshotVersionMismatch: return "SNAPSHOT_VERSION_MISMATCH";
        case ErrorCode::SnapshotChecksumMismatch: return "SNAPSHOT_CHECKSUM_MISMATCH";
        case ErrorCode::SnapshotIoError: return "SNAPSHOT_IO_ERROR";
        case ErrorCode::SnapshotDimensionMismatch: return "SNAPSHOT_DIMENSION_MISMATCH";
        case ErrorCode::SnapshotDataInvalid: return "SNAPSHOT_DATA_INVALID";
        case ErrorCode::SnapshotFingerprintMismatch: return "SNAPSHOT_FINGERPRINT_MISMATCH";
        case ErrorCode::InvalidArgument: return "INVALID_ARGUMENT";
        case ErrorCode::InvalidScenario: return "INVALID_SCENARIO";
        case ErrorCode::InternalError: return "INTERNAL_ERROR";
    }
    return "INTERNAL_ERROR";
}

std::uint32_t error_code_number(ErrorCode code) noexcept {
    return static_cast<std::uint32_t>(code);
}

int exit_code_for(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::None:
            return 0;
        case ErrorCode::SnapshotChecksumMismatch:
        case ErrorCode::SnapshotVersionMismatch:
        case ErrorCode::SnapshotMagicMismatch:
        case ErrorCode::SnapshotTruncated:
        case ErrorCode::SnapshotDimensionMismatch:
        case ErrorCode::SnapshotDataInvalid:
        case ErrorCode::SnapshotFingerprintMismatch:
            return 4;
        case ErrorCode::InternalError:
            return 5;
        default:
            return 2;
    }
}

BusinessError::BusinessError(ErrorCode code, std::string message)
    : std::runtime_error(message), code_(code), message_(std::move(message)) {}

BusinessError::BusinessError(
    ErrorCode code, std::string message,
    std::vector<std::pair<std::string, std::string>> details)
    : std::runtime_error(message),
      code_(code),
      message_(std::move(message)),
      details_(std::move(details)) {}

void BusinessError::add_detail(std::string key, std::string value) {
    details_.emplace_back(std::move(key), std::move(value));
}

std::string BusinessError::to_json() const {
    std::vector<JsonField> fields;
    fields.emplace_back("ok", json_bool(false));
    fields.emplace_back("code", json_string(error_code_name(code_)));
    fields.emplace_back("codeNumber", json_u64(error_code_number(code_)));
    fields.emplace_back("message", json_string(message_));
    for (const auto& detail : details_) {
        fields.emplace_back(detail.first, json_string(detail.second));
    }
    return json_object(fields);
}

[[noreturn]] void throw_error(
    ErrorCode code, const std::string& message,
    std::vector<std::pair<std::string, std::string>> details) {
    throw BusinessError(code, message, std::move(details));
}

}

