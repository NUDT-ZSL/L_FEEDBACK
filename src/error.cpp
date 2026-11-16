#include "error.hpp"

namespace skel {

const char* errorCodeName(ErrorCode code) {
    switch (code) {
    case ErrorCode::Ok: return "OK";
    case ErrorCode::OpenFailed: return "OPEN_FAILED";
    case ErrorCode::ReadFailed: return "READ_FAILED";
    case ErrorCode::InvalidMagic: return "INVALID_MAGIC";
    case ErrorCode::UnsupportedVersion: return "UNSUPPORTED_VERSION";
    case ErrorCode::Truncated: return "TRUNCATED";
    case ErrorCode::InvalidCount: return "INVALID_COUNT";
    case ErrorCode::InvalidNodeIndex: return "INVALID_NODE_INDEX";
    case ErrorCode::InvalidParentIndex: return "INVALID_PARENT_INDEX";
    case ErrorCode::ParentCycle: return "PARENT_CYCLE";
    case ErrorCode::InvalidTrackNode: return "INVALID_TRACK_NODE";
    case ErrorCode::NonMonotonicTime: return "NON_MONOTONIC_TIME";
    case ErrorCode::InvalidNumericValue: return "INVALID_NUMERIC_VALUE";
    case ErrorCode::ZeroQuaternion: return "ZERO_QUATERNION";
    case ErrorCode::InvalidWeightCount: return "INVALID_WEIGHT_COUNT";
    case ErrorCode::InvalidWeight: return "INVALID_WEIGHT";
    case ErrorCode::WeightSumZero: return "WEIGHT_SUM_ZERO";
    case ErrorCode::InvalidBoneIndex: return "INVALID_BONE_INDEX";
    case ErrorCode::InvalidTriangleIndex: return "INVALID_TRIANGLE_INDEX";
    case ErrorCode::InvalidCompositionOrder: return "INVALID_COMPOSITION_ORDER";
    case ErrorCode::InvalidJson: return "INVALID_JSON";
    case ErrorCode::ResourceTooLarge: return "RESOURCE_TOO_LARGE";
    case ErrorCode::WriteFailed: return "WRITE_FAILED";
    case ErrorCode::InvalidFrameRange: return "INVALID_FRAME_RANGE";
    case ErrorCode::InvalidArgument: return "INVALID_ARGUMENT";
    case ErrorCode::OpenGLUnavailable: return "OPENGL_UNAVAILABLE";
    case ErrorCode::NaNProduced: return "NAN_PRODUCED";
    }
    return "UNKNOWN_ERROR";
}

int processExitCode(ErrorCode code) {
    if (code == ErrorCode::Ok) {
        return 0;
    }
    return static_cast<int>(code);
}

Error makeError(ErrorCode code, std::string message) {
    return Error{code, std::move(message)};
}

} // namespace skel

