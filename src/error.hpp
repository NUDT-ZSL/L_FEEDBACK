#pragma once

#include <string>

namespace skel {

// Stable numeric values are part of the command-line contract. Do not renumber
// existing entries when adding new diagnostics.
enum class ErrorCode : int {
    Ok = 0,
    OpenFailed = 10,
    ReadFailed = 11,
    InvalidMagic = 12,
    UnsupportedVersion = 13,
    Truncated = 14,
    InvalidCount = 15,
    InvalidNodeIndex = 16,
    InvalidParentIndex = 17,
    ParentCycle = 18,
    InvalidTrackNode = 19,
    NonMonotonicTime = 20,
    InvalidNumericValue = 21,
    ZeroQuaternion = 22,
    InvalidWeightCount = 23,
    InvalidWeight = 24,
    WeightSumZero = 25,
    InvalidBoneIndex = 26,
    InvalidTriangleIndex = 27,
    InvalidCompositionOrder = 28,
    InvalidJson = 29,
    ResourceTooLarge = 30,
    WriteFailed = 31,
    InvalidFrameRange = 32,
    InvalidArgument = 33,
    OpenGLUnavailable = 50,
    NaNProduced = 70,
};

struct Error {
    ErrorCode code = ErrorCode::Ok;
    std::string message;

    explicit operator bool() const { return code != ErrorCode::Ok; }
};

const char* errorCodeName(ErrorCode code);
int processExitCode(ErrorCode code);
Error makeError(ErrorCode code, std::string message);

} // namespace skel

