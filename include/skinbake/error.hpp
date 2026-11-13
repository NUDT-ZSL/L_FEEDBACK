#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace skinbake {

enum class ErrorCode : std::uint32_t {
    None = 0,

    CurveTimeNotIncreasing = 1001,
    BezierDenominatorZero = 1002,
    CurveEmpty = 1003,
    CurveNonFinite = 1004,
    TangentInvalid = 1005,

    SkeletonParentInvalid = 2001,
    SkeletonCycle = 2002,
    VertexBoneIndexInvalid = 2003,
    VertexTooManyInfluences = 2004,
    WeightSumZero = 2005,
    WeightSumNotOne = 2006,
    WeightInvalid = 2007,
    TransformNonFinite = 2008,
    MeshInvalid = 2009,

    ClipNotFound = 3001,
    ClipDurationInvalid = 3002,
    TransitionInvalid = 3003,
    StateMachineInvalid = 3004,

    SnapshotMagicMismatch = 4001,
    SnapshotTruncated = 4002,
    SnapshotVersionMismatch = 4003,
    SnapshotChecksumMismatch = 4004,
    SnapshotIoError = 4005,
    SnapshotDimensionMismatch = 4006,
    SnapshotDataInvalid = 4007,
    SnapshotFingerprintMismatch = 4008,

    InvalidArgument = 5001,
    InvalidScenario = 5002,
    InternalError = 5003
};

const char* error_code_name(ErrorCode code) noexcept;
std::uint32_t error_code_number(ErrorCode code) noexcept;
int exit_code_for(ErrorCode code) noexcept;

class BusinessError : public std::runtime_error {
public:
    BusinessError(ErrorCode code, std::string message);
    BusinessError(ErrorCode code, std::string message,
                  std::vector<std::pair<std::string, std::string>> details);

    ErrorCode code() const noexcept { return code_; }
    const std::string& message() const noexcept { return message_; }
    const std::vector<std::pair<std::string, std::string>>& details() const noexcept {
        return details_;
    }
    void add_detail(std::string key, std::string value);
    std::string to_json() const;

private:
    ErrorCode code_;
    std::string message_;
    std::vector<std::pair<std::string, std::string>> details_;
};

[[noreturn]] void throw_error(
    ErrorCode code, const std::string& message,
    std::vector<std::pair<std::string, std::string>> details = {});

}

