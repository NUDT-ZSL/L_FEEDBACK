#pragma once

#include <stdexcept>
#include <string>

namespace preview {

enum class ErrorCode {
    Config,
    ObjMissing,
    IndexInvalid,
    MatMissing,
    ImageDecode,
    OutputWrite,
    Usage,
    Internal,
};

class AppError final : public std::runtime_error {
public:
    AppError(ErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

inline const char* errorCodeName(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::Config:
        return "E_CONFIG";
    case ErrorCode::ObjMissing:
        return "E_OBJ_MISSING";
    case ErrorCode::IndexInvalid:
        return "E_INDEX_INVALID";
    case ErrorCode::MatMissing:
        return "E_MAT_MISSING";
    case ErrorCode::ImageDecode:
        return "E_IMAGE_DECODE";
    case ErrorCode::OutputWrite:
        return "E_OUTPUT_WRITE";
    case ErrorCode::Usage:
        return "E_USAGE";
    case ErrorCode::Internal:
        return "E_INTERNAL";
    }
    return "E_INTERNAL";
}

inline int exitCodeFor(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::Config:
        return 10;
    case ErrorCode::ObjMissing:
        return 11;
    case ErrorCode::IndexInvalid:
        return 12;
    case ErrorCode::MatMissing:
        return 13;
    case ErrorCode::ImageDecode:
        return 14;
    case ErrorCode::OutputWrite:
        return 15;
    case ErrorCode::Usage:
        return 2;
    case ErrorCode::Internal:
        return 16;
    }
    return 16;
}

} // namespace preview
