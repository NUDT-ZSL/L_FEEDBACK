#pragma once

#include <stdexcept>
#include <string>

namespace automix {

enum class ErrorCode {
    Ok = 0,
    Usage,
    CfgSyntax,
    CfgMissing,
    CfgValue,
    FileMissing,
    FileFormat,
    FileCorrupt,
    OutputIo,
    Internal,
};

inline const char* errorCodeName(ErrorCode code) {
    switch (code) {
        case ErrorCode::Ok: return "E_OK";
        case ErrorCode::Usage: return "E_USAGE";
        case ErrorCode::CfgSyntax: return "E_CFG_SYNTAX";
        case ErrorCode::CfgMissing: return "E_CFG_MISSING";
        case ErrorCode::CfgValue: return "E_CFG_VALUE";
        case ErrorCode::FileMissing: return "E_FILE_MISSING";
        case ErrorCode::FileFormat: return "E_FILE_FORMAT";
        case ErrorCode::FileCorrupt: return "E_FILE_CORRUPT";
        case ErrorCode::OutputIo: return "E_OUTPUT_IO";
        case ErrorCode::Internal: return "E_INTERNAL";
    }
    return "E_INTERNAL";
}

inline int exitCodeFor(ErrorCode code) {
    switch (code) {
        case ErrorCode::Ok: return 0;
        case ErrorCode::Usage:
        case ErrorCode::CfgSyntax:
        case ErrorCode::CfgMissing:
        case ErrorCode::CfgValue:
            return 2;
        case ErrorCode::FileMissing:
        case ErrorCode::FileFormat:
        case ErrorCode::FileCorrupt:
            return 3;
        case ErrorCode::OutputIo:
            return 4;
        case ErrorCode::Internal:
            return 5;
    }
    return 5;
}

class AppError : public std::runtime_error {
public:
    AppError(ErrorCode code, std::string message)
        : std::runtime_error(std::move(message)), code_(code) {}

    ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

}  // namespace automix
