#pragma once

#include <stdexcept>
#include <string>

namespace vr {

enum class ErrorCode {
    None = 0,
    Usage,
    Io,
    MetaInvalid,
    RateMismatch,
    OutputUnwritable,
    StateMissing,
    StateCorrupt,
    Internal,
};

class AppError : public std::runtime_error {
public:
    AppError(ErrorCode code, std::string message)
        : std::runtime_error(std::move(message)), code_(code) {}

    ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

const char* error_code_name(ErrorCode code) noexcept;
int error_exit_code(ErrorCode code) noexcept;

}  // namespace vr

