#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace vlr {

enum class ErrorCode {
    None = 0,
    Schema = 2,
    Shader = 3,
    Allocation = 4,
    Descriptor = 5,
    DeviceLost = 6,
    Io = 7,
    Vulkan = 8,
    Unsupported = 9,
    NonDeterministic = 10,
    Reference = 11,
    Recovery = 12,
    Usage = 13
};

const char* errorCodeName(ErrorCode code) noexcept;
int errorExitCode(ErrorCode code) noexcept;

class AppError final : public std::runtime_error {
public:
    AppError(ErrorCode code, std::string message)
        : std::runtime_error(std::move(message)), code_(code) {}

    [[nodiscard]] ErrorCode code() const noexcept { return code_; }
    [[nodiscard]] int exitCode() const noexcept { return errorExitCode(code_); }

private:
    ErrorCode code_;
};

[[noreturn]] inline void fail(ErrorCode code, const std::string& message) {
    throw AppError(code, message);
}

} // namespace vlr
