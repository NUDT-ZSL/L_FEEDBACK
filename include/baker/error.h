#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace baker {

enum class ErrorCode : std::uint32_t {
    kOk = 0,
    kUsage = 1,
    kMagic = 2,
    kVersion = 3,
    kLength = 4,
    kValue = 5,
    kIo = 6,
    kArgs = 7,
};

inline const char* error_name(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::kOk: return "OK";
    case ErrorCode::kUsage: return "ERR_USAGE";
    case ErrorCode::kMagic: return "ERR_MAGIC";
    case ErrorCode::kVersion: return "ERR_VERSION";
    case ErrorCode::kLength: return "ERR_LENGTH";
    case ErrorCode::kValue: return "ERR_VALUE";
    case ErrorCode::kIo: return "ERR_IO";
    case ErrorCode::kArgs: return "ERR_ARGS";
    }
    return "ERR_UNKNOWN";
}

class BakerError final : public std::runtime_error {
public:
    BakerError(ErrorCode code, std::string message)
        : std::runtime_error(std::move(message)), code_(code) {}

    [[nodiscard]] ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

class Interrupted final : public std::runtime_error {
public:
    Interrupted(int signal_number, std::string message)
        : std::runtime_error(std::move(message)), signal_number_(signal_number) {}

    [[nodiscard]] int signal_number() const noexcept { return signal_number_; }

private:
    int signal_number_;
};

}  // namespace baker
