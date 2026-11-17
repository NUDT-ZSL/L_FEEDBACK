#pragma once

#include <stdexcept>
#include <string>

namespace sr {

enum class ExitCode : int {
    Ok = 0,
    Internal = 1,
    Usage = 2,
    OutputIo = 3,
    Context = 4,
    Shader = 5,
    Framebuffer = 6,
    Cascade = 7,
    Scene = 8,
};

class AppError final : public std::runtime_error {
public:
    AppError(ExitCode code, std::string symbol, const std::string& message)
        : std::runtime_error(message), code_(code), symbol_(std::move(symbol)) {}

    ExitCode code() const noexcept { return code_; }
    const std::string& symbol() const noexcept { return symbol_; }

private:
    ExitCode code_;
    std::string symbol_;
};

[[noreturn]] inline void fail(ExitCode code, const char* symbol, const std::string& message) {
    throw AppError(code, symbol, message);
}

} // namespace sr
