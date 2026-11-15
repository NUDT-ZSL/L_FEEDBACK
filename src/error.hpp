#pragma once

#include <stdexcept>
#include <string>

namespace scene {

enum class ErrorType {
    Cli,
    Context,
    Config,
    UnknownStage,
    UnknownEffect,
    ShaderCompile,
    FramebufferIncomplete,
    FileIo,
    StateVersion,
    StateMismatch,
    GlRuntime,
    Internal
};

const char* errorTypeName(ErrorType type);
int errorExitCode(ErrorType type);

class AppError : public std::runtime_error {
public:
    AppError(ErrorType type, const std::string& message)
        : std::runtime_error(message), type_(type) {}

    ErrorType type() const noexcept { return type_; }

private:
    ErrorType type_;
};

} // namespace scene

