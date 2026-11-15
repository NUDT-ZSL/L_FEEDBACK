#include "error.hpp"

namespace scene {

const char* errorTypeName(ErrorType type) {
    switch (type) {
    case ErrorType::Cli: return "cli_error";
    case ErrorType::Context: return "context_error";
    case ErrorType::Config: return "config_error";
    case ErrorType::UnknownStage: return "unknown_stage";
    case ErrorType::UnknownEffect: return "unknown_effect";
    case ErrorType::ShaderCompile: return "shader_compile";
    case ErrorType::FramebufferIncomplete: return "framebuffer_incomplete";
    case ErrorType::FileIo: return "io_error";
    case ErrorType::StateVersion: return "unsupported_version";
    case ErrorType::StateMismatch: return "state_restore_mismatch";
    case ErrorType::GlRuntime: return "gl_runtime_error";
    case ErrorType::Internal: return "internal_error";
    }
    return "unknown_error";
}

int errorExitCode(ErrorType type) {
    switch (type) {
    case ErrorType::Cli: return 2;
    case ErrorType::Config: return 3;
    case ErrorType::UnknownStage: return 4;
    case ErrorType::UnknownEffect: return 5;
    case ErrorType::ShaderCompile: return 6;
    case ErrorType::FramebufferIncomplete: return 7;
    case ErrorType::Context: return 8;
    case ErrorType::FileIo: return 9;
    case ErrorType::StateVersion: return 10;
    case ErrorType::StateMismatch: return 11;
    case ErrorType::GlRuntime: return 12;
    case ErrorType::Internal: return 13;
    }
    return 1;
}

} // namespace scene

