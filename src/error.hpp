#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>

namespace demo {

enum class ErrorCode : int {
    Ok = 0,
    GlfwInitializationFailed = 1001,
    WindowOrContextCreationFailed = 1002,
    OpenGl45Unavailable = 1003,
    OpenGlLoaderFailed = 1004,
    ShaderCompileFallback = 2001,
    ShaderLinkFallback = 2002,
    SafeShaderFailed = 2003,
    ShaderInterfaceMismatch = 2004,
    SceneResourceInvalid = 3001,
    GpuBufferUploadFailed = 3002,
    UniformBufferSyncFailed = 3003,
    MeshDynamicUpdateFailed = 3004,
    ScreenshotFailed = 3005,
    PersistentGlError = 1010,
    CommandLineError = 9001,
    UnexpectedFailure = 9002,
    SelfTestFailed = 9003
};

inline int exitCode(ErrorCode code) {
    return static_cast<int>(code);
}

inline const char* errorCodeName(ErrorCode code) {
    switch (code) {
        case ErrorCode::Ok:
            return "OK";
        case ErrorCode::GlfwInitializationFailed:
            return "GLFW_INITIALIZATION_FAILED";
        case ErrorCode::WindowOrContextCreationFailed:
            return "WINDOW_OR_CONTEXT_CREATION_FAILED";
        case ErrorCode::OpenGl45Unavailable:
            return "OPENGL_45_UNAVAILABLE";
        case ErrorCode::OpenGlLoaderFailed:
            return "OPENGL_LOADER_FAILED";
        case ErrorCode::ShaderCompileFallback:
            return "SHADER_COMPILE_FALLBACK";
        case ErrorCode::ShaderLinkFallback:
            return "SHADER_LINK_FALLBACK";
        case ErrorCode::SafeShaderFailed:
            return "SAFE_SHADER_FAILED";
        case ErrorCode::ShaderInterfaceMismatch:
            return "SHADER_INTERFACE_MISMATCH";
        case ErrorCode::SceneResourceInvalid:
            return "SCENE_RESOURCE_INVALID";
        case ErrorCode::GpuBufferUploadFailed:
            return "GPU_BUFFER_UPLOAD_FAILED";
        case ErrorCode::UniformBufferSyncFailed:
            return "UNIFORM_BUFFER_SYNC_FAILED";
        case ErrorCode::MeshDynamicUpdateFailed:
            return "MESH_DYNAMIC_UPDATE_FAILED";
        case ErrorCode::ScreenshotFailed:
            return "SCREENSHOT_FAILED";
        case ErrorCode::PersistentGlError:
            return "PERSISTENT_GL_ERROR";
        case ErrorCode::CommandLineError:
            return "COMMAND_LINE_ERROR";
        case ErrorCode::UnexpectedFailure:
            return "UNEXPECTED_FAILURE";
        case ErrorCode::SelfTestFailed:
            return "SELF_TEST_FAILED";
    }
    return "UNKNOWN_ERROR";
}

class AppError : public std::runtime_error {
public:
    AppError(ErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    ErrorCode code() const noexcept {
        return code_;
    }

    int exitCode() const noexcept {
        return demo::exitCode(code_);
    }

private:
    ErrorCode code_;
};

inline void reportError(const AppError& error) {
    // stderr is intentional: shader and context failures must be visible in CI logs.
    std::fprintf(stderr, "[ERROR][E%04d/%s] %s\n",
                 error.exitCode(), errorCodeName(error.code()), error.what());
}

inline void reportUnexpectedError(const std::exception& error) {
    std::fprintf(stderr, "[ERROR][E%04d/%s] %s\n",
                 exitCode(ErrorCode::UnexpectedFailure),
                 errorCodeName(ErrorCode::UnexpectedFailure),
                 error.what());
}

} // namespace demo
