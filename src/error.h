#pragma once

#include <exception>
#include <string>

namespace df {

enum class ErrorCode {
    E_CONFIG_PARSE = 2,
    E_GBUFFER_FAIL = 3,
    E_SHADER_COMPILE = 4,
    E_IO = 5,
    E_GL_CONTEXT = 6,
    E_ARGS = 7,
    E_RENDER_FAIL = 8,
};

inline const char* errorCodeName(ErrorCode code) {
    switch (code) {
    case ErrorCode::E_CONFIG_PARSE: return "E_CONFIG_PARSE";
    case ErrorCode::E_GBUFFER_FAIL: return "E_GBUFFER_FAIL";
    case ErrorCode::E_SHADER_COMPILE: return "E_SHADER_COMPILE";
    case ErrorCode::E_IO: return "E_IO";
    case ErrorCode::E_GL_CONTEXT: return "E_GL_CONTEXT";
    case ErrorCode::E_ARGS: return "E_ARGS";
    case ErrorCode::E_RENDER_FAIL: return "E_RENDER_FAIL";
    }
    return "E_UNKNOWN";
}

inline int exitCode(ErrorCode code) {
    return static_cast<int>(code);
}

class AppError : public std::exception {
public:
    AppError(ErrorCode code, std::string detail)
        : code_(code), detail_(std::move(detail)) {}

    const char* what() const noexcept override { return detail_.c_str(); }
    ErrorCode code() const { return code_; }

private:
    ErrorCode code_;
    std::string detail_;
};

} // namespace df

