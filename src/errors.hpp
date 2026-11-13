#pragma once

#include <stdexcept>
#include <string>

namespace lfb {

enum class ExitCode : int {
    E_RUNTIME = 1,
    E_SCENE_PARSE = 2,
    E_SCENE_VALID = 3,
    E_OUTPUT_IO = 4,
    E_CACHE_IO = 5,
    E_USAGE = 64,
};

inline const char* exitCodeName(ExitCode code) {
    switch (code) {
        case ExitCode::E_RUNTIME: return "E_RUNTIME";
        case ExitCode::E_SCENE_PARSE: return "E_SCENE_PARSE";
        case ExitCode::E_SCENE_VALID: return "E_SCENE_VALID";
        case ExitCode::E_OUTPUT_IO: return "E_OUTPUT_IO";
        case ExitCode::E_CACHE_IO: return "E_CACHE_IO";
        case ExitCode::E_USAGE: return "E_USAGE";
    }
    return "E_RUNTIME";
}

class BusinessError : public std::runtime_error {
public:
    BusinessError(ExitCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    ExitCode code() const noexcept { return code_; }

private:
    ExitCode code_;
};

}  // namespace lfb
