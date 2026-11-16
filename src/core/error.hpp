#pragma once

#include <stdexcept>
#include <string>

namespace uid166 {

enum class ErrorCode : int {
    ok = 0,
    usage = 1001,
    sceneMissing = 2001,
    sceneRead = 2002,
    sceneParse = 2003,
    sceneSchema = 2004,
    animationMissing = 2101,
    animationRead = 2102,
    animationCorrupt = 2103,
    animationSchema = 2104,
    deviceInit = 3001,
    swapChainInit = 3002,
    shaderCompile = 3003,
    resourceCreate = 3004,
    threadPool = 3005,
    outputWrite = 4001,
    internal = 5000,
};

class AppError final : public std::runtime_error {
public:
    AppError(ErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    [[nodiscard]] ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

[[noreturn]] void throwError(ErrorCode code, const std::string& message);
[[nodiscard]] const char* errorCodeName(ErrorCode code) noexcept;
void reportError(const AppError& error);

} // namespace uid166

