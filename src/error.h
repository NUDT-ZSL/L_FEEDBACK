#pragma once

#include <string>
#include <string_view>

namespace softraster {

enum class ErrorCode : int {
    None = 0,
    Io = 1,
    JsonParse = 2,
    MissingField = 3,
    TypeMismatch = 4,
    InvalidValue = 5,
    NonFiniteFloat = 6,
    IndexOutOfRange = 7,
    ShadowMapTooLarge = 8,
    DuplicateTransparentDepth = 9,
    MemoryLimitExceeded = 10,
    OutputWrite = 11,
};

struct RenderError {
    ErrorCode code = ErrorCode::None;
    std::string name;
    std::string message;
    std::string path;
};

[[nodiscard]] RenderError makeError(ErrorCode code, std::string message, std::string path = {});
[[nodiscard]] std::string formatErrorJson(const RenderError& error);

}  // namespace softraster
