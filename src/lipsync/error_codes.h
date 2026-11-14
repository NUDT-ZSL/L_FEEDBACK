#pragma once

#include <string>
#include <string_view>

namespace lipsync {

enum class ErrorCode {
    Ok = 0,
    E_IO = 1,              // 文件读写失败
    E_PARSE_JSON = 2,      // JSON 解析失败（含损坏的 JSON 文件）
    E_AUDIO_DECODE = 3,    // 音频解码/打开失败
    E_RESAMPLE = 4,        // 重采样或声道转换失败
    E_PHONEME_OVERFLOW = 5, // 音素区间超出音频时长 > 100ms
    E_CONFIG = 6,          // 配置文件非法
    E_INTERNAL = 7,        // 其他内部错误
};

inline std::string_view error_code_name(ErrorCode code) {
    switch (code) {
        case ErrorCode::Ok: return "OK";
        case ErrorCode::E_IO: return "E_IO";
        case ErrorCode::E_PARSE_JSON: return "E_PARSE_JSON";
        case ErrorCode::E_AUDIO_DECODE: return "E_AUDIO_DECODE";
        case ErrorCode::E_RESAMPLE: return "E_RESAMPLE";
        case ErrorCode::E_PHONEME_OVERFLOW: return "E_PHONEME_OVERFLOW";
        case ErrorCode::E_CONFIG: return "E_CONFIG";
        case ErrorCode::E_INTERNAL: return "E_INTERNAL";
    }
    return "E_INTERNAL";
}

// 可恢复的处理错误：携带错误码与上下文信息，但不会中断其他文件的处理。
struct ProcessError {
    ErrorCode code = ErrorCode::Ok;
    std::string message;

    ProcessError() = default;
    ProcessError(ErrorCode c, std::string msg) : code(c), message(std::move(msg)) {}

    explicit operator bool() const { return code != ErrorCode::Ok; }
};

} // namespace lipsync
