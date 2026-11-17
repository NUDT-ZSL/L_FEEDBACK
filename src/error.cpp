#include "error.h"

#include <sstream>

namespace softraster {
namespace {

const char* errorName(ErrorCode code) {
    switch (code) {
        case ErrorCode::None: return "NONE";
        case ErrorCode::Io: return "IO_ERROR";
        case ErrorCode::JsonParse: return "JSON_PARSE_ERROR";
        case ErrorCode::MissingField: return "MISSING_FIELD";
        case ErrorCode::TypeMismatch: return "TYPE_MISMATCH";
        case ErrorCode::InvalidValue: return "INVALID_VALUE";
        case ErrorCode::NonFiniteFloat: return "NON_FINITE_FLOAT";
        case ErrorCode::IndexOutOfRange: return "INDEX_OUT_OF_RANGE";
        case ErrorCode::ShadowMapTooLarge: return "SHADOW_MAP_TOO_LARGE";
        case ErrorCode::DuplicateTransparentDepth: return "DUPLICATE_TRANSPARENT_DEPTH";
        case ErrorCode::MemoryLimitExceeded: return "MEMORY_LIMIT_EXCEEDED";
        case ErrorCode::OutputWrite: return "OUTPUT_WRITE_ERROR";
    }
    return "UNKNOWN";
}

std::string escapeJson(std::string_view text) {
    std::ostringstream out;
    for (const unsigned char ch : text) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20U) {
                    const char* hex = "0123456789abcdef";
                    out << "\\u00" << hex[(ch >> 4U) & 0x0fU] << hex[ch & 0x0fU];
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

}  // namespace

RenderError makeError(ErrorCode code, std::string message, std::string path) {
    return RenderError{code, errorName(code), std::move(message), std::move(path)};
}

std::string formatErrorJson(const RenderError& error) {
    std::ostringstream out;
    out << "{\"error\":{\"code\":" << static_cast<int>(error.code)
        << ",\"name\":\"" << error.name << "\",\"message\":\"" << escapeJson(error.message)
        << "\"";
    if (!error.path.empty()) {
        out << ",\"path\":\"" << escapeJson(error.path) << "\"";
    }
    out << "}}\n";
    return out.str();
}

}  // namespace softraster
