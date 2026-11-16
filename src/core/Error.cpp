#include "core/Error.hpp"

#include <iostream>

namespace lcr {
namespace {

std::string escapeJson(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (const char c : value) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          out += "?";
        } else {
          out += c;
        }
    }
  }
  return out;
}

}  // namespace

const char* errorCodeName(const ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::InvalidArgument:
      return "E_INVALID_ARGUMENT";
    case ErrorCode::TextureGeneration:
      return "E_TEXTURE_GENERATION";
    case ErrorCode::ShaderCompile:
      return "E_SHADER_COMPILE";
    case ErrorCode::ShaderLink:
      return "E_SHADER_LINK";
    case ErrorCode::BoneIndex:
      return "E_BONE_INDEX";
    case ErrorCode::GpuInit:
      return "E_GPU_INIT";
    case ErrorCode::Io:
      return "E_IO";
    case ErrorCode::InternalInvariant:
      return "E_INTERNAL_INVARIANT";
  }
  return "E_UNKNOWN";
}

int errorExitCode(const ErrorCode code) noexcept {
  return static_cast<int>(code);
}

AppError::AppError(const ErrorCode code, std::string stage, std::string message)
    : std::runtime_error(message), code_(code), stage_(std::move(stage)) {}

void fail(const ErrorCode code, std::string stage, std::string message) {
  throw AppError(code, std::move(stage), std::move(message));
}

void emitErrorToStderr(const AppError& error) {
  std::cerr << "{\"error\":{\"code\":\"" << errorCodeName(error.code())
            << "\",\"exit_code\":" << errorExitCode(error.code())
            << ",\"stage\":\"" << escapeJson(error.stage())
            << "\",\"message\":\"" << escapeJson(error.what()) << "\"}}\n";
  std::cerr.flush();
}

}  // namespace lcr

