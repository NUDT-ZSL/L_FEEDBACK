#include "errors.hpp"

namespace vlr {

const char* errorCodeName(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::None: return "OK";
    case ErrorCode::Schema: return "E_SCHEMA";
    case ErrorCode::Shader: return "E_SHADER";
    case ErrorCode::Allocation: return "E_ALLOC";
    case ErrorCode::Descriptor: return "E_DESC";
    case ErrorCode::DeviceLost: return "E_DEVICE_LOST";
    case ErrorCode::Io: return "E_IO";
    case ErrorCode::Vulkan: return "E_VULKAN";
    case ErrorCode::Unsupported: return "E_UNSUPPORTED";
    case ErrorCode::NonDeterministic: return "E_NONDETERMINISTIC";
    case ErrorCode::Reference: return "E_REFERENCE";
    case ErrorCode::Recovery: return "E_RECOVERY";
    case ErrorCode::Usage: return "E_USAGE";
    }
    return "E_UNKNOWN";
}

int errorExitCode(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::None: return 0;
    case ErrorCode::Schema: return 2;
    case ErrorCode::Shader: return 3;
    case ErrorCode::Allocation: return 4;
    case ErrorCode::Descriptor: return 5;
    case ErrorCode::DeviceLost: return 6;
    case ErrorCode::Io: return 7;
    case ErrorCode::Vulkan: return 8;
    case ErrorCode::Unsupported: return 9;
    case ErrorCode::NonDeterministic: return 10;
    case ErrorCode::Reference: return 11;
    case ErrorCode::Recovery: return 12;
    case ErrorCode::Usage: return 13;
    }
    return 1;
}

} // namespace vlr
