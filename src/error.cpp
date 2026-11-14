#include "error.h"

namespace vr {

const char* error_code_name(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::None: return "OK";
        case ErrorCode::Usage: return "E_USAGE";
        case ErrorCode::Io: return "E_IO";
        case ErrorCode::MetaInvalid: return "E_META_INVALID";
        case ErrorCode::RateMismatch: return "E_RATE_MISMATCH";
        case ErrorCode::OutputUnwritable: return "E_OUTPUT_UNWRITABLE";
        case ErrorCode::StateMissing: return "E_STATE_MISSING";
        case ErrorCode::StateCorrupt: return "E_STATE_CORRUPT";
        case ErrorCode::Internal: return "E_INTERNAL";
    }
    return "E_INTERNAL";
}

int error_exit_code(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::None: return 0;
        case ErrorCode::Usage: return 2;
        case ErrorCode::Io: return 3;
        case ErrorCode::MetaInvalid: return 4;
        case ErrorCode::RateMismatch: return 5;
        case ErrorCode::OutputUnwritable: return 6;
        case ErrorCode::StateMissing: return 7;
        case ErrorCode::StateCorrupt: return 8;
        case ErrorCode::Internal: return 9;
    }
    return 9;
}

}  // namespace vr

