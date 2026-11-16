#include "core/error.hpp"

#include <iostream>

namespace uid166 {

void throwError(ErrorCode code, const std::string& message) {
    throw AppError(code, message);
}

const char* errorCodeName(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::ok: return "OK";
    case ErrorCode::usage: return "USAGE";
    case ErrorCode::sceneMissing: return "SCENE_MISSING";
    case ErrorCode::sceneRead: return "SCENE_READ";
    case ErrorCode::sceneParse: return "SCENE_PARSE";
    case ErrorCode::sceneSchema: return "SCENE_SCHEMA";
    case ErrorCode::animationMissing: return "ANIMATION_MISSING";
    case ErrorCode::animationRead: return "ANIMATION_READ";
    case ErrorCode::animationCorrupt: return "ANIMATION_CORRUPT";
    case ErrorCode::animationSchema: return "ANIMATION_SCHEMA";
    case ErrorCode::deviceInit: return "DEVICE_INIT";
    case ErrorCode::swapChainInit: return "SWAPCHAIN_INIT";
    case ErrorCode::shaderCompile: return "SHADER_COMPILE";
    case ErrorCode::resourceCreate: return "RESOURCE_CREATE";
    case ErrorCode::threadPool: return "THREAD_POOL";
    case ErrorCode::outputWrite: return "OUTPUT_WRITE";
    case ErrorCode::internal: return "INTERNAL";
    }
    return "UNKNOWN";
}

void reportError(const AppError& error) {
    std::cerr << "[ERROR] code=" << static_cast<int>(error.code())
              << " name=" << errorCodeName(error.code())
              << " message=" << error.what() << '\n';
}

} // namespace uid166

