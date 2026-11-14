#include "lf/error_code.h"

namespace lf {

const char* error_code_name(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::Ok:
      return "E_OK";
    case ErrorCode::E_RENDER_WINDOW_CREATE:
      return "E_RENDER_WINDOW_CREATE";
    case ErrorCode::E_RENDER_CONTEXT_CREATE:
      return "E_RENDER_CONTEXT_CREATE";
    case ErrorCode::E_RENDER_GL_LOAD:
      return "E_RENDER_GL_LOAD";
    case ErrorCode::E_SHADER_COMPILE:
      return "E_SHADER_COMPILE";
    case ErrorCode::E_SHADER_LINK:
      return "E_SHADER_LINK";
    case ErrorCode::E_TEXTURE_LOAD:
      return "E_TEXTURE_LOAD";
    case ErrorCode::E_FRAMEBUFFER_INCOMPLETE:
      return "E_FRAMEBUFFER_INCOMPLETE";
    case ErrorCode::E_CONFIG_INVALID:
      return "E_CONFIG_INVALID";
    case ErrorCode::E_CONFIG_RELOAD_INVALID:
      return "E_CONFIG_RELOAD_INVALID";
    case ErrorCode::E_SNAPSHOT_CORRUPT:
      return "E_SNAPSHOT_CORRUPT";
    case ErrorCode::E_SNAPSHOT_VERSION_UNSUPPORTED:
      return "E_SNAPSHOT_VERSION_UNSUPPORTED";
    case ErrorCode::E_SNAPSHOT_IO:
      return "E_SNAPSHOT_IO";
    case ErrorCode::E_RESOURCE_MISSING:
      return "E_RESOURCE_MISSING";
    case ErrorCode::E_MESH_GENERATION:
      return "E_MESH_GENERATION";
    case ErrorCode::E_ARGUMENT:
      return "E_ARGUMENT";
    case ErrorCode::E_INTERNAL:
      return "E_INTERNAL";
  }
  return "E_INTERNAL";
}

int error_code_number(ErrorCode code) noexcept {
  return static_cast<int>(code);
}

}  // namespace lf

