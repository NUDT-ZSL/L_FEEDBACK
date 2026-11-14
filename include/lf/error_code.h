#pragma once

#include <stdexcept>
#include <string>

namespace lf {

enum class ErrorCode : int {
  Ok = 0,

  E_RENDER_WINDOW_CREATE = 1001,
  E_RENDER_CONTEXT_CREATE = 1002,
  E_RENDER_GL_LOAD = 1003,
  E_SHADER_COMPILE = 1004,
  E_SHADER_LINK = 1005,
  E_TEXTURE_LOAD = 1006,
  E_FRAMEBUFFER_INCOMPLETE = 1007,

  E_CONFIG_INVALID = 1101,
  E_CONFIG_RELOAD_INVALID = 1102,

  E_SNAPSHOT_CORRUPT = 1201,
  E_SNAPSHOT_VERSION_UNSUPPORTED = 1202,
  E_SNAPSHOT_IO = 1203,

  E_RESOURCE_MISSING = 1301,
  E_MESH_GENERATION = 1302,
  E_ARGUMENT = 1401,
  E_INTERNAL = 9000
};

const char* error_code_name(ErrorCode code) noexcept;
int error_code_number(ErrorCode code) noexcept;

class AppError : public std::runtime_error {
 public:
  AppError(ErrorCode code, std::string message)
      : std::runtime_error(std::move(message)), code_(code) {}

  [[nodiscard]] ErrorCode code() const noexcept { return code_; }

 private:
  ErrorCode code_;
};

}  // namespace lf

