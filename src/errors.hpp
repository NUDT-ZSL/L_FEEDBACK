#pragma once

#include <stdexcept>
#include <string>

namespace terrain {

struct AppError : std::runtime_error {
  std::string code;
  int exitCode;

  AppError(std::string errorCode, std::string message, int processExitCode)
      : std::runtime_error(std::move(message)),
        code(std::move(errorCode)),
        exitCode(processExitCode) {}
};

inline AppError usageError(const std::string& message) {
  return AppError("INVALID_ARGUMENT", message, 2);
}

inline AppError businessError(const std::string& code, const std::string& message) {
  return AppError(code, message, 1);
}

}  // namespace terrain
