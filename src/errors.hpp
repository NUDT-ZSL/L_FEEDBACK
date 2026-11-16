#pragma once

#include <string>

namespace orv {

struct AppError {
  std::string code;
  std::string message;
  int exit_code = 1;
  std::string path;
  std::string extra_json;
};

inline AppError sceneError(std::string code, std::string message, std::string path = {}) {
  return AppError{std::move(code), std::move(message), 2, std::move(path), {}};
}

inline AppError renderError(std::string code, int exit_code, std::string message,
                            std::string path = {}) {
  return AppError{std::move(code), std::move(message), exit_code, std::move(path), {}};
}

inline AppError cliError(std::string code, std::string message) {
  return AppError{std::move(code), std::move(message), 5, {}, {}};
}

inline AppError ioError(std::string code, int exit_code, std::string message,
                        std::string path = {}) {
  return AppError{std::move(code), std::move(message), exit_code, std::move(path), {}};
}

}  // namespace orv
