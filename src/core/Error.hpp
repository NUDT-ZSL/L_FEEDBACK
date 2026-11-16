#pragma once

#include <stdexcept>
#include <string>

namespace lcr {

enum class ErrorCode : int {
  InvalidArgument = 10,
  TextureGeneration = 20,
  ShaderCompile = 30,
  ShaderLink = 31,
  BoneIndex = 40,
  GpuInit = 50,
  Io = 60,
  InternalInvariant = 70,
};

const char* errorCodeName(ErrorCode code) noexcept;
int errorExitCode(ErrorCode code) noexcept;

class AppError final : public std::runtime_error {
 public:
  AppError(ErrorCode code, std::string stage, std::string message);

  ErrorCode code() const noexcept { return code_; }
  const std::string& stage() const noexcept { return stage_; }

 private:
  ErrorCode code_;
  std::string stage_;
};

[[noreturn]] void fail(ErrorCode code, std::string stage, std::string message);

// Emits one JSON object followed by a newline. No rendering state is mutated here.
void emitErrorToStderr(const AppError& error);

}  // namespace lcr

